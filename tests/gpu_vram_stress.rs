//! GPU VRAM stress tests.
//!
//! Tests what happens when GPU memory is exhausted or under heavy pressure.
//! Validates that the library handles CUDA out-of-memory conditions gracefully
//! rather than panicking.
//!
//! These tests require a CUDA-capable GPU. They are skipped automatically
//! when no GPU is available (i.e., `CudaContext::new(0)` fails).
//!
//! **IMPORTANT:** These tests MUST be run single-threaded because they all
//! compete for the same GPU VRAM:
//!
//! ```sh
//! cargo test --test gpu_vram_stress -- --ignored --test-threads=1
//! ```
//!
//! # CUDA Async Allocator Caveat
//!
//! CUDA's `cuMemAllocAsync` (used by cudarc) may leave the stream's memory
//! pool in a permanently degraded state after OOM. Once the pool is exhausted,
//! even freeing all allocations and synchronizing the stream may not restore
//! it.
//!
//! Because of this, all OOM-inducing scenarios are consolidated into a single
//! test function (`test_z_oom_scenarios`) that runs last. This avoids the
//! problem of one test poisoning the allocator for subsequent tests.

mod test_utils;

use kvik_rs::{CompatMode, FileHandle};

/// Helper: attempt to create a CUDA context, returning None if unavailable.
fn try_cuda_context() -> Option<std::sync::Arc<cudarc::driver::CudaContext>> {
    cudarc::driver::CudaContext::new(0).ok()
}

/// Helper: query free and total VRAM in bytes.
///
/// Binds the context to the current thread, then calls the driver API.
/// Panics if the query fails — callers should only skip when no GPU exists.
fn mem_info(ctx: &std::sync::Arc<cudarc::driver::CudaContext>) -> (usize, usize) {
    ctx.bind_to_thread()
        .expect("failed to bind CUDA context to thread");
    cudarc::driver::result::mem_get_info().expect("failed to query VRAM info")
}

// =====================================================================
// Non-destructive tests (don't exhaust VRAM, safe to run in any order)
// =====================================================================

/// GDS write should succeed even when most VRAM is consumed by other
/// allocations, as long as the buffer being written is valid.
#[test]
#[ignore = "requires single-threaded execution: cargo test --test gpu_vram_stress -- --ignored --test-threads=1"]
fn test_a_gds_io_under_memory_pressure() {
    let ctx = match try_cuda_context() {
        Some(c) => c,
        None => {
            eprintln!("SKIP: no CUDA GPU");
            return;
        }
    };

    let (free, _total) = mem_info(&ctx);
    let stream = ctx.default_stream();

    // Allocate I/O buffers first, before filling VRAM.
    let io_size = 1024 * 1024; // 1 MiB
    let data = test_utils::gen_data(io_size);

    let dev_buf = stream
        .clone_htod(&data)
        .expect("failed to allocate I/O buffer");
    let mut dev_read = stream
        .alloc_zeros::<u8>(io_size)
        .expect("failed to allocate read buffer");

    // Synchronize so the I/O buffers are fully materialized before we
    // fill the remaining VRAM.
    let _ = stream.synchronize();

    // Fill ~80% of remaining free VRAM with dummy allocations.
    let chunk_size = 128 * 1024 * 1024;
    let remaining_free = free.saturating_sub(2 * io_size);
    let target_fill = (remaining_free as f64 * 0.80) as usize;
    let num_fill_chunks = target_fill / chunk_size;
    let mut fill_allocations = Vec::new();

    for _ in 0..num_fill_chunks {
        match stream.alloc_zeros::<u8>(chunk_size) {
            Ok(slice) => fill_allocations.push(slice),
            Err(_) => break,
        }
    }

    let tmp = tempfile::NamedTempFile::new().expect("failed to create temp file");
    let handle =
        FileHandle::open(tmp.path(), "w+", 0o644, CompatMode::Auto).expect("failed to open file");

    if !handle.is_gds_available() {
        eprintln!("SKIP: GDS not available");
        return;
    }

    // Write under pressure.
    let written = handle
        .write(&dev_buf, io_size, 0, 0)
        .expect("GDS write failed under memory pressure");
    assert_eq!(written, io_size);

    // Read back under pressure.
    let read = handle
        .read(&mut dev_read, io_size, 0, 0)
        .expect("GDS read failed under memory pressure");
    assert_eq!(read, io_size);

    // Verify data correctness.
    let result = stream.clone_dtoh(&dev_read).expect("clone_dtoh failed");
    assert_eq!(
        result, data,
        "data mismatch after I/O under memory pressure"
    );
}

/// Verify that concurrent GDS I/O operations from multiple threads don't
/// corrupt data under VRAM pressure.
#[test]
#[ignore = "requires single-threaded execution: cargo test --test gpu_vram_stress -- --ignored --test-threads=1"]
fn test_b_concurrent_gds_io_under_pressure() {
    let ctx = match try_cuda_context() {
        Some(c) => c,
        None => {
            eprintln!("SKIP: no CUDA GPU");
            return;
        }
    };

    let stream = ctx.default_stream();

    // Fill ~30% of VRAM to create pressure (not too much — concurrent threads
    // each need their own I/O buffers).
    let (free, _total) = mem_info(&ctx);

    let chunk_size = 128 * 1024 * 1024;
    let target_fill = (free as f64 * 0.30) as usize;
    let num_fill = target_fill / chunk_size;
    let mut fill_allocs = Vec::new();
    for _ in 0..num_fill {
        match stream.alloc_zeros::<u8>(chunk_size) {
            Ok(s) => fill_allocs.push(s),
            Err(_) => break,
        }
    }

    // Synchronize so fill allocations are materialized.
    let _ = stream.synchronize();

    // Spawn multiple threads, each doing GDS write+read with its own buffer.
    let io_size = 2 * 1024 * 1024; // 2 MiB per thread
    let num_threads = 4;

    let tmp_dir = tempfile::tempdir().expect("failed to create temp dir");
    let errors: Vec<Option<String>> = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for tid in 0..num_threads {
            let ctx_ref = &ctx;
            let tmp_dir_ref = &tmp_dir;
            handles.push(scope.spawn(move || {
                let stream = ctx_ref.default_stream();
                let data = test_utils::gen_data(io_size);

                let dev_buf = match stream.clone_htod(&data) {
                    Ok(b) => b,
                    Err(e) => return Some(format!("thread {tid}: clone_htod failed: {e}")),
                };

                let path = tmp_dir_ref.path().join(format!("thread_{tid}.bin"));
                let handle = match FileHandle::open(&path, "w+", 0o644, CompatMode::Auto) {
                    Ok(h) => h,
                    Err(e) => return Some(format!("thread {tid}: open failed: {e}")),
                };

                if !handle.is_gds_available() {
                    return Some(format!("thread {tid}: GDS not available"));
                }

                if let Err(e) = handle.write(&dev_buf, io_size, 0, 0) {
                    return Some(format!("thread {tid}: write failed: {e}"));
                }

                let mut dev_read = match stream.alloc_zeros::<u8>(io_size) {
                    Ok(b) => b,
                    Err(e) => return Some(format!("thread {tid}: alloc read buf failed: {e}")),
                };

                if let Err(e) = handle.read(&mut dev_read, io_size, 0, 0) {
                    return Some(format!("thread {tid}: read failed: {e}"));
                }

                let result = match stream.clone_dtoh(&dev_read) {
                    Ok(r) => r,
                    Err(e) => return Some(format!("thread {tid}: clone_dtoh failed: {e}")),
                };

                if result != data {
                    return Some(format!("thread {tid}: data mismatch"));
                }

                None // success
            }));
        }

        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // Check if any thread reported GDS unavailable (then skip).
    let gds_unavailable = errors.iter().any(|e| {
        e.as_deref()
            .is_some_and(|s| s.contains("GDS not available"))
    });
    if gds_unavailable {
        eprintln!("SKIP: GDS not available for concurrent test");
        return;
    }

    // All threads should succeed.
    for (tid, err) in errors.iter().enumerate() {
        assert!(
            err.is_none(),
            "thread {tid} failed: {}",
            err.as_deref().unwrap_or("")
        );
    }
}

/// Many small allocations and frees to check for VRAM fragmentation issues.
/// Uses only ~50% of VRAM to stay non-destructive.
#[test]
#[ignore = "requires single-threaded execution: cargo test --test gpu_vram_stress -- --ignored --test-threads=1"]
fn test_c_vram_fragmentation_stress() {
    let ctx = match try_cuda_context() {
        Some(c) => c,
        None => {
            eprintln!("SKIP: no CUDA GPU");
            return;
        }
    };

    let (free, _total) = mem_info(&ctx);
    let stream = ctx.default_stream();

    // Allocate many small buffers (up to ~50% of free VRAM), free every other
    // one, then try a larger allocation that requires coalesced free space.
    let small_size = 4 * 1024 * 1024; // 4 MiB
    let max_allocs = (free / 2) / small_size;
    let num_allocs = max_allocs.min(64);
    let mut allocations: Vec<Option<cudarc::driver::CudaSlice<u8>>> = Vec::new();

    for _ in 0..num_allocs {
        match stream.alloc_zeros::<u8>(small_size) {
            Ok(slice) => allocations.push(Some(slice)),
            Err(_) => break,
        }
    }

    let actual_allocs = allocations.len();
    assert!(
        actual_allocs >= 4,
        "could only allocate {} buffers, need at least 4",
        actual_allocs
    );

    // Free every other allocation to create fragmentation.
    let mut freed_count = 0;
    for i in (0..allocations.len()).step_by(2) {
        allocations[i] = None;
        freed_count += 1;
    }

    // Synchronize to ensure async frees complete.
    let _ = stream.synchronize();

    // Try to allocate a buffer that's 2x the small size — the allocator may
    // need to handle fragmented free space.
    let medium_size = small_size * 2;
    let medium_result = stream.alloc_zeros::<u8>(medium_size);

    // We don't assert success here since fragmentation behavior depends on
    // the CUDA allocator implementation. We just verify no panic.
    if medium_result.is_ok() {
        // Verify we can use it for I/O.
        let data = test_utils::gen_data(medium_size);
        let dev_buf = stream.clone_htod(&data).expect("clone_htod for medium");

        let tmp = tempfile::NamedTempFile::new().expect("failed to create temp file");
        let handle = FileHandle::open(tmp.path(), "w+", 0o644, CompatMode::Auto)
            .expect("failed to open file");

        if handle.is_gds_available() {
            let written = handle
                .write(&dev_buf, medium_size, 0, 0)
                .expect("GDS write after fragmentation");
            assert_eq!(written, medium_size);
        }
    }

    // Clean up remaining allocations.
    allocations.clear();
    let _ = stream.synchronize();

    // After full cleanup, should be able to allocate again.
    let post_cleanup = stream.alloc_zeros::<u8>(small_size);
    assert!(
        post_cleanup.is_ok(),
        "allocation after full cleanup should succeed: {:?}",
        post_cleanup.err()
    );

    eprintln!(
        "fragmentation test: allocated {actual_allocs} x {}MiB, freed {freed_count}, \
         medium alloc {}",
        small_size / (1024 * 1024),
        if medium_result.is_ok() {
            "succeeded"
        } else {
            "failed (expected under fragmentation)"
        }
    );
}

// =====================================================================
// Destructive test (exhausts VRAM, must run last)
//
// All OOM scenarios are consolidated into a single test because CUDA's
// async allocator may become permanently poisoned after OOM. Running
// separate tests would cause cascading failures from the first OOM test
// poisoning the allocator for all subsequent tests.
// =====================================================================

/// Exhausts GPU VRAM in multiple scenarios and validates that cudarc and
/// kvik-rs return errors (not panics) for all OOM conditions.
///
/// Scenarios tested (in order):
/// 1. GDS I/O with a pre-allocated buffer while remaining VRAM is full
/// 2. Allocating more VRAM than free should return an error
/// 3. Incremental allocation until OOM returns an error
/// 4. clone_htod with exhausted VRAM returns an error
///
/// After any of these scenarios, the CUDA async allocator may be permanently
/// poisoned for this process.
#[test]
#[ignore = "requires single-threaded execution: cargo test --test gpu_vram_stress -- --ignored --test-threads=1"]
fn test_z_oom_scenarios() {
    let ctx = match try_cuda_context() {
        Some(c) => c,
        None => {
            eprintln!("SKIP: no CUDA GPU");
            return;
        }
    };

    let (free, _total) = mem_info(&ctx);
    let stream = ctx.default_stream();

    eprintln!("VRAM: {} MiB free, {} MiB total", free / (1024 * 1024), _total / (1024 * 1024));

    // ------------------------------------------------------------------
    // Scenario 1: GDS I/O with pre-allocated buffer while VRAM is ~95% full
    //
    // Uses controlled fill (% of free) to avoid triggering OOM, which
    // would permanently poison the CUDA async allocator. Tests that I/O
    // works under extreme pressure.
    //
    // THIS MUST RUN BEFORE ANY OOM-TRIGGERING SCENARIO.
    // ------------------------------------------------------------------
    eprintln!("\n--- Scenario 1: GDS I/O under ~95% VRAM pressure ---");
    {
        let io_size = 4 * 1024 * 1024; // 4 MiB
        let data = test_utils::gen_data(io_size);
        let dev_buf = stream
            .clone_htod(&data)
            .expect("failed to allocate I/O buffer for scenario 1");
        let _ = stream.synchronize();

        // Fill ~95% of remaining free VRAM (after I/O buffer allocation).
        let chunk_size = 128 * 1024 * 1024;
        let remaining = free.saturating_sub(io_size);
        let target_fill = (remaining as f64 * 0.95) as usize;
        let num_chunks = target_fill / chunk_size;
        let mut fill_allocs = Vec::new();
        for _ in 0..num_chunks {
            match stream.alloc_zeros::<u8>(chunk_size) {
                Ok(slice) => fill_allocs.push(slice),
                Err(_) => break,
            }
        }
        eprintln!(
            "filled {} MiB of VRAM ({} chunks)",
            fill_allocs.len() * chunk_size / (1024 * 1024),
            fill_allocs.len()
        );

        let tmp = tempfile::NamedTempFile::new().expect("failed to create temp file");
        let handle = FileHandle::open(tmp.path(), "w+", 0o644, CompatMode::Auto)
            .expect("failed to open file");

        if !handle.is_gds_available() {
            eprintln!("SKIP scenario 1: GDS not available");
        } else {
            // I/O with pre-allocated buffer under high pressure should work.
            let written = handle
                .write(&dev_buf, io_size, 0, 0)
                .expect("GDS write failed under ~95% VRAM pressure");
            assert_eq!(written, io_size);
            eprintln!("GDS write succeeded under ~95% VRAM pressure");
        }

        // Free everything for the next scenario.
        fill_allocs.clear();
        drop(dev_buf);
        let _ = stream.synchronize();
    }

    // ------------------------------------------------------------------
    // Scenario 2: Single allocation exceeding free VRAM → Err, not panic
    //
    // WARNING: This WILL poison the CUDA async allocator. All scenarios
    // after this point must tolerate a poisoned allocator.
    // ------------------------------------------------------------------
    eprintln!("\n--- Scenario 2: alloc exceeding free VRAM ---");
    {
        // Re-query to get current free VRAM after scenario 1 cleanup.
        let free_now = cudarc::driver::result::mem_get_info()
            .expect("VRAM query failed before any OOM scenario — unexpected")
            .0;

        let over_size = free_now.saturating_add(1024 * 1024 * 1024);
        let result = stream.alloc_zeros::<u8>(over_size);
        assert!(
            result.is_err(),
            "allocating {} MiB (free was {} MiB) should fail, but succeeded",
            over_size / (1024 * 1024),
            free_now / (1024 * 1024)
        );
        eprintln!(
            "OOM correctly returned error for {} MiB allocation (free: {} MiB)",
            over_size / (1024 * 1024),
            free_now / (1024 * 1024)
        );

        let _ = stream.synchronize();
    }

    // ------------------------------------------------------------------
    // Scenario 3: Incremental allocation until OOM
    //
    // WARNING: This *will* poison the CUDA async allocator. All scenarios
    // after this point must tolerate a poisoned allocator.
    // ------------------------------------------------------------------
    eprintln!("\n--- Scenario 3: incremental alloc until OOM ---");
    {
        let mut allocations = Vec::new();
        let mut hit_oom = false;

        // Allocate in progressively smaller chunks to truly exhaust VRAM.
        for &chunk_size in &[
            128 * 1024 * 1024, // 128 MiB
            16 * 1024 * 1024,  // 16 MiB
            1024 * 1024,       // 1 MiB
        ] {
            loop {
                match stream.alloc_zeros::<u8>(chunk_size) {
                    Ok(slice) => allocations.push(slice),
                    Err(_) => {
                        hit_oom = true;
                        break;
                    }
                }
            }
        }

        assert!(hit_oom, "expected OOM but never failed");
        eprintln!("allocated {} buffers before OOM", allocations.len());

        // Drop all and try recovery.
        allocations.clear();
        let _ = stream.synchronize();

        let after_free = stream.alloc_zeros::<u8>(1024 * 1024);
        if after_free.is_ok() {
            eprintln!("recovery after incremental OOM: succeeded");
        } else {
            eprintln!("recovery after incremental OOM: failed (known CUDA async allocator limitation)");
        }
    }

    // ------------------------------------------------------------------
    // Scenario 4: clone_htod with exhausted VRAM → Err, not panic
    //
    // The allocator is likely poisoned from scenario 3. We verify that
    // clone_htod returns Err (not panic) regardless of allocator state.
    // ------------------------------------------------------------------
    eprintln!("\n--- Scenario 4: clone_htod under exhausted VRAM ---");
    {
        // Fill VRAM again (may fail immediately if allocator is poisoned).
        let mut fill_allocs = Vec::new();
        let chunk_size = 128 * 1024 * 1024;
        while let Ok(s) = stream.alloc_zeros::<u8>(chunk_size) {
            fill_allocs.push(s);
        }
        let small_chunk = 4 * 1024 * 1024;
        while let Ok(s) = stream.alloc_zeros::<u8>(small_chunk) {
            fill_allocs.push(s);
        }

        if fill_allocs.is_empty() {
            // Allocator is already poisoned from scenario 3 — try clone_htod
            // anyway. It should still return Err, not panic.
            eprintln!("allocator already poisoned, testing clone_htod directly");
        }

        let try_size = 1024 * 1024; // 1 MiB
        let data = vec![0u8; try_size];
        let result = stream.clone_htod(&data);
        assert!(
            result.is_err(),
            "clone_htod should fail when VRAM is exhausted"
        );
        eprintln!("clone_htod correctly returned error under exhausted VRAM");

        fill_allocs.clear();
        let _ = stream.synchronize();

        let recovered = stream.clone_htod(&data);
        if recovered.is_ok() {
            eprintln!("recovery after clone_htod OOM: succeeded");
        } else {
            eprintln!("recovery after clone_htod OOM: failed (known CUDA async allocator limitation)");
        }
    }
}
