//! Full GDS device I/O example.
//!
//! Demonstrates GPUDirect Storage with device memory when a CUDA GPU is
//! available. Falls back gracefully when GDS is not present.
//!
//! Mirrors C++ kvikio's `basic_io.cpp` example.
//!
//! # Requirements
//!
//! - NVIDIA GPU with CUDA support
//! - cuFile/GDS drivers installed (or runs in compat mode)
//!
//! # Usage
//!
//! ```sh
//! cargo run --example basic_gds_io
//! ```

use std::time::Instant;

use kvik_rs::{CompatMode, Config, FileHandle};

fn main() {
    println!("kvik-rs: GDS Device I/O Example");
    println!("================================\n");

    let config = Config::get();
    println!("Configuration:");
    println!("  compat_mode:  {}", config.compat_mode);
    println!("  num_threads:  {}", config.num_threads);
    println!("  task_size:    {} bytes", config.task_size);
    println!("  gds_threshold: {} bytes", config.gds_threshold);
    println!();

    // Parameters (mirrors C++ NELEM=1024, SIZE=NELEM*sizeof(int)=4096).
    let nelem: usize = 1024;
    let size = nelem * std::mem::size_of::<i32>();
    println!("Test parameters:");
    println!("  elements:     {nelem}");
    println!("  total size:   {size} bytes");
    println!();

    // --- Step 1: Try to initialize CUDA ---
    println!("--- CUDA Initialization ---");
    let ctx = match cudarc::driver::CudaContext::new(0) {
        Ok(ctx) => {
            println!("  CUDA device 0: initialized successfully");
            ctx
        }
        Err(e) => {
            println!("  CUDA device 0: not available ({e})");
            println!("\n  Falling back to host-only I/O demo.\n");
            run_host_only_demo(nelem, size);
            return;
        }
    };

    let stream = ctx.default_stream();

    // --- Step 2: Allocate device memory ---
    println!("\n--- Device Memory ---");
    let data: Vec<u8> = (0..nelem).flat_map(|i| (i as i32).to_ne_bytes()).collect();

    let dev_a = stream
        .clone_htod(&data)
        .expect("clone_htod failed for dev_a");
    let mut dev_b = stream
        .alloc_zeros::<u8>(size)
        .expect("alloc_zeros failed for dev_b");
    println!("  Allocated dev_a ({size} bytes) - source data");
    println!("  Allocated dev_b ({size} bytes) - target buffer");

    // --- Step 3: Create temp file and open with Auto compat mode ---
    let tmp = tempfile::NamedTempFile::new().expect("failed to create temp file");
    let path = tmp.path();
    println!("\n--- File Handle ---");
    println!("  Temp file: {}", path.display());

    let handle =
        FileHandle::open(path, "w+", 0o644, CompatMode::Auto).expect("failed to open file");
    println!("  compat_mode:   {}", handle.compat_mode());
    println!("  gds_available: {}", handle.is_gds_available());

    // --- Step 4: GDS Write ---
    println!("\n--- GDS Write ---");
    let start = Instant::now();
    match handle.write(&dev_a, size, 0, 0) {
        Ok(written) => {
            let elapsed = start.elapsed();
            println!("  Wrote {written} bytes in {:.1} us", elapsed.as_micros());
            assert_eq!(written, size, "short write");
        }
        Err(e) => {
            println!("  GDS write not available: {e}");
            println!("  (This is expected on systems without GDS/cuFile support)");
            println!("\n  Falling back to host-only I/O demo.\n");
            drop(handle);
            run_host_only_demo(nelem, size);
            return;
        }
    }

    // --- Step 5: GDS Read ---
    println!("\n--- GDS Read ---");
    let start = Instant::now();
    match handle.read(&mut dev_b, size, 0, 0) {
        Ok(read) => {
            let elapsed = start.elapsed();
            println!("  Read {read} bytes in {:.1} us", elapsed.as_micros());
            assert_eq!(read, size, "short read");
        }
        Err(e) => {
            println!("  GDS read failed: {e}");
            return;
        }
    }

    // --- Step 6: Verify ---
    println!("\n--- Verification ---");
    let result = stream.clone_dtoh(&dev_b).expect("clone_dtoh failed");
    for i in 0..nelem {
        let offset = i * 4;
        let expected = (i as i32).to_ne_bytes();
        let actual = &result[offset..offset + 4];
        assert_eq!(actual, &expected, "data mismatch at element {i}");
    }
    println!("  Data verification: PASSED ({nelem} elements correct)");

    // --- Step 7: Host I/O path (POSIX through FileHandle) ---
    println!("\n--- Host Memory I/O (POSIX path) ---");
    let host_data: Vec<u8> = (0..nelem)
        .flat_map(|i| ((i + 500) as i32).to_ne_bytes())
        .collect();

    let start = Instant::now();
    let written = handle.write_host(&host_data, 0).expect("write_host failed");
    let write_elapsed = start.elapsed();
    println!(
        "  Host write: {written} bytes in {:.1} us",
        write_elapsed.as_micros()
    );

    let start = Instant::now();
    let mut host_read = vec![0u8; size];
    let read = handle
        .read_host(&mut host_read, 0)
        .expect("read_host failed");
    let read_elapsed = start.elapsed();
    println!(
        "  Host read:  {read} bytes in {:.1} us",
        read_elapsed.as_micros()
    );

    assert_eq!(host_data, host_read, "host data mismatch");
    println!("  Host verification: PASSED");

    println!("\nDone.");
}

/// Fallback demo when no CUDA device is available.
fn run_host_only_demo(nelem: usize, size: usize) {
    println!("--- Host-Only I/O Demo ---");

    let data: Vec<u8> = (0..nelem).flat_map(|i| (i as i32).to_ne_bytes()).collect();

    let tmp = tempfile::NamedTempFile::new().expect("failed to create temp file");
    let path = tmp.path();

    let handle = FileHandle::open(path, "w+", 0o644, CompatMode::On).expect("failed to open file");

    let start = Instant::now();
    let written = handle.write_host(&data, 0).expect("write_host failed");
    let write_elapsed = start.elapsed();
    println!(
        "  Write: {written} bytes in {:.1} us",
        write_elapsed.as_micros()
    );

    let start = Instant::now();
    let mut read_buf = vec![0u8; size];
    let read = handle
        .read_host(&mut read_buf, 0)
        .expect("read_host failed");
    let read_elapsed = start.elapsed();
    println!(
        "  Read:  {read} bytes in {:.1} us",
        read_elapsed.as_micros()
    );

    assert_eq!(data, read_buf, "data mismatch");
    println!("  Verification: PASSED ({nelem} elements correct)");
    println!("\nDone.");
}
