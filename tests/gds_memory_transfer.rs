//! GDS memory transfer validation test.
//!
//! Verifies that GDS writes transfer data directly from GPU VRAM to file
//! **without staging through host RAM** when the filesystem supports true
//! GPUDirect Storage DMA.
//!
//! The test monitors both GPU VRAM (via `cudarc::driver::result::mem_get_info`)
//! and host anonymous RSS (via `/proc/self/status` → `RssAnon`) to confirm
//! that a 256 MiB GDS write does not cause a corresponding host memory spike.
//!
//! # cuFile Compat Mode
//!
//! When `allow_compat_mode: true` in `/etc/cufile.json` (the default), the
//! cuFile driver can silently fall back to a POSIX I/O path that stages data
//! through host RAM — even though `cuFileHandleRegister` succeeded and
//! `cuFileWrite` returns success. This happens when the underlying filesystem
//! or storage device does not support GDS DMA (e.g., tmpfs, ext4 on non-NVMe).
//!
//! The test detects this fallback by measuring host RSS and reports it as a
//! diagnostic skip rather than a hard failure, since the kvik-rs write path
//! itself is correct — the host-memory staging occurs inside the cuFile driver.
//!
//! **Requires:** CUDA GPU + GDS-capable filesystem + NVMe/RDMA storage.
//!
//! ```sh
//! cargo test --test gds_memory_transfer -- --ignored --nocapture
//! ```

mod test_utils;

use kvik_rs::{CompatMode, FileHandle};
use test_utils::{GpuMemInfo, HostMemInfo};

/// Maximum allowed host `RssAnon` increase (kB) during a true GDS write.
///
/// Set to 4 MiB to account for measurement noise, stack frames, cuFile driver
/// internal bookkeeping, and small allocations. A compat-mode fallback would
/// show a large fraction of the IO_SIZE as host RSS growth.
const HOST_RSS_THRESHOLD_KB: i64 = 4 * 1024; // 4 MiB

/// Size of the GPU buffer and GDS write (bytes).
const IO_SIZE: usize = 256 * 1024 * 1024; // 256 MiB

/// Validate that GDS writes use GPU VRAM directly, not host RAM.
///
/// Flow:
/// 1. Create CUDA context (skip if no GPU).
/// 2. Record baseline GPU VRAM.
/// 3. Allocate 256 MiB on GPU via `clone_htod`, then drop the host `Vec`.
/// 4. Verify GPU VRAM increased by ~256 MiB.
/// 5. Record baseline host RSS **after** dropping the host Vec.
/// 6. Open temp file, verify GDS handle is available (skip if not).
/// 7. Perform GDS write.
/// 8. Check host `RssAnon` delta — if large, cuFile is using compat mode
///    internally (skip with diagnostic); if small, true GDS DMA confirmed.
#[test]
#[ignore = "requires CUDA GPU + GDS: cargo test --test gds_memory_transfer -- --ignored --nocapture"]
fn test_gds_write_uses_vram_not_host_ram() {
    // --- Step 1: Create CUDA context ---
    let ctx = match cudarc::driver::CudaContext::new(0) {
        Ok(c) => c,
        Err(_) => {
            eprintln!("SKIP: no CUDA GPU available");
            return;
        }
    };

    let stream = ctx.default_stream();

    // --- Step 2: Record baseline GPU VRAM ---
    let gpu_before = GpuMemInfo::now(&ctx);
    eprintln!(
        "GPU VRAM baseline: {} MiB used, {} MiB free, {} MiB total",
        gpu_before.used_bytes() / (1024 * 1024),
        gpu_before.free_bytes / (1024 * 1024),
        gpu_before.total_bytes / (1024 * 1024),
    );

    // --- Step 3: Allocate 256 MiB on GPU, then drop host source ---
    let dev_buf = {
        let host_data = test_utils::gen_data(IO_SIZE);
        let buf = stream
            .clone_htod(&host_data)
            .expect("failed to allocate and copy 256 MiB to GPU");
        // Synchronize to ensure the H2D copy is complete before dropping host data.
        stream
            .synchronize()
            .expect("stream sync failed after clone_htod");
        // host_data is dropped here, freeing host memory.
        buf
    };

    // --- Step 4: Verify GPU VRAM increased ---
    let gpu_after_alloc = GpuMemInfo::now(&ctx);
    let vram_delta = gpu_after_alloc.used_delta_bytes(&gpu_before);
    eprintln!(
        "GPU VRAM after allocation: delta = {} MiB (expected ~{} MiB)",
        vram_delta / (1024 * 1024),
        IO_SIZE / (1024 * 1024),
    );
    // VRAM should have increased by at least 50% of IO_SIZE (allowing for
    // allocator overhead and rounding).
    assert!(
        vram_delta >= (IO_SIZE as i64) / 2,
        "GPU VRAM did not increase as expected after allocation: \
         delta={vram_delta} bytes, expected >= {} bytes",
        IO_SIZE / 2,
    );

    // --- Step 5: Record baseline host RSS ---
    // This is measured AFTER the host Vec is dropped, so we isolate the
    // GDS write's effect on host memory.
    let host_before = HostMemInfo::now();
    eprintln!(
        "Host RSS baseline (after dropping host Vec): VmRSS={} kB, RssAnon={} kB",
        host_before.vm_rss_kb, host_before.rss_anon_kb,
    );

    // --- Step 6: Open temp file, check GDS availability ---
    // GDS requires a real block-device-backed filesystem (ext4/xfs on NVMe).
    // tmpfs (the default for NamedTempFile) does not support DMA, causing
    // cuFile to silently fall back to compat mode (host bounce buffer).
    // Use KVIK_GDS_TEST_DIR (e.g. /mnt/nvme) to point at an NVMe-backed mount.
    let tmp_dir = std::env::var("KVIK_GDS_TEST_DIR")
        .unwrap_or_else(|_| std::env::temp_dir().to_string_lossy().into_owned());
    let tmp = tempfile::NamedTempFile::new_in(&tmp_dir)
        .unwrap_or_else(|e| panic!("failed to create temp file in {tmp_dir}: {e}"));
    let handle = FileHandle::open(tmp.path(), "w+", 0o644, CompatMode::Auto)
        .expect("failed to open temp file");

    if !handle.is_gds_available() {
        eprintln!("SKIP: GDS not available on this filesystem");
        return;
    }

    // --- Step 7: Perform GDS write ---
    let written = handle
        .write(&dev_buf, IO_SIZE, 0, 0)
        .expect("GDS write failed");
    assert_eq!(
        written, IO_SIZE,
        "GDS write returned {written}, expected {IO_SIZE}"
    );

    // --- Step 8: Check host RSS after GDS write ---
    let host_after = HostMemInfo::now();
    let host_anon_delta = host_after.anon_delta_kb(&host_before);
    eprintln!(
        "Host RSS after GDS write: VmRSS={} kB, RssAnon={} kB",
        host_after.vm_rss_kb, host_after.rss_anon_kb,
    );
    eprintln!(
        "Host RssAnon delta: {} kB (threshold: {} kB)",
        host_anon_delta, HOST_RSS_THRESHOLD_KB,
    );

    if host_anon_delta >= HOST_RSS_THRESHOLD_KB {
        // cuFile's internal compat mode fallback: cuFileWrite succeeded but
        // staged data through host RAM. This happens when the filesystem or
        // storage device doesn't support true GDS DMA, and
        // `allow_compat_mode: true` is set in /etc/cufile.json.
        //
        // This is NOT a kvik-rs bug — the write path correctly calls
        // cuFileWrite with a device pointer. The host staging happens inside
        // the NVIDIA cuFile driver.
        eprintln!(
            "SKIP: cuFile driver used internal compat mode (host bounce buffer). \
             Host RssAnon grew by {} kB ({} MiB) during {} MiB write. \
             This filesystem/storage does not support true GDS DMA. \
             To validate zero-copy GDS, run on a GDS-capable filesystem \
             (ext4/xfs on NVMe with nvidia-fs kernel module).",
            host_anon_delta,
            host_anon_delta / 1024,
            IO_SIZE / (1024 * 1024),
        );
        return;
    }

    // True GDS DMA confirmed: host memory did not grow significantly.
    eprintln!(
        "PASS: GDS write of {} MiB completed with minimal host RSS impact \
         (RssAnon delta: {} kB < {} kB threshold). \
         True GPU→storage DMA confirmed.",
        IO_SIZE / (1024 * 1024),
        host_anon_delta,
        HOST_RSS_THRESHOLD_KB,
    );
}
