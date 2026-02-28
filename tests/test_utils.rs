//! Shared test utilities for kvik-rs integration tests.
//!
//! Provides reusable helpers for temp files, deterministic data generation,
//! environment variable guards, and aligned/unaligned buffer allocation.
//! Mirrors patterns from C++ kvikio's `cpp/tests/utils/utils.hpp` and
//! `cpp/tests/utils/env.hpp`.

#![allow(dead_code)]

use std::env;

use kvik_rs::align::{is_aligned_ptr, page_size};

/// RAII guard for temporarily setting environment variables.
///
/// Mirrors C++ kvikio's `EnvVarContext`. Saves the current value of each
/// variable on construction and restores it on drop.
///
/// # Safety
///
/// Environment variable manipulation is inherently not thread-safe.
/// Tests using this guard should be run with `--test-threads=1` or
/// use unique variable names to avoid interference.
pub struct EnvVarGuard {
    saved: Vec<(String, Option<String>)>,
}

impl EnvVarGuard {
    /// Create a guard that sets the given environment variables.
    ///
    /// Each `(key, value)` pair is set immediately. The previous values
    /// are saved and will be restored when the guard is dropped.
    pub fn new(vars: &[(&str, &str)]) -> Self {
        let mut saved = Vec::with_capacity(vars.len());
        for (key, value) in vars {
            saved.push((key.to_string(), env::var(key).ok()));
            // SAFETY: Tests using EnvVarGuard should not run in parallel
            // with other tests that read the same env vars.
            unsafe { env::set_var(key, value) };
        }
        Self { saved }
    }

    /// Create a guard that removes the given environment variables.
    pub fn remove(vars: &[&str]) -> Self {
        let mut saved = Vec::with_capacity(vars.len());
        for key in vars {
            saved.push((key.to_string(), env::var(key).ok()));
            // SAFETY: See above.
            unsafe { env::remove_var(key) };
        }
        Self { saved }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        for (key, prev) in &self.saved {
            // SAFETY: Restoring previous state.
            match prev {
                Some(val) => unsafe { env::set_var(key, val) },
                None => unsafe { env::remove_var(key) },
            }
        }
    }
}

/// Generate deterministic test data of the given size.
///
/// Produces a sequential byte pattern (0, 1, 2, ..., 255, 0, 1, ...),
/// similar to C++ kvikio's `DevBuffer::arange` pattern.
pub fn gen_data(size: usize) -> Vec<u8> {
    (0..size).map(|i| (i % 256) as u8).collect()
}

/// Compare two byte slices with informative error messages on mismatch.
///
/// Mirrors C++ kvikio's `expect_equal` function. Panics with details
/// about the first mismatched position.
pub fn assert_data_eq(expected: &[u8], actual: &[u8]) {
    assert_eq!(
        expected.len(),
        actual.len(),
        "buffer length mismatch: expected {}, got {}",
        expected.len(),
        actual.len()
    );

    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        if e != a {
            panic!(
                "data mismatch at byte {i}: expected 0x{e:02x}, got 0x{a:02x} \
                 (first mismatch of {} bytes)",
                expected.len()
            );
        }
    }
}

/// A page-aligned host memory buffer.
///
/// Allocated via `posix_memalign` to satisfy Direct I/O alignment requirements.
/// Mirrors C++ kvikio's `CustomHostAllocator<T, 4096>`.
pub struct AlignedBuffer {
    ptr: *mut u8,
    size: usize,
}

impl AlignedBuffer {
    /// Allocate a page-aligned buffer of the given size.
    pub fn new(size: usize) -> Self {
        let ps = page_size();
        let mut ptr: *mut libc::c_void = std::ptr::null_mut();
        let ret = unsafe { libc::posix_memalign(&mut ptr, ps, size) };
        assert_eq!(ret, 0, "posix_memalign failed");
        assert!(!ptr.is_null());
        assert!(is_aligned_ptr(ptr as *const u8, ps));
        // Zero-initialize.
        unsafe { std::ptr::write_bytes(ptr as *mut u8, 0, size) };
        Self {
            ptr: ptr as *mut u8,
            size,
        }
    }

    /// Fill the buffer with the given data, which must fit.
    pub fn fill_from(&mut self, data: &[u8]) {
        assert!(data.len() <= self.size, "data exceeds buffer size");
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.ptr, data.len());
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.size) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size) }
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        unsafe { libc::free(self.ptr as *mut libc::c_void) };
    }
}

// SAFETY: The buffer is solely owned and not shared.
unsafe impl Send for AlignedBuffer {}

/// A deliberately mis-aligned host memory buffer.
///
/// Allocated page-aligned, then exposed with a configurable byte offset
/// to force the unaligned code paths (bounce buffer usage).
/// Mirrors C++ kvikio's `CustomHostAllocator<T, 4096, 123>`.
pub struct UnalignedBuffer {
    /// The underlying page-aligned allocation.
    aligned: AlignedBuffer,
    /// Byte offset from the start of the aligned allocation.
    offset: usize,
    /// Usable size (aligned.size - offset).
    usable_size: usize,
}

impl UnalignedBuffer {
    /// Allocate a buffer with `usable_size` bytes, mis-aligned by `offset` bytes.
    ///
    /// The default offset (123 bytes) mirrors the C++ kvikio test convention.
    pub fn new(usable_size: usize) -> Self {
        Self::with_offset(usable_size, 123)
    }

    /// Allocate with a specific misalignment offset.
    pub fn with_offset(usable_size: usize, offset: usize) -> Self {
        let total = usable_size + offset;
        let aligned = AlignedBuffer::new(total);
        let ps = page_size();
        // Verify the base is aligned but our view is not.
        assert!(is_aligned_ptr(aligned.as_ptr(), ps));
        if offset > 0 {
            assert!(!is_aligned_ptr(unsafe { aligned.as_ptr().add(offset) }, ps));
        }
        Self {
            aligned,
            offset,
            usable_size,
        }
    }

    /// Fill the usable portion with the given data.
    pub fn fill_from(&mut self, data: &[u8]) {
        assert!(data.len() <= self.usable_size, "data exceeds buffer size");
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                self.aligned.ptr.add(self.offset),
                data.len(),
            );
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.aligned.ptr.add(self.offset), self.usable_size) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(self.aligned.ptr.add(self.offset), self.usable_size)
        }
    }

    pub fn as_ptr(&self) -> *const u8 {
        unsafe { self.aligned.ptr.add(self.offset) }
    }

    pub fn len(&self) -> usize {
        self.usable_size
    }

    pub fn is_empty(&self) -> bool {
        self.usable_size == 0
    }
}

/// Host memory info from `/proc/self/status` (Linux only).
///
/// Tracks `VmRSS` (total resident set) and `RssAnon` (anonymous resident
/// pages, i.e., heap/stack, excluding file-backed pages).
pub struct HostMemInfo {
    /// Total resident set size in kB.
    pub vm_rss_kb: u64,
    /// Anonymous (non-file-backed) resident pages in kB.
    pub rss_anon_kb: u64,
}

impl HostMemInfo {
    /// Snapshot current host memory usage from `/proc/self/status`.
    pub fn now() -> Self {
        let status = std::fs::read_to_string("/proc/self/status")
            .expect("failed to read /proc/self/status");
        let mut vm_rss_kb = 0u64;
        let mut rss_anon_kb = 0u64;
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                vm_rss_kb = parse_kb_field(rest);
            } else if let Some(rest) = line.strip_prefix("RssAnon:") {
                rss_anon_kb = parse_kb_field(rest);
            }
        }
        Self {
            vm_rss_kb,
            rss_anon_kb,
        }
    }

    /// Signed delta in anonymous RSS (kB) relative to a baseline.
    pub fn anon_delta_kb(&self, before: &HostMemInfo) -> i64 {
        self.rss_anon_kb as i64 - before.rss_anon_kb as i64
    }
}

/// Parse a `/proc/self/status` value field like `"   12345 kB"` into kB.
fn parse_kb_field(s: &str) -> u64 {
    s.trim()
        .strip_suffix("kB")
        .or_else(|| s.trim().strip_suffix("KB"))
        .unwrap_or(s.trim())
        .trim()
        .parse::<u64>()
        .expect("failed to parse /proc/self/status memory field")
}

/// GPU VRAM info from the cudarc driver API.
pub struct GpuMemInfo {
    /// Free VRAM in bytes.
    pub free_bytes: usize,
    /// Total VRAM in bytes.
    pub total_bytes: usize,
}

impl GpuMemInfo {
    /// Snapshot current GPU VRAM usage. Binds the context to the current thread.
    pub fn now(ctx: &std::sync::Arc<cudarc::driver::CudaContext>) -> Self {
        ctx.bind_to_thread().expect("bind failed");
        let (free, total) =
            cudarc::driver::result::mem_get_info().expect("mem_get_info failed");
        Self {
            free_bytes: free,
            total_bytes: total,
        }
    }

    /// Bytes of VRAM currently in use.
    pub fn used_bytes(&self) -> usize {
        self.total_bytes - self.free_bytes
    }

    /// Signed delta in used VRAM (bytes) relative to a baseline.
    pub fn used_delta_bytes(&self, before: &GpuMemInfo) -> i64 {
        self.used_bytes() as i64 - before.used_bytes() as i64
    }
}

/// Check if Direct I/O (O_DIRECT) is supported on the filesystem containing
/// the given path. Returns `true` if O_DIRECT opens succeed.
pub fn is_direct_io_supported(path: &std::path::Path) -> bool {
    let c_path = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
    let fd = unsafe {
        libc::open(
            c_path.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECT,
            0o644 as libc::mode_t,
        )
    };
    if fd >= 0 {
        unsafe { libc::close(fd) };
        true
    } else {
        false
    }
}
