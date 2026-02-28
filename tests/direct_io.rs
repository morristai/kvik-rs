//! Direct I/O alignment integration tests.
//!
//! Mirrors C++ kvikio's `test_basic_io.cpp` DirectIOTest fixture.
//! Tests aligned and unaligned buffers, bounce buffer paths, and
//! cross-verification with raw POSIX syscalls.

mod test_utils;

use std::os::fd::RawFd;

use kvik_rs::align::{is_aligned_ptr, page_size};
use kvik_rs::{CompatMode, Config, FileHandle};

use test_utils::{assert_data_eq, gen_data, is_direct_io_supported, AlignedBuffer, UnalignedBuffer};

/// Helper: open a file without O_DIRECT.
fn open_buffered(path: &std::path::Path, flags: i32) -> RawFd {
    let c_path = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
    let fd = unsafe { libc::open(c_path.as_ptr(), flags, 0o644 as libc::mode_t) };
    assert!(fd >= 0, "failed to open {}", path.display());
    fd
}

/// Helper: raw pwrite.
unsafe fn raw_pwrite(fd: RawFd, buf: *const u8, count: usize, offset: i64) -> isize {
    unsafe { libc::pwrite(fd, buf as *const libc::c_void, count, offset) }
}

/// Helper: raw pread.
unsafe fn raw_pread(fd: RawFd, buf: *mut u8, count: usize, offset: i64) -> isize {
    unsafe { libc::pread(fd, buf as *mut libc::c_void, count, offset) }
}

// ---- pwrite with aligned buffer ----

#[test]
fn test_pwrite_aligned_buffer() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path();
    let ps = page_size();

    // 1 MiB + 1234 bytes (unaligned total, like C++ test).
    let size = 1024 * 1024 + 1234;
    let data = gen_data(size);

    // Allocate page-aligned buffer and fill with test data.
    let mut aligned_buf = AlignedBuffer::new(size);
    aligned_buf.fill_from(&data);
    assert!(is_aligned_ptr(aligned_buf.as_ptr(), ps));

    // Write using FileHandle.
    let handle = FileHandle::open(path, "w", 0o644, CompatMode::On).unwrap();
    handle.write_host(aligned_buf.as_slice(), 0).unwrap();
    drop(handle);

    // Verify with raw pread (independent of kvik-rs read path).
    let fd = open_buffered(path, libc::O_RDONLY);
    let mut verify_buf = vec![0u8; size];
    let n = unsafe { raw_pread(fd, verify_buf.as_mut_ptr(), size, 0) };
    unsafe { libc::close(fd) };
    assert_eq!(n as usize, size);
    assert_data_eq(&data, &verify_buf);
}

// ---- pwrite with unaligned buffer ----

#[test]
fn test_pwrite_unaligned_buffer() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path();
    let ps = page_size();

    let size = 1024 * 1024 + 1234;
    let data = gen_data(size);

    // Allocate deliberately mis-aligned buffer.
    let mut unaligned_buf = UnalignedBuffer::new(size);
    unaligned_buf.fill_from(&data);
    assert!(!is_aligned_ptr(unaligned_buf.as_ptr(), ps));

    // Write using FileHandle — exercises bounce buffer path.
    let handle = FileHandle::open(path, "w", 0o644, CompatMode::On).unwrap();
    handle.write_host(unaligned_buf.as_slice(), 0).unwrap();
    drop(handle);

    // Verify with raw pread.
    let fd = open_buffered(path, libc::O_RDONLY);
    let mut verify_buf = vec![0u8; size];
    let n = unsafe { raw_pread(fd, verify_buf.as_mut_ptr(), size, 0) };
    unsafe { libc::close(fd) };
    assert_eq!(n as usize, size);
    assert_data_eq(&data, &verify_buf);
}

// ---- pread with aligned buffer ----

#[test]
fn test_pread_aligned_buffer() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path();
    let ps = page_size();

    let size = 1024 * 1024 + 1234;
    let data = gen_data(size);

    // Write ground truth with raw syscall.
    let fd = open_buffered(path, libc::O_WRONLY | libc::O_CREAT | libc::O_TRUNC);
    let n = unsafe { raw_pwrite(fd, data.as_ptr(), size, 0) };
    unsafe { libc::close(fd) };
    assert_eq!(n as usize, size);

    // Read into page-aligned buffer using FileHandle.
    let mut aligned_buf = AlignedBuffer::new(size);
    assert!(is_aligned_ptr(aligned_buf.as_ptr(), ps));

    let handle = FileHandle::open(path, "r", 0o644, CompatMode::On).unwrap();
    let read = handle.read_host(aligned_buf.as_mut_slice(), 0).unwrap();
    assert_eq!(read, size);
    assert_data_eq(&data, aligned_buf.as_slice());
}

// ---- pread with unaligned buffer ----

#[test]
fn test_pread_unaligned_buffer() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path();
    let ps = page_size();

    let size = 1024 * 1024 + 1234;
    let data = gen_data(size);

    // Write ground truth with raw syscall.
    let fd = open_buffered(path, libc::O_WRONLY | libc::O_CREAT | libc::O_TRUNC);
    let n = unsafe { raw_pwrite(fd, data.as_ptr(), size, 0) };
    unsafe { libc::close(fd) };
    assert_eq!(n as usize, size);

    // Read into deliberately mis-aligned buffer using FileHandle.
    let mut unaligned_buf = UnalignedBuffer::new(size);
    assert!(!is_aligned_ptr(unaligned_buf.as_ptr(), ps));

    let handle = FileHandle::open(path, "r", 0o644, CompatMode::On).unwrap();
    let read = handle.read_host(unaligned_buf.as_mut_slice(), 0).unwrap();
    assert_eq!(read, size);
    assert_data_eq(&data, unaligned_buf.as_slice());
}

// ---- Direct I/O flag tests ----

#[test]
fn test_direct_io_flag_read_enabled() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path();

    // Write test data.
    let data = gen_data(page_size() * 2);
    let handle = FileHandle::open(path, "w", 0o644, CompatMode::On).unwrap();
    handle.write_host(&data, 0).unwrap();
    drop(handle);

    // Save and set config.
    let saved = Config::get();
    Config::update(|c| c.auto_direct_io_read = true);

    let handle = FileHandle::open(path, "r", 0o644, CompatMode::On).unwrap();
    let mut buf = vec![0u8; data.len()];
    let read = handle.read_host(&mut buf, 0).unwrap();
    assert_eq!(read, data.len());
    assert_data_eq(&data, &buf);

    Config::set(saved);
}

#[test]
fn test_direct_io_flag_read_disabled() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path();

    let data = gen_data(page_size() * 2);
    let handle = FileHandle::open(path, "w", 0o644, CompatMode::On).unwrap();
    handle.write_host(&data, 0).unwrap();
    drop(handle);

    // Disable Direct I/O for reads.
    let saved = Config::get();
    Config::update(|c| c.auto_direct_io_read = false);

    let handle = FileHandle::open(path, "r", 0o644, CompatMode::On).unwrap();
    let mut buf = vec![0u8; data.len()];
    let read = handle.read_host(&mut buf, 0).unwrap();
    assert_eq!(read, data.len());
    assert_data_eq(&data, &buf);

    Config::set(saved);
}

#[test]
fn test_direct_io_flag_write_enabled() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path();

    let data = gen_data(page_size() * 2);

    let saved = Config::get();
    Config::update(|c| c.auto_direct_io_write = true);

    let handle = FileHandle::open(path, "w", 0o644, CompatMode::On).unwrap();
    let written = handle.write_host(&data, 0).unwrap();
    assert_eq!(written, data.len());
    drop(handle);

    // Verify.
    let fd = open_buffered(path, libc::O_RDONLY);
    let mut verify_buf = vec![0u8; data.len()];
    let n = unsafe { raw_pread(fd, verify_buf.as_mut_ptr(), data.len(), 0) };
    unsafe { libc::close(fd) };
    assert_eq!(n as usize, data.len());
    assert_data_eq(&data, &verify_buf);

    Config::set(saved);
}

#[test]
fn test_direct_io_flag_write_disabled() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path();

    let data = gen_data(page_size() * 2);

    let saved = Config::get();
    Config::update(|c| c.auto_direct_io_write = false);

    let handle = FileHandle::open(path, "w", 0o644, CompatMode::On).unwrap();
    let written = handle.write_host(&data, 0).unwrap();
    assert_eq!(written, data.len());
    drop(handle);

    // Verify.
    let fd = open_buffered(path, libc::O_RDONLY);
    let mut verify_buf = vec![0u8; data.len()];
    let n = unsafe { raw_pread(fd, verify_buf.as_mut_ptr(), data.len(), 0) };
    unsafe { libc::close(fd) };
    assert_eq!(n as usize, data.len());
    assert_data_eq(&data, &verify_buf);

    Config::set(saved);
}

// ---- Cross-verification: write with kvik-rs, read with raw POSIX; and vice versa ----

#[test]
fn test_cross_verify_kvik_write_raw_read() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path();
    let size = 50000;
    let data = gen_data(size);

    // Write with kvik-rs.
    let handle = FileHandle::open(path, "w", 0o644, CompatMode::On).unwrap();
    handle.write_host(&data, 0).unwrap();
    drop(handle);

    // Read with raw pread.
    let fd = open_buffered(path, libc::O_RDONLY);
    let mut buf = vec![0u8; size];
    let mut total = 0;
    while total < size {
        let n = unsafe { raw_pread(fd, buf.as_mut_ptr().add(total), size - total, total as i64) };
        assert!(n > 0, "raw_pread returned {n}");
        total += n as usize;
    }
    unsafe { libc::close(fd) };
    assert_data_eq(&data, &buf);
}

#[test]
fn test_cross_verify_raw_write_kvik_read() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path();
    let size = 50000;
    let data = gen_data(size);

    // Write with raw pwrite.
    let fd = open_buffered(path, libc::O_WRONLY | libc::O_CREAT | libc::O_TRUNC);
    let mut total = 0;
    while total < size {
        let n = unsafe {
            raw_pwrite(fd, data.as_ptr().add(total), size - total, total as i64)
        };
        assert!(n > 0, "raw_pwrite returned {n}");
        total += n as usize;
    }
    unsafe { libc::close(fd) };

    // Read with kvik-rs.
    let handle = FileHandle::open(path, "r", 0o644, CompatMode::On).unwrap();
    let mut buf = vec![0u8; size];
    let read = handle.read_host(&mut buf, 0).unwrap();
    assert_eq!(read, size);
    assert_data_eq(&data, &buf);
}

/// Cross-verification with Direct I/O if the filesystem supports it.
#[test]
fn test_cross_verify_with_direct_io() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path();

    if !is_direct_io_supported(path) {
        eprintln!("Skipping test_cross_verify_with_direct_io: O_DIRECT not supported");
        return;
    }

    let ps = page_size();
    // Use page-aligned size for Direct I/O.
    let size = ps * 4;
    let data = gen_data(size);

    // Write with kvik-rs (Direct I/O enabled).
    let saved = Config::get();
    Config::update(|c| c.auto_direct_io_write = true);

    let handle = FileHandle::open(path, "w", 0o644, CompatMode::On).unwrap();
    handle.write_host(&data, 0).unwrap();
    drop(handle);

    Config::set(saved);

    // Read back with raw pread (buffered).
    let fd = open_buffered(path, libc::O_RDONLY);
    let mut buf = vec![0u8; size];
    let n = unsafe { raw_pread(fd, buf.as_mut_ptr(), size, 0) };
    unsafe { libc::close(fd) };
    assert_eq!(n as usize, size);
    assert_data_eq(&data, &buf);
}
