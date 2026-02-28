//! End-to-end host I/O integration tests for FileHandle.
//!
//! Mirrors C++ kvikio's `test_basic_io.cpp` BasicIOTest fixture.
//! All tests use `CompatMode::On` (POSIX-only) to run without a GPU.

mod test_utils;

use kvik_rs::{CompatMode, ErrorKind, FileHandle};

use test_utils::{assert_data_eq, gen_data};

/// 4 KB write + read round-trip.
#[test]
fn test_write_read_small() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path();
    let data = gen_data(4096);

    let handle = FileHandle::open(path, "w", 0o644, CompatMode::On).unwrap();
    let written = handle.write_host(&data, 0).unwrap();
    assert_eq!(written, data.len());
    drop(handle);

    let handle = FileHandle::open(path, "r", 0o644, CompatMode::On).unwrap();
    assert_eq!(handle.nbytes().unwrap(), data.len() as u64);
    let mut buf = vec![0u8; data.len()];
    let read = handle.read_host(&mut buf, 0).unwrap();
    assert_eq!(read, data.len());
    assert_data_eq(&data, &buf);
}

/// ~8 MB write + read round-trip with deliberately unaligned size.
/// Mirrors C++ test: `1024 * 1024 + 124` elements.
#[test]
fn test_write_read_large() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path();
    let size = 1024 * 1024 + 124;
    let data = gen_data(size);

    let handle = FileHandle::open(path, "w", 0o644, CompatMode::On).unwrap();
    let written = handle.write_host(&data, 0).unwrap();
    assert_eq!(written, size);
    drop(handle);

    let handle = FileHandle::open(path, "r", 0o644, CompatMode::On).unwrap();
    assert_eq!(handle.nbytes().unwrap(), size as u64);
    let mut buf = vec![0u8; size];
    let read = handle.read_host(&mut buf, 0).unwrap();
    assert_eq!(read, size);
    assert_data_eq(&data, &buf);
}

/// Write full file, read back a subset at non-zero offset.
#[test]
fn test_write_read_at_offset() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path();
    let data = gen_data(10000);

    let handle = FileHandle::open(path, "w", 0o644, CompatMode::On).unwrap();
    handle.write_host(&data, 0).unwrap();
    drop(handle);

    let handle = FileHandle::open(path, "r", 0o644, CompatMode::On).unwrap();

    // Read 2000 bytes starting at offset 5000.
    let mut buf = vec![0u8; 2000];
    let read = handle.read_host(&mut buf, 5000).unwrap();
    assert_eq!(read, 2000);
    assert_data_eq(&data[5000..7000], &buf);
}

/// Write with one handle, read with a different handle.
#[test]
fn test_write_read_multiple_handles() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path();
    let data = gen_data(8192);

    // Write handle.
    let write_handle = FileHandle::open(path, "w", 0o644, CompatMode::On).unwrap();
    write_handle.write_host(&data, 0).unwrap();
    drop(write_handle);

    // Read handle - entirely separate.
    let read_handle = FileHandle::open(path, "r", 0o644, CompatMode::On).unwrap();
    let mut buf = vec![0u8; data.len()];
    let read = read_handle.read_host(&mut buf, 0).unwrap();
    assert_eq!(read, data.len());
    assert_data_eq(&data, &buf);
}

/// Force CompatMode::On and verify I/O works correctly.
#[test]
fn test_write_read_compat_on() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path();
    let data = gen_data(16384);

    let handle = FileHandle::open(path, "w", 0o644, CompatMode::On).unwrap();
    assert_eq!(handle.compat_mode(), CompatMode::On);
    assert!(!handle.is_gds_available());
    handle.write_host(&data, 0).unwrap();
    drop(handle);

    let handle = FileHandle::open(path, "r", 0o644, CompatMode::On).unwrap();
    assert_eq!(handle.compat_mode(), CompatMode::On);
    let mut buf = vec![0u8; data.len()];
    handle.read_host(&mut buf, 0).unwrap();
    assert_data_eq(&data, &buf);
}

/// Read with offset past file end — expect short read (0 bytes).
#[test]
fn test_read_beyond_eof() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path();
    let data = gen_data(100);

    let handle = FileHandle::open(path, "w", 0o644, CompatMode::On).unwrap();
    handle.write_host(&data, 0).unwrap();
    drop(handle);

    let handle = FileHandle::open(path, "r", 0o644, CompatMode::On).unwrap();
    let mut buf = vec![0u8; 100];
    // Offset is beyond file end.
    let read = handle.read_host(&mut buf, 200).unwrap();
    assert_eq!(read, 0);
}

/// Read and write with zero-length buffers.
#[test]
fn test_zero_size_operations() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path();

    let handle = FileHandle::open(path, "w+", 0o644, CompatMode::On).unwrap();
    let written = handle.write_host(&[], 0).unwrap();
    assert_eq!(written, 0);

    let mut empty_buf = [];
    let read = handle.read_host(&mut empty_buf, 0).unwrap();
    assert_eq!(read, 0);
}

/// Operations on a closed handle should fail appropriately.
#[test]
fn test_closed_handle_read() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path();

    // Write some data first.
    let data = gen_data(100);
    let handle = FileHandle::open(path, "w", 0o644, CompatMode::On).unwrap();
    handle.write_host(&data, 0).unwrap();
    drop(handle);

    let mut handle = FileHandle::open(path, "r", 0o644, CompatMode::On).unwrap();
    handle.close();

    // Reading from a closed handle (fd = -1) should return an error.
    let mut buf = vec![0u8; 100];
    let result = handle.read_host(&mut buf, 0);
    assert!(result.is_err());
}

/// Verify file metadata is accessible.
#[test]
fn test_file_metadata() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path();
    let data = gen_data(12345);

    let handle = FileHandle::open(path, "w", 0o644, CompatMode::On).unwrap();
    handle.write_host(&data, 0).unwrap();

    // Refresh to pick up new size.
    let size = handle.refresh_nbytes().unwrap();
    assert_eq!(size, 12345);
    assert_eq!(handle.nbytes().unwrap(), 12345);
    assert_eq!(handle.path(), path);
    assert!(handle.fd() >= 0);
}

/// Open a nonexistent file for reading should return NotFound.
#[test]
fn test_open_nonexistent() {
    let err = FileHandle::open(
        std::path::Path::new("/nonexistent/integration_test_file.bin"),
        "r",
        0o644,
        CompatMode::On,
    )
    .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::NotFound);
}

/// Invalid flag string should return ConfigInvalid.
#[test]
fn test_open_invalid_flags() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let err = FileHandle::open(tmp.path(), "x", 0o644, CompatMode::On).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::ConfigInvalid);
}

/// Write at a non-zero file offset (append-like behavior with w+).
#[test]
fn test_write_at_offset() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path();

    let handle = FileHandle::open(path, "w+", 0o644, CompatMode::On).unwrap();

    // Write "AAAA" at offset 0.
    handle.write_host(&[0xAA; 4], 0).unwrap();
    // Write "BBBB" at offset 4.
    handle.write_host(&[0xBB; 4], 4).unwrap();

    // Read entire file.
    let size = handle.refresh_nbytes().unwrap();
    assert_eq!(size, 8);
    let mut buf = vec![0u8; 8];
    handle.read_host(&mut buf, 0).unwrap();
    assert_eq!(&buf[..4], &[0xAA; 4]);
    assert_eq!(&buf[4..], &[0xBB; 4]);
}

/// Debug formatting should include key handle information.
#[test]
fn test_file_handle_debug() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let handle = FileHandle::open(tmp.path(), "r", 0o644, CompatMode::On).unwrap();
    let debug = format!("{handle:?}");
    assert!(debug.contains("FileHandle"));
    assert!(debug.contains("compat_mode"));
}

/// Partial reads: read less data than the file contains.
#[test]
fn test_partial_read() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path();
    let data = gen_data(10000);

    let handle = FileHandle::open(path, "w", 0o644, CompatMode::On).unwrap();
    handle.write_host(&data, 0).unwrap();
    drop(handle);

    let handle = FileHandle::open(path, "r", 0o644, CompatMode::On).unwrap();
    // Read only 500 bytes.
    let mut buf = vec![0u8; 500];
    let read = handle.read_host(&mut buf, 0).unwrap();
    assert_eq!(read, 500);
    assert_data_eq(&data[..500], &buf);
}
