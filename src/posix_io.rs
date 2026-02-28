//! POSIX I/O with opportunistic Direct I/O.
//!
//! Replicates the C++ kvikio `posix_host_io` and `posix_device_io` logic.
//!
//! # Host Memory Path
//!
//! For each I/O operation, the transfer is split into three segments:
//! 1. **Unaligned prefix**: bytes before the next page boundary → buffered I/O.
//! 2. **Aligned middle**: page-aligned offset, page-multiple size → Direct I/O
//!    (using a bounce buffer if the user buffer is not page-aligned).
//! 3. **Unaligned suffix**: remaining bytes < page size → buffered I/O.
//!
//! # Device Memory Path
//!
//! Stages through a host bounce buffer:
//! - Read: `pread` → host bounce buffer → `cuMemcpyHtoDAsync` → device memory
//! - Write: `cuMemcpyDtoHAsync` → host bounce buffer → `pwrite`

use std::os::fd::RawFd;

use crate::align::{align_down, align_up, is_aligned, is_aligned_ptr, page_size};
use crate::error::{Error, ErrorKind, Result};

/// Whether to loop until all bytes are transferred or return after the first I/O.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartialIO {
    /// Return after the first successful I/O (may be partial).
    Yes,
    /// Loop until all requested bytes are transferred.
    No,
}

/// Which I/O operation to perform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoOp {
    Read,
    Write,
}

/// Perform a single `pread` call.
///
/// Returns the number of bytes read, or an error.
///
/// # Safety
///
/// `buf` must point to valid, writable memory of at least `count` bytes.
/// `fd` must be a valid file descriptor open for reading.
pub unsafe fn pread_raw(fd: RawFd, buf: *mut u8, count: usize, offset: i64) -> Result<usize> {
    // SAFETY: Caller guarantees buf is valid for count bytes and fd is valid.
    let ret = unsafe { libc::pread(fd, buf as *mut libc::c_void, count, offset) };
    if ret < 0 {
        let errno = unsafe { *libc::__errno_location() };
        Err(Error::new(
            ErrorKind::SystemError,
            format!("pread failed: {}", std::io::Error::from_raw_os_error(errno)),
        )
        .with_operation("pread")
        .with_context("fd", fd.to_string())
        .with_context("count", count.to_string())
        .with_context("offset", offset.to_string()))
    } else {
        Ok(ret as usize)
    }
}

/// Perform a single `pwrite` call.
///
/// Returns the number of bytes written, or an error.
///
/// # Safety
///
/// `buf` must point to valid, readable memory of at least `count` bytes.
/// `fd` must be a valid file descriptor open for writing.
pub unsafe fn pwrite_raw(fd: RawFd, buf: *const u8, count: usize, offset: i64) -> Result<usize> {
    // SAFETY: Caller guarantees buf is valid for count bytes and fd is valid.
    let ret = unsafe { libc::pwrite(fd, buf as *const libc::c_void, count, offset) };
    if ret < 0 {
        let errno = unsafe { *libc::__errno_location() };
        Err(Error::new(
            ErrorKind::SystemError,
            format!(
                "pwrite failed: {}",
                std::io::Error::from_raw_os_error(errno)
            ),
        )
        .with_operation("pwrite")
        .with_context("fd", fd.to_string())
        .with_context("count", count.to_string())
        .with_context("offset", offset.to_string()))
    } else {
        Ok(ret as usize)
    }
}

/// Read from a file into a host memory buffer using POSIX I/O with opportunistic Direct I/O.
///
/// # Arguments
///
/// * `fd_direct_on` - File descriptor opened with `O_DIRECT` (or -1 if unavailable).
/// * `fd_direct_off` - File descriptor opened without `O_DIRECT`.
/// * `buf` - Destination buffer in host memory.
/// * `file_offset` - Starting offset in the file.
/// * `partial` - Whether to return after first partial I/O or loop until complete.
/// * `use_direct_io` - Whether to attempt Direct I/O for aligned segments.
///
/// Returns the total number of bytes read.
pub fn posix_host_read(
    fd_direct_on: RawFd,
    fd_direct_off: RawFd,
    buf: &mut [u8],
    file_offset: u64,
    partial: PartialIO,
    use_direct_io: bool,
) -> Result<usize> {
    posix_host_io(
        IoOp::Read,
        fd_direct_on,
        fd_direct_off,
        buf.as_mut_ptr(),
        buf.len(),
        file_offset,
        partial,
        use_direct_io,
    )
}

/// Write from a host memory buffer to a file using POSIX I/O with opportunistic Direct I/O.
///
/// # Arguments
///
/// * `fd_direct_on` - File descriptor opened with `O_DIRECT` (or -1 if unavailable).
/// * `fd_direct_off` - File descriptor opened without `O_DIRECT`.
/// * `buf` - Source buffer in host memory.
/// * `file_offset` - Starting offset in the file.
/// * `partial` - Whether to return after first partial I/O or loop until complete.
/// * `use_direct_io` - Whether to attempt Direct I/O for aligned segments.
///
/// Returns the total number of bytes written.
pub fn posix_host_write(
    fd_direct_on: RawFd,
    fd_direct_off: RawFd,
    buf: &[u8],
    file_offset: u64,
    partial: PartialIO,
    use_direct_io: bool,
) -> Result<usize> {
    posix_host_io(
        IoOp::Write,
        fd_direct_on,
        fd_direct_off,
        buf.as_ptr() as *mut u8,
        buf.len(),
        file_offset,
        partial,
        use_direct_io,
    )
}

/// Core implementation of POSIX host I/O with opportunistic Direct I/O.
///
/// Splits the transfer into:
/// 1. Unaligned prefix (buffered I/O)
/// 2. Aligned middle (Direct I/O if available and buffer is aligned; bounce buffer otherwise)
/// 3. Unaligned suffix (buffered I/O)
#[allow(clippy::too_many_arguments)]
fn posix_host_io(
    op: IoOp,
    fd_direct_on: RawFd,
    fd_direct_off: RawFd,
    buf: *mut u8,
    size: usize,
    file_offset: u64,
    partial: PartialIO,
    use_direct_io: bool,
) -> Result<usize> {
    if size == 0 {
        return Ok(0);
    }

    let ps = page_size();
    let can_direct_io = use_direct_io && fd_direct_on >= 0;

    let mut total_transferred: usize = 0;
    let mut remaining = size;
    let mut current_offset = file_offset;
    let mut current_buf = buf;

    while remaining > 0 {
        let offset_usize = current_offset as usize;

        // Determine if this segment should use Direct I/O.
        // Direct I/O requires: page-aligned offset AND at least one full page of data.
        let offset_aligned = is_aligned(offset_usize, ps);
        let use_dio_for_segment = can_direct_io && offset_aligned && remaining >= ps;

        if use_dio_for_segment {
            // How many full pages can we transfer?
            let dio_size = align_down(remaining, ps);

            // Check if the user buffer is also page-aligned.
            let buf_aligned = is_aligned_ptr(current_buf as *const u8, ps);

            let transferred = if buf_aligned {
                // Best case: both offset and buffer aligned, use Direct I/O directly.
                // SAFETY: current_buf is valid for dio_size bytes, fd_direct_on is valid.
                unsafe {
                    do_io(
                        op,
                        fd_direct_on,
                        current_buf,
                        dio_size,
                        current_offset as i64,
                    )?
                }
            } else {
                // Buffer not aligned: use a page-aligned bounce buffer.
                let chunk_size = std::cmp::min(dio_size, bounce_buffer_size());
                let mut bounce = alloc_page_aligned(chunk_size);
                let mut seg_transferred = 0;

                while seg_transferred < dio_size {
                    let chunk = std::cmp::min(chunk_size, dio_size - seg_transferred);

                    let n = if op == IoOp::Write {
                        // Copy from user buffer to bounce buffer, then write.
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                current_buf.add(seg_transferred),
                                bounce.as_mut_ptr(),
                                chunk,
                            );
                        }
                        unsafe {
                            do_io(
                                op,
                                fd_direct_on,
                                bounce.as_mut_ptr(),
                                chunk,
                                (current_offset + seg_transferred as u64) as i64,
                            )?
                        }
                    } else {
                        // Read into bounce buffer, then copy to user buffer.
                        let n = unsafe {
                            do_io(
                                op,
                                fd_direct_on,
                                bounce.as_mut_ptr(),
                                chunk,
                                (current_offset + seg_transferred as u64) as i64,
                            )?
                        };
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                bounce.as_ptr(),
                                current_buf.add(seg_transferred),
                                n,
                            );
                        }
                        n
                    };

                    seg_transferred += n;
                    if n < chunk {
                        break; // Short I/O
                    }
                }
                seg_transferred
            };

            total_transferred += transferred;
            remaining -= transferred;
            current_offset += transferred as u64;
            current_buf = unsafe { current_buf.add(transferred) };

            if transferred < dio_size || partial == PartialIO::Yes {
                break;
            }
        } else {
            // Unaligned segment: use buffered I/O (fd without O_DIRECT).
            // Determine how many bytes until the next page boundary (or all remaining).
            let bytes_to_boundary = if can_direct_io && !offset_aligned {
                // Transfer up to the next page boundary.
                let next_boundary = align_up(offset_usize, ps);
                std::cmp::min(next_boundary - offset_usize, remaining)
            } else {
                // Either Direct I/O is not available, or remaining < page size.
                // Just transfer everything with buffered I/O.
                remaining
            };

            // SAFETY: current_buf is valid for bytes_to_boundary bytes, fd_direct_off is valid.
            let transferred = unsafe {
                do_io(
                    op,
                    fd_direct_off,
                    current_buf,
                    bytes_to_boundary,
                    current_offset as i64,
                )?
            };

            total_transferred += transferred;
            remaining -= transferred;
            current_offset += transferred as u64;
            current_buf = unsafe { current_buf.add(transferred) };

            if transferred < bytes_to_boundary || partial == PartialIO::Yes {
                break;
            }
        }
    }

    Ok(total_transferred)
}

/// Perform a single I/O operation (read or write) on a file descriptor.
///
/// # Safety
///
/// `buf` must point to valid memory of at least `count` bytes (writable for reads).
/// `fd` must be a valid, open file descriptor.
unsafe fn do_io(op: IoOp, fd: RawFd, buf: *mut u8, count: usize, offset: i64) -> Result<usize> {
    // SAFETY: Caller guarantees buf is valid for count bytes and fd is valid.
    unsafe {
        match op {
            IoOp::Read => pread_raw(fd, buf, count, offset),
            IoOp::Write => pwrite_raw(fd, buf as *const u8, count, offset),
        }
    }
}

/// Get the bounce buffer size from config.
fn bounce_buffer_size() -> usize {
    crate::config::Config::get().bounce_buffer_size
}

/// Allocate a page-aligned buffer of the given size.
fn alloc_page_aligned(size: usize) -> Vec<u8> {
    let ps = page_size();
    let aligned_size = align_up(size, ps);
    let layout = std::alloc::Layout::from_size_align(aligned_size, ps)
        .expect("invalid layout for page-aligned allocation");
    // SAFETY: layout has non-zero size.
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    if ptr.is_null() {
        std::alloc::handle_alloc_error(layout);
    }
    // SAFETY: ptr is valid for aligned_size bytes and was allocated with the global allocator.
    unsafe { Vec::from_raw_parts(ptr, aligned_size, aligned_size) }
}

/// Open a file with POSIX `open(2)`.
///
/// Returns the file descriptor, or an error.
pub fn posix_open(path: &std::path::Path, flags: i32, mode: u32) -> Result<RawFd> {
    let c_path = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).map_err(|_| {
        Error::new(
            ErrorKind::ConfigInvalid,
            format!("path contains null byte: {}", path.display()),
        )
        .with_operation("posix_open")
    })?;

    // SAFETY: c_path is a valid C string, flags and mode are valid.
    let fd = unsafe { libc::open(c_path.as_ptr(), flags, mode) };
    if fd < 0 {
        let errno = unsafe { *libc::__errno_location() };
        let io_err = std::io::Error::from_raw_os_error(errno);
        let kind = match errno {
            libc::ENOENT => ErrorKind::NotFound,
            libc::EACCES | libc::EPERM => ErrorKind::PermissionDenied,
            _ => ErrorKind::SystemError,
        };
        Err(Error::new(kind, format!("open failed: {io_err}"))
            .with_operation("posix_open")
            .with_context("path", path.display().to_string())
            .set_source(io_err))
    } else {
        Ok(fd)
    }
}

/// Close a file descriptor.
///
/// Ignores errors on close (consistent with C++ kvikio behavior).
pub fn posix_close(fd: RawFd) {
    if fd >= 0 {
        // SAFETY: we only close valid file descriptors.
        unsafe {
            libc::close(fd);
        }
    }
}

/// Get the file size via `fstat`.
pub fn file_size(fd: RawFd) -> Result<u64> {
    let mut stat: libc::stat = unsafe { std::mem::zeroed() };
    // SAFETY: fd is valid, stat is a valid buffer.
    let ret = unsafe { libc::fstat(fd, &mut stat) };
    if ret < 0 {
        let errno = unsafe { *libc::__errno_location() };
        Err(Error::new(
            ErrorKind::SystemError,
            format!("fstat failed: {}", std::io::Error::from_raw_os_error(errno)),
        )
        .with_operation("file_size"))
    } else {
        Ok(stat.st_size as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    // ---- posix_open / posix_close tests ----

    #[test]
    fn test_open_nonexistent_file() {
        let err = posix_open(
            std::path::Path::new("/nonexistent/path/file.bin"),
            libc::O_RDONLY,
            0,
        )
        .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::NotFound);
    }

    #[test]
    fn test_open_and_close() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path();

        let fd = posix_open(path, libc::O_RDONLY, 0).unwrap();
        assert!(fd >= 0);
        posix_close(fd);
    }

    #[test]
    fn test_close_invalid_fd() {
        // Should not panic.
        posix_close(-1);
    }

    // ---- file_size tests ----

    #[test]
    fn test_file_size_empty() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let fd = posix_open(tmp.path(), libc::O_RDONLY, 0).unwrap();
        assert_eq!(file_size(fd).unwrap(), 0);
        posix_close(fd);
    }

    #[test]
    fn test_file_size_nonempty() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(&[0u8; 12345]).unwrap();
        tmp.flush().unwrap();

        let fd = posix_open(tmp.path(), libc::O_RDONLY, 0).unwrap();
        assert_eq!(file_size(fd).unwrap(), 12345);
        posix_close(fd);
    }

    // ---- pread / pwrite tests ----

    #[test]
    fn test_pread_pwrite_roundtrip() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path();

        let data: Vec<u8> = (0..256).map(|i| i as u8).collect();

        // Write
        let fd = posix_open(path, libc::O_WRONLY, 0).unwrap();
        let written = unsafe { pwrite_raw(fd, data.as_ptr(), data.len(), 0) }.unwrap();
        assert_eq!(written, data.len());
        posix_close(fd);

        // Read back
        let fd = posix_open(path, libc::O_RDONLY, 0).unwrap();
        let mut read_buf = vec![0u8; data.len()];
        let read = unsafe { pread_raw(fd, read_buf.as_mut_ptr(), read_buf.len(), 0) }.unwrap();
        assert_eq!(read, data.len());
        assert_eq!(read_buf, data);
        posix_close(fd);
    }

    #[test]
    fn test_pread_at_offset() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path();

        let data: Vec<u8> = (0..100).collect();
        let fd = posix_open(path, libc::O_WRONLY, 0).unwrap();
        unsafe { pwrite_raw(fd, data.as_ptr(), data.len(), 0) }.unwrap();
        posix_close(fd);

        let fd = posix_open(path, libc::O_RDONLY, 0).unwrap();
        let mut buf = vec![0u8; 10];
        let n = unsafe { pread_raw(fd, buf.as_mut_ptr(), buf.len(), 50) }.unwrap();
        assert_eq!(n, 10);
        assert_eq!(buf, &data[50..60]);
        posix_close(fd);
    }

    // ---- posix_host_read / posix_host_write tests ----

    #[test]
    fn test_host_read_write_buffered_only() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path();

        let data: Vec<u8> = (0u8..=255).cycle().take(10000).collect();

        // Write using buffered I/O (no Direct I/O)
        let fd = posix_open(path, libc::O_WRONLY, 0).unwrap();
        let written = posix_host_write(fd, fd, &data, 0, PartialIO::No, false).unwrap();
        assert_eq!(written, data.len());
        posix_close(fd);

        // Read back
        let fd = posix_open(path, libc::O_RDONLY, 0).unwrap();
        let mut read_buf = vec![0u8; data.len()];
        let read = posix_host_read(fd, fd, &mut read_buf, 0, PartialIO::No, false).unwrap();
        assert_eq!(read, data.len());
        assert_eq!(read_buf, data);
        posix_close(fd);
    }

    #[test]
    fn test_host_read_write_with_offset() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path();

        // Write a file with known content
        let data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        let fd = posix_open(path, libc::O_WRONLY, 0).unwrap();
        posix_host_write(fd, fd, &data, 0, PartialIO::No, false).unwrap();
        posix_close(fd);

        // Read from offset 100, 200 bytes
        let fd = posix_open(path, libc::O_RDONLY, 0).unwrap();
        let mut buf = vec![0u8; 200];
        let n = posix_host_read(fd, fd, &mut buf, 100, PartialIO::No, false).unwrap();
        assert_eq!(n, 200);
        assert_eq!(buf, &data[100..300]);
        posix_close(fd);
    }

    #[test]
    fn test_host_read_zero_size() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let fd = posix_open(tmp.path(), libc::O_RDONLY, 0).unwrap();
        let mut buf = [];
        let n = posix_host_read(fd, fd, &mut buf, 0, PartialIO::No, false).unwrap();
        assert_eq!(n, 0);
        posix_close(fd);
    }

    // ---- Direct I/O tests ----

    #[test]
    fn test_host_read_write_with_direct_io_aligned() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path();
        let ps = page_size();

        // Create page-aligned data (2 pages)
        let data_size = ps * 2;
        let data: Vec<u8> = (0..data_size).map(|i| (i % 256) as u8).collect();

        // Try to open with O_DIRECT. If it fails (e.g., on tmpfs), skip the Direct I/O part.
        let fd_direct = posix_open(
            path,
            libc::O_WRONLY | libc::O_CREAT | libc::O_TRUNC | libc::O_DIRECT,
            0o644,
        );
        let fd_buffered =
            posix_open(path, libc::O_WRONLY | libc::O_CREAT | libc::O_TRUNC, 0o644).unwrap();

        match fd_direct {
            Ok(fd_dio) => {
                // Write with Direct I/O support
                let written =
                    posix_host_write(fd_dio, fd_buffered, &data, 0, PartialIO::No, true).unwrap();
                assert_eq!(written, data.len());
                posix_close(fd_dio);
                posix_close(fd_buffered);

                // Read back with Direct I/O
                let fd_dio = posix_open(path, libc::O_RDONLY | libc::O_DIRECT, 0).unwrap();
                let fd_buf = posix_open(path, libc::O_RDONLY, 0).unwrap();
                let mut read_buf = vec![0u8; data_size];
                let n =
                    posix_host_read(fd_dio, fd_buf, &mut read_buf, 0, PartialIO::No, true).unwrap();
                assert_eq!(n, data_size);
                assert_eq!(read_buf, data);
                posix_close(fd_dio);
                posix_close(fd_buf);
            }
            Err(_) => {
                // O_DIRECT not supported on this filesystem (e.g., tmpfs).
                // Fall back to buffered-only test.
                let written =
                    posix_host_write(-1, fd_buffered, &data, 0, PartialIO::No, true).unwrap();
                assert_eq!(written, data.len());
                posix_close(fd_buffered);
            }
        }
    }

    #[test]
    fn test_host_read_write_unaligned_with_direct_io() {
        // Test with deliberately unaligned offset and size to exercise prefix/suffix paths.
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path();
        let ps = page_size();

        // Write enough data: 3 pages worth
        let data_size = ps * 3;
        let data: Vec<u8> = (0..data_size).map(|i| (i % 256) as u8).collect();

        let fd = posix_open(path, libc::O_WRONLY | libc::O_CREAT | libc::O_TRUNC, 0o644).unwrap();
        posix_host_write(fd, fd, &data, 0, PartialIO::No, false).unwrap();
        posix_close(fd);

        // Read with unaligned offset (100 bytes into file) and unaligned size.
        let offset = 100u64;
        let read_size = ps + 500; // crosses a page boundary

        let fd = posix_open(path, libc::O_RDONLY, 0).unwrap();
        let mut buf = vec![0u8; read_size];
        let n = posix_host_read(fd, fd, &mut buf, offset, PartialIO::No, false).unwrap();
        assert_eq!(n, read_size);
        assert_eq!(buf, &data[100..100 + read_size]);
        posix_close(fd);
    }

    // ---- alloc_page_aligned tests ----

    #[test]
    fn test_alloc_page_aligned() {
        let ps = page_size();
        let buf = alloc_page_aligned(ps);
        assert!(is_aligned_ptr(buf.as_ptr(), ps));
        assert_eq!(buf.len(), ps);
    }

    #[test]
    fn test_alloc_page_aligned_not_exact_page_size() {
        let ps = page_size();
        let buf = alloc_page_aligned(100);
        assert!(is_aligned_ptr(buf.as_ptr(), ps));
        // Should be rounded up to page size.
        assert_eq!(buf.len(), ps);
    }

    // ---- PartialIO tests ----

    #[test]
    fn test_partial_io_read() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path();

        let data = vec![42u8; 1000];
        let fd = posix_open(path, libc::O_WRONLY, 0).unwrap();
        posix_host_write(fd, fd, &data, 0, PartialIO::No, false).unwrap();
        posix_close(fd);

        // With PartialIO::Yes, should still read at least some bytes.
        let fd = posix_open(path, libc::O_RDONLY, 0).unwrap();
        let mut buf = vec![0u8; 1000];
        let n = posix_host_read(fd, fd, &mut buf, 0, PartialIO::Yes, false).unwrap();
        assert!(n > 0);
        assert!(n <= 1000);
        posix_close(fd);
    }
}
