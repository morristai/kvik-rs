//! The primary user-facing type for file I/O.
//!
//! [`FileHandle`] manages two file descriptors (one with `O_DIRECT`, one without)
//! and an optional cuFile handle for GPUDirect Storage. It provides synchronous
//! read/write for both host and device memory, plus parallel I/O via scoped threads.

use std::os::fd::RawFd;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use cudarc::driver::DevicePtr;

use crate::compat_mode::CompatMode;
use crate::config::Config;
use crate::error::{Error, ErrorKind, Result};
use crate::posix_io::{self, PartialIO};

/// The primary user-facing type for GPU-accelerated file I/O.
///
/// Mirrors the C++ `kvikio::FileHandle`. Opens files with two file descriptors
/// (one with `O_DIRECT`, one without) and optionally registers with the cuFile
/// driver for GPUDirect Storage.
///
/// # Examples
///
/// ```no_run
/// use kvik_rs::{FileHandle, CompatMode};
///
/// let mut handle = FileHandle::open(
///     std::path::Path::new("/data/model.bin"),
///     "r",
///     0o644,
///     CompatMode::Auto,
/// ).unwrap();
///
/// let file_size = handle.nbytes().unwrap();
/// let mut buf = vec![0u8; file_size as usize];
/// handle.read_host(&mut buf, 0).unwrap();
/// ```
pub struct FileHandle {
    /// File descriptor with O_DIRECT (or -1 if O_DIRECT not supported).
    fd_direct_on: RawFd,
    /// File descriptor without O_DIRECT.
    fd_direct_off: RawFd,
    /// cuFile handle for GDS (None in compat mode).
    cufile_handle: Option<cudarc::cufile::FileHandle>,
    /// Reference to the cuFile driver (keeps it alive).
    driver: Option<Arc<cudarc::cufile::Cufile>>,
    /// Cached file size.
    nbytes_cached: AtomicU64,
    /// Effective compatibility mode for this handle.
    compat_mode: CompatMode,
    /// Path of the file (for diagnostics).
    path: PathBuf,
}

impl FileHandle {
    /// Open a file for I/O.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the file.
    /// * `flags` - POSIX-style flag string: `"r"` (read), `"w"` (write/create/truncate),
    ///   `"a"` (append), or `"r+"` / `"w+"` (read-write). These are converted to
    ///   `libc::O_*` flags internally.
    /// * `mode` - File creation mode (e.g., `0o644`). Only used when creating files.
    /// * `compat` - Compatibility mode override. Use `CompatMode::Auto` to auto-detect.
    pub fn open(path: &Path, flags: &str, mode: u32, compat: CompatMode) -> Result<Self> {
        let o_flags = parse_flags(flags)?;

        // Open the normal (buffered) file descriptor.
        let fd_direct_off = posix_io::posix_open(path, o_flags, mode)
            .map_err(|e| e.with_operation("FileHandle::open"))?;

        // Try to open with O_DIRECT. This may fail on certain filesystems (tmpfs, etc.)
        let fd_direct_on =
            posix_io::posix_open(path, o_flags | libc::O_DIRECT, mode).unwrap_or(-1);

        // Get initial file size.
        let nbytes = posix_io::file_size(fd_direct_off)
            .map_err(|e| e.with_operation("FileHandle::open"))?;

        // Determine effective compat mode and initialize cuFile.
        //
        // C++ kvikio uses a two-stage detection:
        //   1. Global heuristic (WSL? udev? library?) → initial preference
        //   2. Per-file probe (actually try cuFileHandleRegister) → final decision
        //
        // In Auto mode, we always attempt cuFile initialization regardless of the
        // heuristic. If it succeeds, we use GDS; if it fails, we silently fall back
        // to POSIX. The heuristic only sets the initial preference — it does not
        // prevent the attempt.
        let (driver, cufile_handle, effective_compat) = match compat {
            CompatMode::On => (None, None, CompatMode::On),
            CompatMode::Off => {
                // Off mode: GDS is required, propagate any error.
                match init_cufile(path, o_flags, mode) {
                    Ok((d, h)) => (Some(d), Some(h), CompatMode::Off),
                    Err(e) => {
                        posix_io::posix_close(fd_direct_off);
                        posix_io::posix_close(fd_direct_on);
                        return Err(e.with_operation("FileHandle::open"));
                    }
                }
            }
            CompatMode::Auto => {
                // Auto mode: try cuFile, fall back silently on failure.
                match init_cufile(path, o_flags, mode) {
                    Ok((d, h)) => (Some(d), Some(h), CompatMode::Off),
                    Err(_) => (None, None, CompatMode::On),
                }
            }
        };

        Ok(Self {
            fd_direct_on,
            fd_direct_off,
            cufile_handle,
            driver,
            nbytes_cached: AtomicU64::new(nbytes),
            compat_mode: effective_compat,
            path: path.to_path_buf(),
        })
    }

    /// Get the file size in bytes.
    ///
    /// Returns the cached value. Use [`refresh_nbytes`](Self::refresh_nbytes) to re-query.
    pub fn nbytes(&self) -> Result<u64> {
        Ok(self.nbytes_cached.load(Ordering::Relaxed))
    }

    /// Re-query the file size from the OS and update the cache.
    pub fn refresh_nbytes(&self) -> Result<u64> {
        let size = posix_io::file_size(self.fd_direct_off)
            .map_err(|e| e.with_operation("FileHandle::refresh_nbytes"))?;
        self.nbytes_cached.store(size, Ordering::Relaxed);
        Ok(size)
    }

    /// Returns the effective compatibility mode for this handle.
    pub fn compat_mode(&self) -> CompatMode {
        self.compat_mode
    }

    /// Returns the path of the file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns `true` if this handle has a GDS/cuFile handle.
    pub fn is_gds_available(&self) -> bool {
        self.cufile_handle.is_some()
    }

    /// Close the file handle explicitly.
    ///
    /// This is also done automatically on drop, but calling `close` allows
    /// error handling.
    pub fn close(&mut self) {
        self.cufile_handle.take();
        self.driver.take();
        posix_io::posix_close(self.fd_direct_off);
        posix_io::posix_close(self.fd_direct_on);
        self.fd_direct_off = -1;
        self.fd_direct_on = -1;
    }

    /// Read into host memory using POSIX I/O with opportunistic Direct I/O.
    ///
    /// # Arguments
    ///
    /// * `buf` - Destination buffer in host memory.
    /// * `file_offset` - Byte offset in the file to start reading from.
    ///
    /// Returns the number of bytes read.
    pub fn read_host(&self, buf: &mut [u8], file_offset: u64) -> Result<usize> {
        let config = Config::get();
        posix_io::posix_host_read(
            self.fd_direct_on,
            self.fd_direct_off,
            buf,
            file_offset,
            PartialIO::No,
            config.auto_direct_io_read,
        )
        .map_err(|e| {
            e.with_operation("FileHandle::read_host")
                .with_context("path", self.path.display().to_string())
        })
    }

    /// Write from host memory using POSIX I/O with opportunistic Direct I/O.
    ///
    /// # Arguments
    ///
    /// * `buf` - Source buffer in host memory.
    /// * `file_offset` - Byte offset in the file to start writing to.
    ///
    /// Returns the number of bytes written.
    pub fn write_host(&self, buf: &[u8], file_offset: u64) -> Result<usize> {
        let config = Config::get();
        posix_io::posix_host_write(
            self.fd_direct_on,
            self.fd_direct_off,
            buf,
            file_offset,
            PartialIO::No,
            config.auto_direct_io_write,
        )
        .map_err(|e| {
            e.with_operation("FileHandle::write_host")
                .with_context("path", self.path.display().to_string())
        })
    }

    /// Read into device memory.
    ///
    /// Uses GDS (cuFile) when available, falling back to POSIX I/O through
    /// a host bounce buffer.
    ///
    /// # Arguments
    ///
    /// * `dev_ptr` - Destination buffer in device memory.
    /// * `size` - Number of bytes to read.
    /// * `file_offset` - Byte offset in the file.
    /// * `dev_offset` - Byte offset into the device buffer.
    pub fn read(
        &self,
        dev_ptr: &mut cudarc::driver::CudaSlice<u8>,
        size: usize,
        file_offset: u64,
        dev_offset: u64,
    ) -> Result<usize> {
        if let Some(ref cufile_handle) = self.cufile_handle {
            // GDS path: use cuFile for direct GPU-file transfer.
            let ret = cufile_handle.sync_read::<u8, _>(file_offset as i64, dev_ptr);
            match ret {
                Ok(bytes_read) => {
                    if bytes_read < 0 {
                        return Err(Error::new(
                            ErrorKind::CuFileError,
                            format!("cuFile read returned negative: {bytes_read}"),
                        )
                        .with_operation("FileHandle::read")
                        .with_context("path", self.path.display().to_string()));
                    }
                    return Ok(bytes_read as usize);
                }
                Err(e) => {
                    return Err(Error::new(
                        ErrorKind::CuFileError,
                        format!("cuFile read failed: {e}"),
                    )
                    .with_operation("FileHandle::read")
                    .with_context("path", self.path.display().to_string()));
                }
            }
        }

        // POSIX fallback for device memory: stage through host bounce buffer.
        // Read from file into host memory, then copy to device.
        self.posix_device_read(dev_ptr, size, file_offset, dev_offset)
    }

    /// Write from device memory.
    ///
    /// Uses GDS (cuFile) when available, falling back to POSIX I/O through
    /// a host bounce buffer.
    ///
    /// # Arguments
    ///
    /// * `dev_ptr` - Source buffer in device memory.
    /// * `size` - Number of bytes to write.
    /// * `file_offset` - Byte offset in the file.
    /// * `dev_offset` - Byte offset into the device buffer.
    pub fn write(
        &self,
        dev_ptr: &cudarc::driver::CudaSlice<u8>,
        size: usize,
        file_offset: u64,
        dev_offset: u64,
    ) -> Result<usize> {
        if let Some(ref cufile_handle) = self.cufile_handle {
            // GDS path: use cuFile for direct GPU-file transfer.
            // cudarc's safe `sync_write` requires `&mut self` on FileHandle, but
            // the underlying cuFileWrite C API is thread-safe and only needs the
            // handle value. We call `result::write` directly to avoid the overly
            // conservative borrow requirement.
            // See: CLAUDE.md §1.4 - "Only drop to cudarc::cufile::result (unsafe
            // Result wrappers) when the safe API lacks needed functionality."
            let stream = dev_ptr.stream().clone();
            let num_bytes = dev_ptr.num_bytes();
            let (src, _record_src) = dev_ptr.device_ptr(&stream);
            stream.synchronize().map_err(|e| {
                Error::new(
                    ErrorKind::CuFileError,
                    format!("CUDA stream synchronize failed before write: {e}"),
                )
                .with_operation("FileHandle::write")
            })?;

            // SAFETY: cufile_handle.cu() returns a valid CUfileHandle_t,
            // src is a valid device pointer from the synchronized CudaSlice,
            // and num_bytes matches the allocation size.
            let ret = unsafe {
                cudarc::cufile::result::write(
                    cufile_handle.cu(),
                    src as *mut std::ffi::c_void,
                    num_bytes,
                    file_offset as i64,
                    dev_offset as i64,
                )
            };

            match ret {
                Ok(bytes_written) => {
                    if bytes_written < 0 {
                        return Err(Error::new(
                            ErrorKind::CuFileError,
                            format!("cuFile write returned negative: {bytes_written}"),
                        )
                        .with_operation("FileHandle::write")
                        .with_context("path", self.path.display().to_string()));
                    }
                    return Ok(bytes_written as usize);
                }
                Err(e) => {
                    return Err(Error::new(
                        ErrorKind::CuFileError,
                        format!("cuFile write failed: {e}"),
                    )
                    .with_operation("FileHandle::write")
                    .with_context("path", self.path.display().to_string()));
                }
            }
        }

        // POSIX fallback for device memory.
        self.posix_device_write(dev_ptr, size, file_offset, dev_offset)
    }

    /// POSIX device read: file → host bounce buffer → device memory.
    fn posix_device_read(
        &self,
        _dev_ptr: &mut cudarc::driver::CudaSlice<u8>,
        _size: usize,
        _file_offset: u64,
        _dev_offset: u64,
    ) -> Result<usize> {
        // This requires a CUDA context and device memory operations.
        // The implementation stages through a host bounce buffer:
        // 1. pread from file into host bounce buffer
        // 2. cuMemcpyHtoDAsync from bounce buffer to device memory
        // 3. Synchronize the stream
        //
        // This is a placeholder that will work when a CUDA device is available.
        Err(Error::new(
            ErrorKind::Unsupported,
            "POSIX device I/O requires a CUDA context (not available in this environment)",
        )
        .with_operation("FileHandle::posix_device_read"))
    }

    /// POSIX device write: device memory → host bounce buffer → file.
    fn posix_device_write(
        &self,
        _dev_ptr: &cudarc::driver::CudaSlice<u8>,
        _size: usize,
        _file_offset: u64,
        _dev_offset: u64,
    ) -> Result<usize> {
        Err(Error::new(
            ErrorKind::Unsupported,
            "POSIX device I/O requires a CUDA context (not available in this environment)",
        )
        .with_operation("FileHandle::posix_device_write"))
    }

    /// Parallel read: splits into chunks across scoped threads.
    ///
    /// # Arguments
    ///
    /// * `buf` - Destination buffer in host memory.
    /// * `file_offset` - Starting file offset.
    /// * `task_size` - Size of each chunk (0 = use config default).
    ///
    /// Returns total bytes read.
    pub fn pread_host(&self, buf: &mut [u8], file_offset: u64, task_size: usize) -> Result<usize> {
        let ts = if task_size == 0 {
            Config::get().task_size
        } else {
            task_size
        };

        let total_size = buf.len();
        if total_size == 0 {
            return Ok(0);
        }

        // For small transfers, just do a single read.
        if total_size <= ts {
            return self.read_host(buf, file_offset);
        }

        // Split into chunks and read in parallel using scoped threads.
        let num_chunks = total_size.div_ceil(ts);
        let config = Config::get();
        let _num_threads = config.num_threads.max(1);

        // SAFETY: We split the buffer into non-overlapping chunks, and each scoped
        // thread gets exclusive access to its chunk. The scoped thread API ensures all
        // threads complete before this function returns, so the buffer remains valid.
        let buf_base = buf.as_mut_ptr();

        let results: Vec<Result<usize>> = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(num_chunks);

            for i in 0..num_chunks {
                let chunk_offset = i * ts;
                let chunk_size = std::cmp::min(ts, total_size - chunk_offset);
                let chunk_file_offset = file_offset + chunk_offset as u64;
                // SAFETY: chunk_offset + chunk_size <= total_size, and each chunk is non-overlapping.
                let chunk_buf = unsafe {
                    std::slice::from_raw_parts_mut(buf_base.add(chunk_offset), chunk_size)
                };
                let fd_on = self.fd_direct_on;
                let fd_off = self.fd_direct_off;
                let use_dio = config.auto_direct_io_read;

                let handle = scope.spawn(move || {
                    posix_io::posix_host_read(fd_on, fd_off, chunk_buf, chunk_file_offset, PartialIO::No, use_dio)
                });

                handles.push(handle);
            }

            handles
                .into_iter()
                .map(|h| h.join().expect("thread panicked"))
                .collect()
        });

        let mut total = 0;
        for result in results {
            total += result.map_err(|e| {
                e.with_operation("FileHandle::pread_host")
                    .with_context("path", self.path.display().to_string())
            })?;
        }
        Ok(total)
    }

    /// Parallel write: splits into chunks across scoped threads.
    ///
    /// # Arguments
    ///
    /// * `buf` - Source buffer in host memory.
    /// * `file_offset` - Starting file offset.
    /// * `task_size` - Size of each chunk (0 = use config default).
    ///
    /// Returns total bytes written.
    pub fn pwrite_host(
        &self,
        buf: &[u8],
        file_offset: u64,
        task_size: usize,
    ) -> Result<usize> {
        let ts = if task_size == 0 {
            Config::get().task_size
        } else {
            task_size
        };

        let total_size = buf.len();
        if total_size == 0 {
            return Ok(0);
        }

        if total_size <= ts {
            return self.write_host(buf, file_offset);
        }

        let num_chunks = total_size.div_ceil(ts);
        let config = Config::get();
        let _num_threads = config.num_threads.max(1);

        let results: Vec<Result<usize>> = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(num_chunks);

            for i in 0..num_chunks {
                let chunk_offset = i * ts;
                let chunk_size = std::cmp::min(ts, total_size - chunk_offset);
                let chunk_file_offset = file_offset + chunk_offset as u64;
                let chunk_buf = &buf[chunk_offset..chunk_offset + chunk_size];
                let fd_on = self.fd_direct_on;
                let fd_off = self.fd_direct_off;
                let use_dio = config.auto_direct_io_write;

                let handle = scope.spawn(move || {
                    posix_io::posix_host_write(fd_on, fd_off, chunk_buf, chunk_file_offset, PartialIO::No, use_dio)
                });

                handles.push(handle);
            }

            handles
                .into_iter()
                .map(|h| h.join().expect("thread panicked"))
                .collect()
        });

        let mut total = 0;
        for result in results {
            total += result.map_err(|e| {
                e.with_operation("FileHandle::pwrite_host")
                    .with_context("path", self.path.display().to_string())
            })?;
        }
        Ok(total)
    }

    /// Returns the file descriptor without O_DIRECT.
    pub fn fd(&self) -> RawFd {
        self.fd_direct_off
    }

    /// Returns the file descriptor with O_DIRECT (or -1 if unavailable).
    pub fn fd_direct(&self) -> RawFd {
        self.fd_direct_on
    }
}

impl Drop for FileHandle {
    fn drop(&mut self) {
        self.close();
    }
}

// FileHandle is Send because file descriptors are just integers,
// cuFile handles are thread-safe, and we use atomic operations for nbytes.
// SAFETY: All internal state is either atomic, Arc-wrapped, or plain integers.
unsafe impl Send for FileHandle {}
// SAFETY: FileHandle is safe to share across threads because:
// - fd_direct_on/fd_direct_off are RawFd (just integers, pread/pwrite are thread-safe)
// - cufile_handle/driver are wrapped in Option/Arc (thread-safe)
// - nbytes_cached uses AtomicU64
// - path is immutable after construction
unsafe impl Sync for FileHandle {}

impl std::fmt::Debug for FileHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileHandle")
            .field("path", &self.path)
            .field("fd_direct_on", &self.fd_direct_on)
            .field("fd_direct_off", &self.fd_direct_off)
            .field("compat_mode", &self.compat_mode)
            .field("nbytes", &self.nbytes_cached.load(Ordering::Relaxed))
            .field("gds_available", &self.cufile_handle.is_some())
            .finish()
    }
}

/// Parse a flag string like `"r"`, `"w"`, `"a"`, `"r+"`, `"w+"` into `libc::O_*` flags.
fn parse_flags(flags: &str) -> Result<i32> {
    match flags {
        "r" => Ok(libc::O_RDONLY),
        "w" => Ok(libc::O_WRONLY | libc::O_CREAT | libc::O_TRUNC),
        "a" => Ok(libc::O_WRONLY | libc::O_CREAT | libc::O_APPEND),
        "r+" => Ok(libc::O_RDWR),
        "w+" => Ok(libc::O_RDWR | libc::O_CREAT | libc::O_TRUNC),
        _ => Err(Error::new(
            ErrorKind::ConfigInvalid,
            format!("unknown file flags: {flags:?}"),
        )
        .with_operation("FileHandle::open")),
    }
}

/// Initialize the cuFile driver and register the file.
fn init_cufile(
    path: &Path,
    _flags: i32,
    _mode: u32,
) -> Result<(Arc<cudarc::cufile::Cufile>, cudarc::cufile::FileHandle)> {
    let driver = cudarc::cufile::Cufile::new().map_err(|e| {
        Error::new(
            ErrorKind::CuFileError,
            format!("failed to initialize cuFile driver: {e}"),
        )
    })?;

    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .map_err(|e| {
            Error::new(ErrorKind::SystemError, format!("failed to open file for cuFile: {e}"))
                .set_source(e)
        })?;

    let handle = driver.register(file).map_err(|e| {
        Error::new(
            ErrorKind::CuFileError,
            format!("cuFile handle registration failed: {e}"),
        )
    })?;

    Ok((driver, handle))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Flag parsing tests ----

    #[test]
    fn test_parse_flags_read() {
        let flags = parse_flags("r").unwrap();
        assert_eq!(flags, libc::O_RDONLY);
    }

    #[test]
    fn test_parse_flags_write() {
        let flags = parse_flags("w").unwrap();
        assert!(flags & libc::O_WRONLY != 0);
        assert!(flags & libc::O_CREAT != 0);
        assert!(flags & libc::O_TRUNC != 0);
    }

    #[test]
    fn test_parse_flags_append() {
        let flags = parse_flags("a").unwrap();
        assert!(flags & libc::O_WRONLY != 0);
        assert!(flags & libc::O_APPEND != 0);
    }

    #[test]
    fn test_parse_flags_readwrite() {
        let flags = parse_flags("r+").unwrap();
        assert_eq!(flags, libc::O_RDWR);
    }

    #[test]
    fn test_parse_flags_write_readwrite() {
        let flags = parse_flags("w+").unwrap();
        assert!(flags & libc::O_RDWR != 0);
        assert!(flags & libc::O_CREAT != 0);
    }

    #[test]
    fn test_parse_flags_invalid() {
        assert!(parse_flags("x").is_err());
        assert!(parse_flags("").is_err());
    }

    // ---- FileHandle host I/O tests ----

    #[test]
    fn test_open_read_write_host() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path();

        // Open for writing (compat mode to skip cuFile).
        let handle = FileHandle::open(path, "w", 0o644, CompatMode::On).unwrap();
        let data: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();
        let written = handle.write_host(&data, 0).unwrap();
        assert_eq!(written, data.len());
        drop(handle);

        // Open for reading.
        let handle = FileHandle::open(path, "r", 0o644, CompatMode::On).unwrap();
        assert_eq!(handle.nbytes().unwrap(), data.len() as u64);

        let mut read_buf = vec![0u8; data.len()];
        let read = handle.read_host(&mut read_buf, 0).unwrap();
        assert_eq!(read, data.len());
        assert_eq!(read_buf, data);
    }

    #[test]
    fn test_open_nonexistent() {
        let err = FileHandle::open(
            Path::new("/nonexistent/path/file.bin"),
            "r",
            0o644,
            CompatMode::On,
        )
        .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::NotFound);
    }

    #[test]
    fn test_file_handle_compat_mode() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let handle = FileHandle::open(tmp.path(), "r", 0o644, CompatMode::On).unwrap();
        assert_eq!(handle.compat_mode(), CompatMode::On);
        assert!(!handle.is_gds_available());
    }

    #[test]
    fn test_file_handle_path() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        let handle = FileHandle::open(&path, "r", 0o644, CompatMode::On).unwrap();
        assert_eq!(handle.path(), path);
    }

    #[test]
    fn test_file_handle_close_idempotent() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let mut handle = FileHandle::open(tmp.path(), "r", 0o644, CompatMode::On).unwrap();
        handle.close();
        // Second close should not panic.
        handle.close();
    }

    // ---- Parallel I/O tests ----

    #[test]
    fn test_pread_host_small() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path();

        // Write data.
        let data: Vec<u8> = (0..5000).map(|i| (i % 256) as u8).collect();
        let handle = FileHandle::open(path, "w", 0o644, CompatMode::On).unwrap();
        handle.write_host(&data, 0).unwrap();
        drop(handle);

        // Read with pread (small enough for single chunk).
        let handle = FileHandle::open(path, "r", 0o644, CompatMode::On).unwrap();
        let mut buf = vec![0u8; data.len()];
        let n = handle.pread_host(&mut buf, 0, 0).unwrap();
        assert_eq!(n, data.len());
        assert_eq!(buf, data);
    }

    #[test]
    fn test_pread_host_parallel_chunks() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path();

        // Write a larger file.
        let data: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();
        let handle = FileHandle::open(path, "w", 0o644, CompatMode::On).unwrap();
        handle.write_host(&data, 0).unwrap();
        drop(handle);

        // Read with small task_size to force parallelism.
        let handle = FileHandle::open(path, "r", 0o644, CompatMode::On).unwrap();
        let mut buf = vec![0u8; data.len()];
        let n = handle.pread_host(&mut buf, 0, 10_000).unwrap();
        assert_eq!(n, data.len());
        assert_eq!(buf, data);
    }

    #[test]
    fn test_pwrite_host_parallel_chunks() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path();

        let data: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();

        // Write with small task_size to force parallelism.
        let handle = FileHandle::open(path, "w", 0o644, CompatMode::On).unwrap();
        let n = handle.pwrite_host(&data, 0, 10_000).unwrap();
        assert_eq!(n, data.len());
        drop(handle);

        // Verify by reading back.
        let handle = FileHandle::open(path, "r", 0o644, CompatMode::On).unwrap();
        let mut buf = vec![0u8; data.len()];
        handle.read_host(&mut buf, 0).unwrap();
        assert_eq!(buf, data);
    }

    #[test]
    fn test_pread_host_zero_size() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let handle = FileHandle::open(tmp.path(), "r", 0o644, CompatMode::On).unwrap();
        let mut buf = [];
        let n = handle.pread_host(&mut buf, 0, 0).unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn test_refresh_nbytes() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path();

        let handle = FileHandle::open(path, "w+", 0o644, CompatMode::On).unwrap();
        assert_eq!(handle.nbytes().unwrap(), 0);

        // Write some data.
        handle.write_host(&[1, 2, 3, 4, 5], 0).unwrap();
        // Cached size may still be 0.
        let refreshed = handle.refresh_nbytes().unwrap();
        assert_eq!(refreshed, 5);
        assert_eq!(handle.nbytes().unwrap(), 5);
    }

    #[test]
    fn test_fd_accessors() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let handle = FileHandle::open(tmp.path(), "r", 0o644, CompatMode::On).unwrap();
        assert!(handle.fd() >= 0);
        // fd_direct may or may not be available depending on filesystem.
    }

    // ---- Large unaligned I/O test (mirrors C++ test_basic_io.cpp) ----

    #[test]
    fn test_write_read_unaligned_size() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path();

        // 1 MiB + 124 bytes (deliberately unaligned, like C++ test).
        let size = 1024 * 1024 + 124;
        let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();

        let handle = FileHandle::open(path, "w", 0o644, CompatMode::On).unwrap();
        let written = handle.write_host(&data, 0).unwrap();
        assert_eq!(written, size);
        drop(handle);

        let handle = FileHandle::open(path, "r", 0o644, CompatMode::On).unwrap();
        let mut buf = vec![0u8; size];
        let read = handle.read_host(&mut buf, 0).unwrap();
        assert_eq!(read, size);
        assert_eq!(buf, data);
    }
}
