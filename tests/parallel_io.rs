//! Parallel I/O integration tests.
//!
//! Mirrors C++ kvikio's threadpool test in `test_basic_io.cpp`.
//! Tests pread_host/pwrite_host with various task sizes and thread configurations.

mod test_utils;

use kvik_rs::{CompatMode, Config, FileHandle};

use test_utils::{assert_data_eq, gen_data};

// ---- Various task sizes for pread_host ----

#[test]
fn test_pread_host_task_size_256() {
    pread_host_with_task_size(256);
}

#[test]
fn test_pread_host_task_size_1024() {
    pread_host_with_task_size(1024);
}

#[test]
fn test_pread_host_task_size_4096() {
    pread_host_with_task_size(4096);
}

#[test]
fn test_pread_host_task_size_default() {
    // task_size=0 means use Config default.
    pread_host_with_task_size(0);
}

fn pread_host_with_task_size(task_size: usize) {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path();
    let size = 100_000;
    let data = gen_data(size);

    let handle = FileHandle::open(path, "w", 0o644, CompatMode::On).unwrap();
    handle.write_host(&data, 0).unwrap();
    drop(handle);

    let handle = FileHandle::open(path, "r", 0o644, CompatMode::On).unwrap();
    let mut buf = vec![0u8; size];
    let read = handle.pread_host(&mut buf, 0, task_size).unwrap();
    assert_eq!(read, size);
    assert_data_eq(&data, &buf);
}

// ---- Various task sizes for pwrite_host ----

#[test]
fn test_pwrite_host_task_size_256() {
    pwrite_host_with_task_size(256);
}

#[test]
fn test_pwrite_host_task_size_1024() {
    pwrite_host_with_task_size(1024);
}

#[test]
fn test_pwrite_host_task_size_4096() {
    pwrite_host_with_task_size(4096);
}

#[test]
fn test_pwrite_host_task_size_default() {
    pwrite_host_with_task_size(0);
}

fn pwrite_host_with_task_size(task_size: usize) {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path();
    let size = 100_000;
    let data = gen_data(size);

    let handle = FileHandle::open(path, "w", 0o644, CompatMode::On).unwrap();
    let written = handle.pwrite_host(&data, 0, task_size).unwrap();
    assert_eq!(written, size);
    drop(handle);

    // Verify by reading back.
    let handle = FileHandle::open(path, "r", 0o644, CompatMode::On).unwrap();
    let mut buf = vec![0u8; size];
    handle.read_host(&mut buf, 0).unwrap();
    assert_data_eq(&data, &buf);
}

// ---- Custom thread counts ----

#[test]
fn test_parallel_io_with_1_thread() {
    parallel_io_with_thread_count(1);
}

#[test]
fn test_parallel_io_with_4_threads() {
    parallel_io_with_thread_count(4);
}

#[test]
fn test_parallel_io_with_16_threads() {
    parallel_io_with_thread_count(16);
}

fn parallel_io_with_thread_count(num_threads: usize) {
    let saved = Config::get();
    Config::update(|c| c.num_threads = num_threads);

    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path();
    let size = 200_000;
    let data = gen_data(size);

    // Write.
    let handle = FileHandle::open(path, "w", 0o644, CompatMode::On).unwrap();
    let written = handle.pwrite_host(&data, 0, 10_000).unwrap();
    assert_eq!(written, size);
    drop(handle);

    // Read.
    let handle = FileHandle::open(path, "r", 0o644, CompatMode::On).unwrap();
    let mut buf = vec![0u8; size];
    let read = handle.pread_host(&mut buf, 0, 10_000).unwrap();
    assert_eq!(read, size);
    assert_data_eq(&data, &buf);

    Config::set(saved);
}

// ---- Read from multiple files concurrently (mirrors basic_no_cuda.cpp pattern) ----

#[test]
fn test_parallel_read_multiple_files() {
    let tmp1 = tempfile::NamedTempFile::new().unwrap();
    let tmp2 = tempfile::NamedTempFile::new().unwrap();
    let size = 50_000;
    let data1 = gen_data(size);
    let data2: Vec<u8> = (0..size).map(|i| ((i + 128) % 256) as u8).collect();

    // Write both files.
    let h1 = FileHandle::open(tmp1.path(), "w", 0o644, CompatMode::On).unwrap();
    h1.write_host(&data1, 0).unwrap();
    drop(h1);

    let h2 = FileHandle::open(tmp2.path(), "w", 0o644, CompatMode::On).unwrap();
    h2.write_host(&data2, 0).unwrap();
    drop(h2);

    // Read both files concurrently using separate threads.
    let path1 = tmp1.path().to_path_buf();
    let path2 = tmp2.path().to_path_buf();

    std::thread::scope(|scope| {
        let handle1 = scope.spawn(move || {
            let h = FileHandle::open(&path1, "r", 0o644, CompatMode::On).unwrap();
            let mut buf = vec![0u8; size];
            let n = h.pread_host(&mut buf, 0, 10_000).unwrap();
            assert_eq!(n, size);
            buf
        });

        let handle2 = scope.spawn(move || {
            let h = FileHandle::open(&path2, "r", 0o644, CompatMode::On).unwrap();
            let mut buf = vec![0u8; size];
            let n = h.pread_host(&mut buf, 0, 10_000).unwrap();
            assert_eq!(n, size);
            buf
        });

        let buf1 = handle1.join().unwrap();
        let buf2 = handle2.join().unwrap();

        assert_data_eq(&data1, &buf1);
        assert_data_eq(&data2, &buf2);
    });
}

// ---- Large file with small task_size (many thread spawns) ----

#[test]
fn test_parallel_large_file() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path();

    // ~2 MB file, 4 KB task_size = ~512 thread spawns.
    let size = 2 * 1024 * 1024;
    let data = gen_data(size);

    let handle = FileHandle::open(path, "w", 0o644, CompatMode::On).unwrap();
    let written = handle.pwrite_host(&data, 0, 4096).unwrap();
    assert_eq!(written, size);
    drop(handle);

    let handle = FileHandle::open(path, "r", 0o644, CompatMode::On).unwrap();
    let mut buf = vec![0u8; size];
    let read = handle.pread_host(&mut buf, 0, 4096).unwrap();
    assert_eq!(read, size);
    assert_data_eq(&data, &buf);
}

// ---- Parallel write + read round-trip at non-zero offset ----

#[test]
fn test_parallel_io_at_offset() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path();
    let size = 100_000;
    let data = gen_data(size);

    // Write at offset 0.
    let handle = FileHandle::open(path, "w", 0o644, CompatMode::On).unwrap();
    handle.pwrite_host(&data, 0, 10_000).unwrap();
    drop(handle);

    // Read with offset.
    let handle = FileHandle::open(path, "r", 0o644, CompatMode::On).unwrap();
    let offset = 25_000usize;
    let read_size = 50_000;
    let mut buf = vec![0u8; read_size];
    let read = handle.pread_host(&mut buf, offset as u64, 10_000).unwrap();
    assert_eq!(read, read_size);
    assert_data_eq(&data[offset..offset + read_size], &buf);
}

// ---- Zero-size parallel operations ----

#[test]
fn test_parallel_io_zero_size() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let handle = FileHandle::open(tmp.path(), "w+", 0o644, CompatMode::On).unwrap();

    let mut empty = [];
    assert_eq!(handle.pread_host(&mut empty, 0, 1024).unwrap(), 0);
    assert_eq!(handle.pwrite_host(&[], 0, 1024).unwrap(), 0);
}

// ---- Parallel write followed by parallel read, same handle ----

#[test]
fn test_parallel_write_then_read_same_handle() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path();
    let size = 80_000;
    let data = gen_data(size);

    let handle = FileHandle::open(path, "w+", 0o644, CompatMode::On).unwrap();
    let written = handle.pwrite_host(&data, 0, 8_000).unwrap();
    assert_eq!(written, size);

    let mut buf = vec![0u8; size];
    let read = handle.pread_host(&mut buf, 0, 8_000).unwrap();
    assert_eq!(read, size);
    assert_data_eq(&data, &buf);
}
