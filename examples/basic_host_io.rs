//! Basic host-memory I/O example.
//!
//! Demonstrates simple read/write operations using `FileHandle` with host memory.
//! No CUDA device required — runs in POSIX compatibility mode.
//!
//! Mirrors C++ kvikio's `basic_no_cuda.cpp` example.
//!
//! # Usage
//!
//! ```sh
//! cargo run --example basic_host_io
//! ```

use std::time::Instant;

use kvik_rs::{CompatMode, Config, FileHandle};

fn main() {
    println!("kvik-rs: Basic Host I/O Example");
    println!("================================\n");

    // Print configuration.
    let config = Config::get();
    println!("Configuration:");
    println!("  compat_mode:  {}", config.compat_mode);
    println!("  num_threads:  {}", config.num_threads);
    println!("  task_size:    {} bytes", config.task_size);
    println!();

    // Parameters (mirrors C++ NELEM=1024, SIZE=NELEM*sizeof(int)=4096).
    let nelem: usize = 1024;
    let size = nelem * std::mem::size_of::<i32>();
    println!("Test parameters:");
    println!("  elements:     {nelem}");
    println!("  total size:   {size} bytes");
    println!();

    // Generate test data: sequential integers as bytes.
    let data: Vec<u8> = (0..nelem)
        .flat_map(|i| (i as i32).to_ne_bytes())
        .collect();
    assert_eq!(data.len(), size);

    // Create a temporary file.
    let tmp = tempfile::NamedTempFile::new().expect("failed to create temp file");
    let path = tmp.path();
    println!("Temp file: {}\n", path.display());

    // ---- Write ----
    let start = Instant::now();
    let handle = FileHandle::open(path, "w", 0o644, CompatMode::On)
        .expect("failed to open file for writing");
    let written = handle
        .write_host(&data, 0)
        .expect("write_host failed");
    drop(handle);
    let write_elapsed = start.elapsed();

    assert_eq!(written, size, "short write");
    println!(
        "Write: {written} bytes in {:.1} us",
        write_elapsed.as_micros()
    );

    // ---- Verify file size ----
    let handle = FileHandle::open(path, "r", 0o644, CompatMode::On)
        .expect("failed to open file for reading");
    let file_size = handle.nbytes().expect("nbytes failed");
    assert_eq!(file_size, size as u64, "file size mismatch");
    println!("File size: {file_size} bytes (verified)");

    // ---- Read ----
    let start = Instant::now();
    let mut read_buf = vec![0u8; size];
    let read = handle
        .read_host(&mut read_buf, 0)
        .expect("read_host failed");
    let read_elapsed = start.elapsed();

    assert_eq!(read, size, "short read");
    println!(
        "Read:  {read} bytes in {:.1} us",
        read_elapsed.as_micros()
    );

    // ---- Verify data ----
    for i in 0..nelem {
        let offset = i * 4;
        let expected = (i as i32).to_ne_bytes();
        let actual = &read_buf[offset..offset + 4];
        assert_eq!(
            actual, &expected,
            "data mismatch at element {i}"
        );
    }
    println!("\nData verification: PASSED ({nelem} elements correct)");

    println!("\nDone.");
}
