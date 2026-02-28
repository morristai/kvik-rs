//! Parallel I/O example.
//!
//! Demonstrates parallel read/write using `pwrite_host` / `pread_host` with
//! configurable threads and task sizes. Writes and reads multiple files
//! concurrently.
//!
//! No CUDA device required — runs in POSIX compatibility mode.
//!
//! Mirrors the multi-file pattern from C++ kvikio's `basic_no_cuda.cpp`.
//!
//! # Usage
//!
//! ```sh
//! cargo run --example parallel_io
//! ```

use std::time::Instant;

use kvik_rs::{CompatMode, Config, FileHandle};

fn main() {
    println!("kvik-rs: Parallel I/O Example");
    println!("==============================\n");

    // Configure for parallel I/O.
    let num_threads = 4;
    let task_size = 64 * 1024; // 64 KB chunks.
    Config::update(|c| {
        c.num_threads = num_threads;
        c.task_size = task_size;
    });

    let config = Config::get();
    println!("Configuration:");
    println!("  compat_mode:  {}", config.compat_mode);
    println!("  num_threads:  {}", config.num_threads);
    println!("  task_size:    {} bytes ({} KB)", config.task_size, config.task_size / 1024);
    println!();

    // Parameters.
    let nelem: usize = 1024;
    let size = nelem * std::mem::size_of::<i32>();
    let num_files = 2;
    println!("Test parameters:");
    println!("  elements per file: {nelem}");
    println!("  size per file:     {size} bytes");
    println!("  number of files:   {num_files}");
    println!();

    // Generate test data for each file.
    let data_sets: Vec<Vec<u8>> = (0..num_files)
        .map(|file_idx| {
            (0..nelem)
                .flat_map(|i| ((i + file_idx * 1000) as i32).to_ne_bytes())
                .collect()
        })
        .collect();

    // Create temporary files.
    let tmp_files: Vec<_> = (0..num_files)
        .map(|_| tempfile::NamedTempFile::new().expect("failed to create temp file"))
        .collect();

    // ---- Parallel Write ----
    println!("--- Parallel Write ---");
    let start = Instant::now();

    let mut total_written = 0;
    for (i, tmp) in tmp_files.iter().enumerate() {
        let handle = FileHandle::open(tmp.path(), "w", 0o644, CompatMode::On)
            .expect("failed to open for writing");
        let written = handle
            .pwrite_host(&data_sets[i], 0, task_size)
            .expect("pwrite_host failed");
        assert_eq!(written, size);
        total_written += written;
    }

    let write_elapsed = start.elapsed();
    let write_throughput = total_written as f64 / write_elapsed.as_secs_f64() / 1e6;
    println!(
        "  Wrote {} bytes across {} files in {:.1} us ({:.1} MB/s)",
        total_written,
        num_files,
        write_elapsed.as_micros(),
        write_throughput
    );

    // ---- Verify file sizes ----
    for (i, tmp) in tmp_files.iter().enumerate() {
        let handle = FileHandle::open(tmp.path(), "r", 0o644, CompatMode::On)
            .expect("failed to open for reading");
        let nbytes = handle.nbytes().expect("nbytes failed");
        assert_eq!(nbytes, size as u64, "file {i} size mismatch");
    }
    println!("  File sizes verified.");

    // ---- Parallel Read ----
    println!("\n--- Parallel Read ---");
    let start = Instant::now();

    let mut total_read = 0;
    let mut read_results: Vec<Vec<u8>> = Vec::with_capacity(num_files);
    for tmp in &tmp_files {
        let handle = FileHandle::open(tmp.path(), "r", 0o644, CompatMode::On)
            .expect("failed to open for reading");
        let mut buf = vec![0u8; size];
        let read = handle
            .pread_host(&mut buf, 0, task_size)
            .expect("pread_host failed");
        assert_eq!(read, size);
        total_read += read;
        read_results.push(buf);
    }

    let read_elapsed = start.elapsed();
    let read_throughput = total_read as f64 / read_elapsed.as_secs_f64() / 1e6;
    println!(
        "  Read {} bytes across {} files in {:.1} us ({:.1} MB/s)",
        total_read,
        num_files,
        read_elapsed.as_micros(),
        read_throughput
    );

    // ---- Verify data ----
    println!("\n--- Verification ---");
    for (i, (original, readback)) in data_sets.iter().zip(read_results.iter()).enumerate() {
        assert_eq!(
            original, readback,
            "data mismatch in file {i}"
        );
        println!("  File {i}: PASSED ({nelem} elements correct)");
    }

    println!("\nDone.");
}
