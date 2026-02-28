//! Device I/O Benchmark: GDS throughput
//!
//! Measures GDS `read()`/`write()` throughput for GPU device memory at various
//! transfer sizes. Exits gracefully (code 0) if no CUDA GPU or GDS is unavailable.
//!
//! # Requirements
//!
//! - NVIDIA GPU with CUDA support
//! - cuFile/GDS drivers installed
//!
//! # Usage
//!
//! ```sh
//! cargo run --example bench_device_io --release
//! cargo run --example bench_device_io --release -- --size 1MiB,128MiB --nruns 10
//! ```

#[path = "common.rs"]
mod common;

use common::{
    BenchTimer, Stats, format_bytes, gen_data, make_temp_file, parse_args, print_config,
    print_header, print_row, throughput_mibs,
};

use kvik_rs::{CompatMode, FileHandle};

const DEFAULT_SIZES: &[usize] = &[
    1024 * 1024,       // 1 MiB
    16 * 1024 * 1024,  // 16 MiB
    128 * 1024 * 1024, // 128 MiB
    512 * 1024 * 1024, // 512 MiB
];

fn main() {
    let args = parse_args(DEFAULT_SIZES);

    print_header("kvik-rs Device I/O Benchmark (GDS)");
    print_config();
    println!();

    // --- CUDA initialization ---
    let ctx = match cudarc::driver::CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            println!("SKIP: no CUDA GPU ({e})");
            return;
        }
    };
    let stream = ctx.default_stream();
    println!("CUDA device 0: initialized");

    // --- Check GDS availability with a probe file ---
    {
        let probe_tmp = make_temp_file(args.dir.as_deref());
        let probe_handle = FileHandle::open(probe_tmp.path(), "w+", 0o644, CompatMode::Auto)
            .expect("failed to open probe file");
        if !probe_handle.is_gds_available() {
            println!("SKIP: GDS not available on this filesystem");
            return;
        }
        println!("GDS: available");
    }

    println!();

    for &size in &args.sizes {
        println!(
            "Size: {} | {} runs, {} warmup",
            format_bytes(size),
            args.nruns,
            args.warmup
        );
        println!(
            "  {:<12} {:<8} {:>14}  {:>8}",
            "Mode", "Op", "Throughput", "Stdev"
        );

        let data = gen_data(size);

        // Allocate device memory and fill with test data.
        let dev_src = stream
            .clone_htod(&data)
            .expect("failed to copy test data to device");

        // --- Write benchmark (GPU → file) ---
        let mut write_throughputs = Vec::with_capacity(args.nruns);

        for run in 0..(args.warmup + args.nruns) {
            let tmp = make_temp_file(args.dir.as_deref());
            let handle = FileHandle::open(tmp.path(), "w+", 0o644, CompatMode::Auto)
                .expect("failed to open file");

            let timer = BenchTimer::start();
            let written = handle
                .write(&dev_src, size, 0, 0)
                .expect("GDS write failed");
            let elapsed = timer.elapsed_secs();

            assert_eq!(written, size, "short write");

            if run >= args.warmup {
                write_throughputs.push(throughput_mibs(size, elapsed));
            }
        }

        let write_stats = Stats::from_samples(&write_throughputs);
        print_row(
            "GDS",
            "write",
            write_stats.mean,
            write_stats.stdev_pct,
            write_stats.n,
        );

        // --- Read benchmark (file → GPU) ---
        let mut read_throughputs = Vec::with_capacity(args.nruns);

        // Create a file with known data for reads.
        let tmp = make_temp_file(args.dir.as_deref());
        {
            let handle = FileHandle::open(tmp.path(), "w+", 0o644, CompatMode::Auto)
                .expect("failed to open file");
            handle
                .write(&dev_src, size, 0, 0)
                .expect("GDS write (setup) failed");
        }

        let mut dev_dst = stream
            .alloc_zeros::<u8>(size)
            .expect("failed to alloc device buffer");

        for run in 0..(args.warmup + args.nruns) {
            let handle = FileHandle::open(tmp.path(), "r", 0o644, CompatMode::Auto)
                .expect("failed to open file for reading");

            let timer = BenchTimer::start();
            let read = handle
                .read(&mut dev_dst, size, 0, 0)
                .expect("GDS read failed");
            let elapsed = timer.elapsed_secs();

            assert_eq!(read, size, "short read");

            // Verify data on first timed run.
            if run == args.warmup {
                let result = stream.clone_dtoh(&dev_dst).expect("clone_dtoh failed");
                assert_eq!(result, data, "data verification failed");
            }

            if run >= args.warmup {
                read_throughputs.push(throughput_mibs(size, elapsed));
            }
        }

        let read_stats = Stats::from_samples(&read_throughputs);
        print_row(
            "GDS",
            "read",
            read_stats.mean,
            read_stats.stdev_pct,
            read_stats.n,
        );

        println!();
    }
}
