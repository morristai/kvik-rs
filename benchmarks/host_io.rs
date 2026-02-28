//! Host I/O Benchmark: GDS-init vs POSIX-only
//!
//! Measures whether having the cuFile driver initialized (Auto mode) affects
//! host-memory I/O performance compared to pure POSIX mode (On). Both code paths
//! use the same POSIX I/O engine for host memory — the difference is cuFile driver
//! initialization overhead.
//!
//! No CUDA GPU required — runs in POSIX compatibility mode.
//!
//! # Usage
//!
//! ```sh
//! cargo run --example bench_host_io --release
//! cargo run --example bench_host_io --release -- --size 4KiB,1MiB,16MiB --nruns 10
//! ```

#[path = "common.rs"]
mod common;

use common::{
    BenchTimer, Stats, format_bytes, gen_data, make_temp_file, parse_args, print_config,
    print_header, print_row, throughput_mibs,
};

use kvik_rs::{CompatMode, FileHandle};

const DEFAULT_SIZES: &[usize] = &[
    4 * 1024,          // 4 KiB
    64 * 1024,         // 64 KiB
    1024 * 1024,       // 1 MiB
    16 * 1024 * 1024,  // 16 MiB
    128 * 1024 * 1024, // 128 MiB
];

fn main() {
    let args = parse_args(DEFAULT_SIZES);

    print_header("kvik-rs Host I/O Benchmark");
    print_config();
    println!();

    let modes: &[(&str, CompatMode)] = &[("Auto", CompatMode::Auto), ("POSIX", CompatMode::On)];

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

        for &(mode_name, compat_mode) in modes {
            // --- Write benchmark ---
            let mut write_throughputs = Vec::with_capacity(args.nruns);

            for run in 0..(args.warmup + args.nruns) {
                let tmp = make_temp_file(args.dir.as_deref());
                let handle = FileHandle::open(tmp.path(), "w", 0o644, compat_mode)
                    .expect("failed to open file for writing");

                let timer = BenchTimer::start();
                let written = handle.write_host(&data, 0).expect("write_host failed");
                let elapsed = timer.elapsed_secs();
                drop(handle);

                assert_eq!(written, size, "short write");

                if run >= args.warmup {
                    write_throughputs.push(throughput_mibs(size, elapsed));
                }
            }

            let write_stats = Stats::from_samples(&write_throughputs);
            print_row(
                mode_name,
                "write",
                write_stats.mean,
                write_stats.stdev_pct,
                write_stats.n,
            );

            // --- Read benchmark ---
            let mut read_throughputs = Vec::with_capacity(args.nruns);

            // Create one file with data for all read runs.
            let tmp = make_temp_file(args.dir.as_deref());
            {
                let handle = FileHandle::open(tmp.path(), "w", 0o644, CompatMode::On)
                    .expect("failed to open file for writing");
                handle.write_host(&data, 0).expect("write_host failed");
            }

            for run in 0..(args.warmup + args.nruns) {
                let handle = FileHandle::open(tmp.path(), "r", 0o644, compat_mode)
                    .expect("failed to open file for reading");

                let mut buf = vec![0u8; size];
                let timer = BenchTimer::start();
                let read = handle.read_host(&mut buf, 0).expect("read_host failed");
                let elapsed = timer.elapsed_secs();

                assert_eq!(read, size, "short read");

                if run >= args.warmup {
                    read_throughputs.push(throughput_mibs(size, elapsed));
                }
            }

            let read_stats = Stats::from_samples(&read_throughputs);
            print_row(
                mode_name,
                "read",
                read_stats.mean,
                read_stats.stdev_pct,
                read_stats.n,
            );
        }

        println!();
    }
}
