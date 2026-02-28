//! Parallel I/O Scaling Benchmark
//!
//! Measures how `pread_host`/`pwrite_host` throughput scales with thread count
//! and task size. Reports speedup relative to single-thread baseline.
//!
//! No CUDA GPU required — runs in POSIX compatibility mode.
//!
//! # Usage
//!
//! ```sh
//! cargo run --example bench_parallel_io --release
//! cargo run --example bench_parallel_io --release -- --size 128MiB --nruns 5
//! ```

#[path = "common.rs"]
mod common;

use common::{
    BenchTimer, Stats, format_bytes, gen_data, make_temp_file, parse_args, print_config,
    print_header, throughput_mibs,
};

use kvik_rs::{CompatMode, Config, FileHandle};

/// Default total transfer size.
const DEFAULT_TOTAL_SIZE: usize = 128 * 1024 * 1024; // 128 MiB

/// Thread counts to test.
const THREAD_COUNTS: &[usize] = &[1, 2, 4, 8];

/// Task sizes to test.
const TASK_SIZES: &[usize] = &[
    256 * 1024,      // 256 KiB
    1024 * 1024,     // 1 MiB
    4 * 1024 * 1024, // 4 MiB
];

fn main() {
    let args = parse_args(&[DEFAULT_TOTAL_SIZE]);
    let total_size = args.sizes[0];

    print_header("kvik-rs Parallel I/O Scaling");
    print_config();
    println!(
        "Total: {} | {} runs, {} warmup",
        format_bytes(total_size),
        args.nruns,
        args.warmup
    );
    println!();

    let data = gen_data(total_size);

    // Table header.
    println!(
        "  {:<8} {:<12} {:>14} {:>14} {:>10}",
        "Threads", "Task Size", "pread MiB/s", "pwrite MiB/s", "Speedup"
    );
    println!("  {}", "-".repeat(62));

    // Track single-thread baselines for speedup calculation (per task_size).
    let mut baseline_read: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
    let mut baseline_write: std::collections::HashMap<usize, f64> =
        std::collections::HashMap::new();

    for &nthreads in THREAD_COUNTS {
        for &task_size in TASK_SIZES {
            // Update global config for this combination.
            Config::update(|c| {
                c.num_threads = nthreads;
                c.task_size = task_size;
            });

            // --- pwrite benchmark ---
            let mut write_throughputs = Vec::with_capacity(args.nruns);

            for run in 0..(args.warmup + args.nruns) {
                let tmp = make_temp_file(args.dir.as_deref());
                let handle = FileHandle::open(tmp.path(), "w", 0o644, CompatMode::On)
                    .expect("failed to open file for writing");

                let timer = BenchTimer::start();
                let written = handle
                    .pwrite_host(&data, 0, task_size)
                    .expect("pwrite_host failed");
                let elapsed = timer.elapsed_secs();

                assert_eq!(written, total_size, "short write");

                if run >= args.warmup {
                    write_throughputs.push(throughput_mibs(total_size, elapsed));
                }
            }

            let write_stats = Stats::from_samples(&write_throughputs);

            // --- pread benchmark ---
            let mut read_throughputs = Vec::with_capacity(args.nruns);

            // Create a file with data for reads.
            let tmp = make_temp_file(args.dir.as_deref());
            {
                let handle = FileHandle::open(tmp.path(), "w", 0o644, CompatMode::On)
                    .expect("failed to open file for writing");
                handle
                    .pwrite_host(&data, 0, task_size)
                    .expect("pwrite_host (setup) failed");
            }

            for run in 0..(args.warmup + args.nruns) {
                let handle = FileHandle::open(tmp.path(), "r", 0o644, CompatMode::On)
                    .expect("failed to open file for reading");

                let mut buf = vec![0u8; total_size];
                let timer = BenchTimer::start();
                let read = handle
                    .pread_host(&mut buf, 0, task_size)
                    .expect("pread_host failed");
                let elapsed = timer.elapsed_secs();

                assert_eq!(read, total_size, "short read");

                if run >= args.warmup {
                    read_throughputs.push(throughput_mibs(total_size, elapsed));
                }
            }

            let read_stats = Stats::from_samples(&read_throughputs);

            // Track baselines and compute speedup.
            if nthreads == 1 {
                baseline_read.insert(task_size, read_stats.mean);
                baseline_write.insert(task_size, write_stats.mean);
            }

            let read_base = baseline_read
                .get(&task_size)
                .copied()
                .unwrap_or(read_stats.mean);
            let write_base = baseline_write
                .get(&task_size)
                .copied()
                .unwrap_or(write_stats.mean);
            let avg_speedup = if read_base > 0.0 && write_base > 0.0 {
                ((read_stats.mean / read_base) + (write_stats.mean / write_base)) / 2.0
            } else {
                1.0
            };

            println!(
                "  {:<8} {:<12} {:>10.1} ±{:.0}% {:>10.1} ±{:.0}%  {:>8.2}x",
                nthreads,
                format_bytes(task_size),
                read_stats.mean,
                read_stats.stdev_pct,
                write_stats.mean,
                write_stats.stdev_pct,
                avg_speedup,
            );
        }
    }

    // Restore default config.
    Config::set(Config::default());

    println!();
}
