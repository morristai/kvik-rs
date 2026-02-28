//! Shared utilities for kvik-rs benchmarks.
//!
//! Provides timing, statistics, formatting, and argument parsing used by all
//! benchmark binaries.

// Each benchmark binary includes this module via `#[path]` and uses a different
// subset of these utilities, so some items appear unused per-binary.
#![allow(dead_code)]

use std::time::Instant;

/// Lightweight timer wrapping `std::time::Instant`.
pub struct BenchTimer {
    start: Instant,
}

impl BenchTimer {
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    /// Returns elapsed time in seconds.
    pub fn elapsed_secs(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }
}

/// Descriptive statistics over a set of samples.
pub struct Stats {
    pub mean: f64,
    pub stdev: f64,
    pub stdev_pct: f64,
    pub n: usize,
}

impl Stats {
    /// Compute mean, stdev, and stdev-as-percentage from a slice of values.
    ///
    /// Returns zeros if `values` is empty.
    pub fn from_samples(values: &[f64]) -> Self {
        let n = values.len();
        if n == 0 {
            return Self {
                mean: 0.0,
                stdev: 0.0,
                stdev_pct: 0.0,
                n: 0,
            };
        }
        let mean = values.iter().sum::<f64>() / n as f64;
        let variance = if n > 1 {
            values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64
        } else {
            0.0
        };
        let stdev = variance.sqrt();
        let stdev_pct = if mean.abs() > 1e-15 {
            (stdev / mean) * 100.0
        } else {
            0.0
        };
        Self {
            mean,
            stdev,
            stdev_pct,
            n,
        }
    }
}

/// Format a byte count as a human-readable string (e.g. "16.0 MiB").
pub fn format_bytes(nbytes: usize) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

    let b = nbytes as f64;
    if b >= GIB {
        format!("{:.1} GiB", b / GIB)
    } else if b >= MIB {
        format!("{:.1} MiB", b / MIB)
    } else if b >= KIB {
        format!("{:.1} KiB", b / KIB)
    } else {
        format!("{nbytes} B")
    }
}

/// Compute throughput in MiB/s given byte count and elapsed seconds.
pub fn throughput_mibs(nbytes: usize, elapsed_secs: f64) -> f64 {
    if elapsed_secs <= 0.0 {
        return 0.0;
    }
    nbytes as f64 / (1024.0 * 1024.0) / elapsed_secs
}

/// Print a section header with underline.
pub fn print_header(title: &str) {
    println!("{title}");
    println!("{}", "=".repeat(title.len()));
}

/// Print a tabular row for throughput results.
pub fn print_row(label: &str, op: &str, throughput_mibs: f64, stdev_pct: f64, nruns: usize) {
    println!(
        "  {:<12} {:<8} {:>10.1} MiB/s  ± {:.1}%  ({nruns} runs)",
        label, op, throughput_mibs, stdev_pct
    );
}

/// Print the current kvik-rs configuration.
pub fn print_config() {
    let config = kvik_rs::Config::get();
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    println!(
        "Config: num_threads={}, task_size={}, bounce_buffer={}, page_size={}",
        config.num_threads,
        format_bytes(config.task_size),
        format_bytes(config.bounce_buffer_size),
        page_size,
    );
}

/// Generate deterministic test data of the given size.
///
/// Fills with a repeating byte pattern for easy verification.
pub fn gen_data(nbytes: usize) -> Vec<u8> {
    (0..nbytes).map(|i| (i % 251) as u8).collect()
}

/// Parsed benchmark arguments.
pub struct BenchArgs {
    /// Transfer sizes to benchmark (in bytes).
    pub sizes: Vec<usize>,
    /// Number of timed runs per configuration.
    pub nruns: usize,
    /// Number of warmup runs (excluded from stats).
    pub warmup: usize,
    /// Directory for temporary files (None = system default).
    pub dir: Option<String>,
}

impl Default for BenchArgs {
    fn default() -> Self {
        Self {
            sizes: vec![],
            nruns: 5,
            warmup: 1,
            dir: None,
        }
    }
}

/// Parse a size string like "4KiB", "16MiB", "1GiB", or a plain number (bytes).
pub fn parse_size(s: &str) -> Result<usize, String> {
    let s = s.trim();
    let (num_str, multiplier) = if let Some(n) = s.strip_suffix("GiB") {
        (n, 1024 * 1024 * 1024)
    } else if let Some(n) = s.strip_suffix("MiB") {
        (n, 1024 * 1024)
    } else if let Some(n) = s.strip_suffix("KiB") {
        (n, 1024)
    } else if let Some(n) = s.strip_suffix("GB") {
        (n, 1_000_000_000)
    } else if let Some(n) = s.strip_suffix("MB") {
        (n, 1_000_000)
    } else if let Some(n) = s.strip_suffix("KB") {
        (n, 1_000)
    } else {
        (s, 1)
    };
    let num: usize = num_str
        .trim()
        .parse()
        .map_err(|e| format!("invalid size {s:?}: {e}"))?;
    Ok(num * multiplier)
}

/// Parse command-line arguments for benchmarks.
///
/// Supports: `--size <size>[,<size>...]`, `--nruns <n>`, `--warmup <n>`, `--dir <path>`
pub fn parse_args(default_sizes: &[usize]) -> BenchArgs {
    let args: Vec<String> = std::env::args().collect();
    let mut bench_args = BenchArgs {
        sizes: default_sizes.to_vec(),
        ..BenchArgs::default()
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--size" | "--sizes" => {
                i += 1;
                if i < args.len() {
                    bench_args.sizes = args[i]
                        .split(',')
                        .map(|s| parse_size(s).unwrap_or_else(|e| panic!("{e}")))
                        .collect();
                }
            }
            "--nruns" => {
                i += 1;
                if i < args.len() {
                    bench_args.nruns = args[i].parse().expect("invalid --nruns value");
                }
            }
            "--warmup" => {
                i += 1;
                if i < args.len() {
                    bench_args.warmup = args[i].parse().expect("invalid --warmup value");
                }
            }
            "--dir" => {
                i += 1;
                if i < args.len() {
                    bench_args.dir = Some(args[i].clone());
                }
            }
            "--help" | "-h" => {
                println!("Usage: <benchmark> [OPTIONS]");
                println!();
                println!("Options:");
                println!("  --size <s>[,<s>...]   Transfer sizes (e.g. 4KiB,16MiB,128MiB)");
                println!("  --nruns <n>           Number of timed runs (default: 5)");
                println!("  --warmup <n>          Number of warmup runs (default: 1)");
                println!("  --dir <path>          Directory for temp files (default: system)");
                println!("  --help                Show this help");
                std::process::exit(0);
            }
            other => {
                eprintln!("warning: unknown argument: {other}");
            }
        }
        i += 1;
    }

    bench_args
}

/// Create a `tempfile::NamedTempFile` in the specified directory, or system default.
pub fn make_temp_file(dir: Option<&str>) -> tempfile::NamedTempFile {
    match dir {
        Some(d) => tempfile::NamedTempFile::new_in(d).expect("failed to create temp file"),
        None => tempfile::NamedTempFile::new().expect("failed to create temp file"),
    }
}
