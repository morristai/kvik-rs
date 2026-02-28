# kvik-rs justfile
# https://github.com/casey/just

# List available recipes
default:
    @just --list

# ---------- Build ----------

# Check compilation without producing binaries
check:
    cargo check

# Build in debug mode
build:
    cargo build

# Build in release mode
build-release:
    cargo build --release

# Build with a specific CUDA feature flag (e.g., just build-cuda cuda-12060)
build-cuda feature:
    cargo build --no-default-features --features {{ feature }}

# ---------- Test ----------

# Run all tests
test:
    cargo test

# Run all tests with output shown
test-verbose:
    cargo test -- --nocapture

# Run unit tests only (lib target)
test-unit:
    cargo test --lib

# Run integration tests only (tests/ directory)
test-integration:
    cargo test --tests

# Run a specific test by name
test-one name:
    cargo test {{ name }} -- --exact

# Run tests matching a pattern
test-filter pattern:
    cargo test {{ pattern }}

# Run tests single-threaded (avoids env var races in config_env tests)
test-serial:
    cargo test -- --test-threads=1

# ---------- Lint ----------

# Run clippy with all targets
clippy:
    cargo clippy --all-targets -- -D warnings

# Run rustfmt check (no changes)
fmt-check:
    cargo fmt -- --check

# Run rustfmt
fmt:
    cargo fmt

# Run all lint checks (clippy + fmt)
lint: clippy fmt-check

# ---------- Doc ----------

# Build documentation
doc:
    cargo doc --no-deps

# Build and open documentation in browser
doc-open:
    cargo doc --no-deps --open

# ---------- Examples ----------

# Run the GDS device I/O example
example-gds:
    cargo run --example basic_gds_io

# Run the host I/O example
example-host:
    cargo run --example basic_host_io

# Run the parallel I/O example
example-parallel:
    cargo run --example parallel_io

# Run a specific example by name
example name:
    cargo run --example {{ name }}

# Run GPU VRAM stress tests (requires CUDA GPU, runs single-threaded)
test-vram-stress:
    cargo test --test gpu_vram_stress -- --ignored --test-threads=1 --nocapture

# Run GDS memory transfer validation test (requires CUDA + GDS)
test-gds-memory:
    cargo test --test gds_memory_transfer -- --ignored --nocapture

# ---------- Benchmarks ----------

# Run all benchmarks
bench-all: bench-host bench-device bench-parallel

# Host I/O throughput (no GPU required)
bench-host *args:
    cargo run --example bench_host_io --release -- {{args}}

# Device I/O throughput (requires CUDA + GDS)
bench-device *args:
    cargo run --example bench_device_io --release -- {{args}}

# Parallel I/O scaling (no GPU required)
bench-parallel *args:
    cargo run --example bench_parallel_io --release -- {{args}}

# ---------- Clean ----------

# Remove build artifacts
clean:
    cargo clean

# ---------- CI-style ----------

# Run the full CI check suite (fmt, clippy, test, doc)
ci: fmt-check clippy test doc
