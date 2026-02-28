# kvik-rs

GPUDirect Storage (GDS) accelerated file I/O for NVIDIA GPUs in Rust.

A Rust equivalent of NVIDIA's [kvikio](https://github.com/rapidsai/kvikio) C++ library, built on top of [cudarc](https://github.com/coreylowman/cudarc).

## Why kvik-rs?

cudarc provides low-level CUDA and cuFile bindings. kvik-rs builds on top of them to provide:

- **Automatic GDS/POSIX fallback** -- detects GDS availability at runtime and falls back to POSIX I/O transparently, so your code works on any system.
- **Opportunistic Direct I/O** -- splits unaligned transfers into aligned Direct I/O segments with buffered I/O for the remainder, maximizing throughput without requiring the caller to manage alignment.
- **Bounce buffer pooling** -- manages page-aligned, CUDA-pinned host buffers for device-to-file staging in POSIX mode, with automatic pool recycling.
- **Parallel I/O** -- splits large transfers into chunks across scoped threads with a single `pread_host`/`pwrite_host` call.
- **Batch I/O** -- submits multiple file operations to the GPU in a single kernel launch via cuFile's batch API.

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
kvik-rs = { version = "0.1", features = ["cuda-version-from-build-system"] }
```

### Device memory I/O (GDS)

```rust
use kvik_rs::{CompatMode, FileHandle};

// Open file with automatic GDS detection
let handle = FileHandle::open(path, "w+", 0o644, CompatMode::Auto)?;

// Write directly from GPU memory to file
let written = handle.write(&dev_buf, size, 0, 0)?;

// Read directly from file to GPU memory
let read = handle.read(&mut dev_buf, size, 0, 0)?;
```

### Host memory I/O (POSIX with opportunistic Direct I/O)

```rust
use kvik_rs::{CompatMode, FileHandle};

let handle = FileHandle::open(path, "w+", 0o644, CompatMode::On)?;

handle.write_host(&data, 0)?;
handle.read_host(&mut buf, 0)?;

// Parallel I/O -- splits across threads automatically
handle.pwrite_host(&data, 0, task_size)?;
handle.pread_host(&mut buf, 0, task_size)?;
```

## Features

- **Three operating modes**: GDS-enforced (`Off`), POSIX-enforced (`On`), auto-detect (`Auto`)
- **Device memory read/write** via cuFile with direct GPU-storage DMA
- **Host memory read/write** with opportunistic `O_DIRECT` and alignment handling
- **Parallel chunked I/O** via scoped threads (`pread_host`, `pwrite_host`)
- **Batch I/O** for submitting multiple operations to cuFile in one call
- **Buffer registration** (`buffer_register`/`buffer_deregister`) for repeated GDS I/O
- **Bounce buffer pool** with page-aligned allocation and automatic recycling
- **Runtime configuration** via environment variables (`KVIKIO_COMPAT_MODE`, `KVIKIO_NTHREADS`, etc.) compatible with C++ kvikio
- **Zero async runtime dependency** -- synchronous by design; callers wrap in `spawn_blocking` as needed
- **CUDA version feature flags** forwarded from cudarc (`cuda-11060` through `cuda-13010`)

## Limitations

- **Linux only.** GDS and `/proc`-based memory monitoring are Linux-specific.
- **No stream-ordered async I/O yet.** The `read_async`/`write_async` API (requiring CUDA >= 12.2) is not yet implemented.
- **No POSIX device memory fallback.** When GDS is unavailable, device memory writes return `Unsupported`; only host memory I/O falls back to POSIX.
- **No per-device thread pools.** C++ kvikio supports separate thread pools per storage device; kvik-rs uses a single pool.
- **`is_gds_available()` does not detect cuFile compat mode.** The cuFile driver may silently fall back to host-staged I/O on non-GDS filesystems even when `cuFileHandleRegister` succeeds.
- **No page cache utilities.** `mincore`/`posix_fadvise` wrappers from C++ kvikio are not ported.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Licensed under [Apache License 2.0](LICENSE).
