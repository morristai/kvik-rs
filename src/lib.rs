//! # kvik-rs
//!
//! GPUDirect Storage (GDS) accelerated file I/O for NVIDIA GPUs.
//!
//! kvik-rs provides Rust developers with a safe, ergonomic API for GPU-accelerated
//! file I/O that mirrors the capabilities of the C++ kvikio library. It is designed
//! both as a standalone library and as an integration target for Apache OpenDAL.
//!
//! ## Architecture
//!
//! kvik-rs builds on top of [`cudarc`]'s cuFile bindings rather than maintaining
//! independent FFI code. It supports three modes of operation:
//!
//! - **GDS mode** (`CompatMode::Off`): Enforces GPUDirect Storage; errors if unavailable.
//! - **POSIX mode** (`CompatMode::On`): Uses POSIX I/O with opportunistic Direct I/O.
//! - **Auto mode** (`CompatMode::Auto`): Tries GDS, falls back to POSIX (default).
//!
//! ## Feature Flags
//!
//! Feature flags mirror cudarc's CUDA version gates:
//!
//! - `cuda-12020`: Enables stream-ordered async I/O (requires CUDA >= 12.2)
//!
//! Batch I/O is always available (requires CUDA >= 11.6 at runtime, but the API
//! is unconditionally compiled since kvik-rs does not expose cuda-11040/cuda-11050).

pub mod align;
pub mod batch;
pub mod bounce_buffer;
pub mod buffer;
pub mod compat_mode;
pub mod config;
pub mod error;
pub mod file_handle;
pub mod posix_io;

pub use compat_mode::CompatMode;
pub use config::Config;
pub use error::{Error, ErrorKind, ErrorStatus, Result};
pub use file_handle::FileHandle;
