//! Device buffer registration for GDS.
//!
//! Registering a device buffer with the cuFile driver improves performance
//! for repeated I/O operations to the same buffer. This module provides
//! safe wrappers around cudarc's `buf_register` / `buf_deregister`.

use std::sync::Arc;

use crate::error::{Error, ErrorKind, Result};

/// Register a device buffer for repeated GDS I/O.
///
/// Pre-registers the buffer with the cuFile driver, which can improve performance
/// by avoiding per-I/O registration overhead.
///
/// No-op if the driver is `None` (compatibility mode).
///
/// # Arguments
///
/// * `driver` - The cuFile driver instance (from a `FileHandle`).
/// * `buf` - The device buffer to register.
pub fn buffer_register(
    driver: &Arc<cudarc::cufile::Cufile>,
    buf: &cudarc::driver::CudaSlice<u8>,
) -> Result<()> {
    driver.buf_register(buf).map_err(|e| {
        Error::new(
            ErrorKind::CuFileError,
            format!("buffer_register failed: {e}"),
        )
        .with_operation("buffer_register")
    })
}

/// Deregister a previously registered device buffer.
///
/// Must be called before the buffer is freed.
///
/// No-op if the driver is `None` (compatibility mode).
///
/// # Arguments
///
/// * `driver` - The cuFile driver instance.
/// * `buf` - The device buffer to deregister.
pub fn buffer_deregister(
    driver: &Arc<cudarc::cufile::Cufile>,
    buf: &cudarc::driver::CudaSlice<u8>,
) -> Result<()> {
    driver.buf_deregister(buf).map_err(|e| {
        Error::new(
            ErrorKind::CuFileError,
            format!("buffer_deregister failed: {e}"),
        )
        .with_operation("buffer_deregister")
    })
}

#[cfg(test)]
mod tests {
    // Integration tests for buffer_register/buffer_deregister require
    // a CUDA device with cuFile support and are in integration/tests/.

    #[test]
    fn test_buffer_module_compiles() {
        // Smoke test to verify the module compiles.
    }
}
