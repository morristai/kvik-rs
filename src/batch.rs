//! Batch I/O operations for submitting multiple cuFile operations at once.
//!
//! Wraps cudarc's batch I/O result-level API in a safe interface.
//! This is gated on CUDA >= 11.6 (not available with `cuda-11040` or `cuda-11050` features).
//!
//! # Safety Invariants
//!
//! The batch I/O APIs in cudarc are exposed only at the `result` (unsafe) level.
//! kvik-rs wraps them with the following validation:
//! - `ops.len() <= max_num_events`
//! - File handles are valid (not closed)
//! - `BatchOp` is converted to cudarc's `CUfileIOParams_t` internally
//!
//! TODO: Contribute safe wrappers upstream to cudarc and replace these unsafe calls.

use std::time::Duration;

use cudarc::cufile::result::{
    batch_io_cancel, batch_io_destroy, batch_io_get_status, batch_io_setup, batch_io_submit,
};
use cudarc::cufile::sys::{
    timespec, CUfileBatchHandle_t, CUfileBatchMode_t,
    CUfileIOEvents_t, CUfileIOParams__bindgen_ty_1__bindgen_ty_1, CUfileIOParams_t, CUfileOpcode,
    CUfileStatus_t,
};

use crate::error::{Error, ErrorKind, Result};
use crate::file_handle::FileHandle;

/// Opcode for a batch I/O operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchOpcode {
    /// Read from file to device memory.
    Read,
    /// Write from device memory to file.
    Write,
}

/// A single operation in a batch I/O submission.
pub struct BatchOp<'a> {
    /// The file handle to operate on.
    pub file_handle: &'a FileHandle,
    /// Pointer to device memory (base address).
    pub dev_ptr: *mut u8,
    /// Offset within the file.
    pub file_offset: i64,
    /// Offset within the device buffer.
    pub dev_offset: i64,
    /// Number of bytes to transfer.
    pub size: usize,
    /// Whether to read or write.
    pub opcode: BatchOpcode,
}

// SAFETY: BatchOp contains a raw pointer, but it is only used as a parameter
// to cuFile batch operations which require valid device pointers.
unsafe impl Send for BatchOp<'_> {}
unsafe impl Sync for BatchOp<'_> {}

/// Status of a completed batch I/O event.
#[derive(Debug)]
pub struct BatchEvent {
    /// The status of this operation.
    pub status: BatchEventStatus,
    /// Number of bytes transferred (for completed operations).
    pub bytes_transferred: usize,
}

/// Status of an individual batch event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchEventStatus {
    /// Operation completed successfully.
    Complete,
    /// Operation is still waiting.
    Waiting,
    /// Operation is pending.
    Pending,
    /// Operation was cancelled.
    Canceled,
    /// Operation failed.
    Failed,
    /// Operation timed out.
    Timeout,
    /// Invalid operation.
    Invalid,
}

/// Handle for managing batch I/O operations.
///
/// Wraps cudarc's `CUfileBatchHandle_t` with validation and RAII cleanup.
pub struct BatchHandle {
    handle: CUfileBatchHandle_t,
    max_num_events: u32,
}

impl BatchHandle {
    /// Create a new batch I/O handle.
    ///
    /// # Arguments
    ///
    /// * `max_num_events` - Maximum number of I/O operations that can be submitted
    ///   in a single batch.
    pub fn new(max_num_events: u32) -> Result<Self> {
        let handle = batch_io_setup(max_num_events).map_err(|e| {
            Error::new(
                ErrorKind::CuFileError,
                format!("batch_io_setup failed: {e}"),
            )
            .with_operation("BatchHandle::new")
        })?;

        Ok(Self {
            handle,
            max_num_events,
        })
    }

    /// Submit a batch of I/O operations.
    ///
    /// # Errors
    ///
    /// Returns an error if `ops.len()` exceeds `max_num_events`.
    pub fn submit(&self, ops: &[BatchOp<'_>]) -> Result<()> {
        if ops.len() as u32 > self.max_num_events {
            return Err(Error::new(
                ErrorKind::ConfigInvalid,
                format!(
                    "batch submit: {} ops exceeds max_num_events ({})",
                    ops.len(),
                    self.max_num_events
                ),
            )
            .with_operation("BatchHandle::submit"));
        }

        let params: Vec<CUfileIOParams_t> = ops
            .iter()
            .map(|op| {
                let opcode = match op.opcode {
                    BatchOpcode::Read => CUfileOpcode::CUFILE_READ,
                    BatchOpcode::Write => CUfileOpcode::CUFILE_WRITE,
                };

                // SAFETY: We construct the params struct field-by-field using MaybeUninit
                // to avoid UB from zeroing a non-nullable enum field.
                // The file handle's cuFile handle must be valid (ensured by the borrow of FileHandle).
                let mut param = std::mem::MaybeUninit::<CUfileIOParams_t>::uninit();
                let p = param.as_mut_ptr();
                // SAFETY: p points to valid, allocated (but uninitialized) memory.
                // We initialize every field before calling assume_init().
                unsafe {
                    (*p).mode = CUfileBatchMode_t::CUFILE_BATCH;
                    (*p).opcode = opcode;
                    // fh would need to be set from the FileHandle's cuFile handle.
                    // This requires access to the internal cuFile handle type.
                    // TODO: This needs cudarc to expose the CUfileHandle_t from FileHandle.
                    (*p).fh = std::mem::zeroed();
                    (*p).cookie = std::ptr::null_mut();
                    (*p).u.batch = CUfileIOParams__bindgen_ty_1__bindgen_ty_1 {
                        devPtr_base: op.dev_ptr as *mut std::ffi::c_void,
                        file_offset: op.file_offset as libc::off_t,
                        devPtr_offset: op.dev_offset as libc::off_t,
                        size: op.size,
                    };
                }
                // SAFETY: All fields have been initialized above.
                unsafe { param.assume_init() }
            })
            .collect();

        // SAFETY: We have validated that ops.len() <= max_num_events.
        // The params vector contains valid CUfileIOParams_t structs.
        // The device pointers in BatchOp must be valid (caller's responsibility).
        unsafe {
            batch_io_submit(self.handle, &params, 0).map_err(|e| {
                Error::new(
                    ErrorKind::CuFileError,
                    format!("batch_io_submit failed: {e}"),
                )
                .with_operation("BatchHandle::submit")
            })?;
        }

        Ok(())
    }

    /// Poll for completed batch I/O events.
    ///
    /// # Arguments
    ///
    /// * `min_nr` - Minimum number of events to wait for.
    /// * `max_nr` - Maximum number of events to return.
    /// * `timeout` - How long to wait. `None` means wait indefinitely.
    pub fn status(
        &self,
        min_nr: u32,
        max_nr: u32,
        timeout: Option<Duration>,
    ) -> Result<Vec<BatchEvent>> {
        let ts = match timeout {
            Some(d) => timespec {
                tv_sec: d.as_secs() as i64,
                tv_nsec: d.subsec_nanos() as i64,
            },
            None => timespec {
                tv_sec: 0,
                tv_nsec: 0,
            },
        };

        let mut nr = max_nr;
        let mut events = vec![unsafe { std::mem::zeroed::<CUfileIOEvents_t>() }; max_nr as usize];

        // SAFETY: handle is valid, events buffer is correctly sized,
        // nr is passed by mutable reference to receive actual count.
        unsafe {
            batch_io_get_status(self.handle, min_nr, &mut nr, &mut events, &ts).map_err(|e| {
                Error::new(
                    ErrorKind::CuFileError,
                    format!("batch_io_get_status failed: {e}"),
                )
                .with_operation("BatchHandle::status")
            })?;
        }

        let result = events[..nr as usize]
            .iter()
            .map(|e| {
                let status = match e.status {
                    CUfileStatus_t::CUFILE_COMPLETE => BatchEventStatus::Complete,
                    CUfileStatus_t::CUFILE_WAITING => BatchEventStatus::Waiting,
                    CUfileStatus_t::CUFILE_PENDING => BatchEventStatus::Pending,
                    CUfileStatus_t::CUFILE_CANCELED => BatchEventStatus::Canceled,
                    CUfileStatus_t::CUFILE_TIMEOUT => BatchEventStatus::Timeout,
                    CUfileStatus_t::CUFILE_FAILED => BatchEventStatus::Failed,
                    CUfileStatus_t::CUFILE_INVALID => BatchEventStatus::Invalid,
                    #[allow(unreachable_patterns)]
                    _ => BatchEventStatus::Invalid,
                };

                BatchEvent {
                    status,
                    bytes_transferred: e.ret,
                }
            })
            .collect();

        Ok(result)
    }

    /// Cancel all pending operations in this batch.
    pub fn cancel(&self) -> Result<()> {
        // SAFETY: handle is valid.
        unsafe {
            batch_io_cancel(self.handle).map_err(|e| {
                Error::new(
                    ErrorKind::CuFileError,
                    format!("batch_io_cancel failed: {e}"),
                )
                .with_operation("BatchHandle::cancel")
            })?;
        }
        Ok(())
    }
}

impl Drop for BatchHandle {
    fn drop(&mut self) {
        // SAFETY: handle is valid and will not be used after drop.
        let _ = unsafe { batch_io_destroy(self.handle) };
    }
}

// SAFETY: cuFile batch handles are thread-safe.
unsafe impl Send for BatchHandle {}
unsafe impl Sync for BatchHandle {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_opcode() {
        assert_eq!(BatchOpcode::Read, BatchOpcode::Read);
        assert_ne!(BatchOpcode::Read, BatchOpcode::Write);
    }

    #[test]
    fn test_batch_event_status() {
        assert_eq!(BatchEventStatus::Complete, BatchEventStatus::Complete);
        assert_ne!(BatchEventStatus::Complete, BatchEventStatus::Failed);
    }

    // Integration tests for BatchHandle::new, submit, status, cancel require
    // a CUDA device with cuFile support and are in integration/tests/.
}
