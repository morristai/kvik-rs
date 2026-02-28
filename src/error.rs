//! Error types for kvik-rs.
//!
//! # Design
//!
//! - [`ErrorKind`] categorizes *what* went wrong.
//! - [`ErrorStatus`] indicates *what to do about it* (retry or not).
//! - [`Error`] combines both with rich context for debugging.
//!
//! Errors are constructed via a fluent builder pattern:
//!
//! ```
//! use kvik_rs::error::{Error, ErrorKind};
//!
//! let err = Error::new(ErrorKind::CuFileError, "cuFile driver initialization failed")
//!     .with_operation("FileHandle::open")
//!     .with_context("path", "/data/file.bin")
//!     .set_temporary();
//! ```

use std::backtrace::{Backtrace, BacktraceStatus};
use std::fmt;

/// A specialized `Result` type for kvik-rs operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Categorizes the type of error that occurred.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ErrorKind {
    /// An unexpected/internal error that should not happen under normal conditions.
    Unexpected,
    /// The requested file or resource was not found.
    NotFound,
    /// Permission denied when accessing a file or resource.
    PermissionDenied,
    /// Invalid configuration or arguments.
    ConfigInvalid,
    /// A POSIX/system call failed.
    SystemError,
    /// A cuFile API call failed.
    CuFileError,
    /// A CUDA operation failed.
    CudaError,
    /// The requested feature is not supported in the current environment.
    Unsupported,
    /// An I/O operation was interrupted.
    Interrupted,
}

impl ErrorKind {
    /// Returns a static string label for this error kind.
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorKind::Unexpected => "Unexpected",
            ErrorKind::NotFound => "NotFound",
            ErrorKind::PermissionDenied => "PermissionDenied",
            ErrorKind::ConfigInvalid => "ConfigInvalid",
            ErrorKind::SystemError => "SystemError",
            ErrorKind::CuFileError => "CuFileError",
            ErrorKind::CudaError => "CudaError",
            ErrorKind::Unsupported => "Unsupported",
            ErrorKind::Interrupted => "Interrupted",
        }
    }

    /// Whether to capture a backtrace for this error kind.
    ///
    /// Capturing a backtrace is expensive at runtime. We only capture it for
    /// kinds where the call site is likely to be surprising (e.g., `Unexpected`).
    ///
    /// See <https://github.com/apache/opendal/discussions/5569>
    fn enable_backtrace(&self) -> bool {
        matches!(self, ErrorKind::Unexpected)
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Indicates whether the caller should retry the operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorStatus {
    /// The error is permanent and should not be retried.
    Permanent,
    /// The error is transient and may succeed on retry.
    Temporary,
    /// The error was transient but persisted after retry attempts.
    /// Should not be retried again.
    Persistent,
}

impl fmt::Display for ErrorStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorStatus::Permanent => f.write_str("permanent"),
            ErrorStatus::Temporary => f.write_str("temporary"),
            ErrorStatus::Persistent => f.write_str("persistent"),
        }
    }
}

/// The main error type for kvik-rs.
pub struct Error {
    kind: ErrorKind,
    message: String,
    status: ErrorStatus,
    operation: &'static str,
    context: Vec<(&'static str, String)>,
    source: Option<Box<dyn std::error::Error + Send + Sync>>,
    backtrace: Option<Box<Backtrace>>,
}

impl Error {
    /// Create a new error with the given kind and message.
    ///
    /// Defaults to `ErrorStatus::Permanent` and empty operation/context.
    ///
    /// For [`ErrorKind::Unexpected`], a backtrace is captured if `RUST_BACKTRACE`
    /// is enabled. For other kinds, backtrace capture is skipped to avoid the
    /// runtime cost.
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            status: ErrorStatus::Permanent,
            operation: "",
            context: Vec::new(),
            source: None,
            // `Backtrace::capture()` checks if backtrace is enabled internally.
            // It's zero cost when `RUST_BACKTRACE` is not set.
            backtrace: kind
                .enable_backtrace()
                .then(Backtrace::capture)
                .filter(|bt| bt.status() == BacktraceStatus::Captured)
                .map(Box::new),
        }
    }

    /// Returns the error kind.
    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    /// Returns the error status.
    pub fn status(&self) -> ErrorStatus {
        self.status
    }

    /// Returns the operation that caused this error.
    pub fn operation(&self) -> &'static str {
        self.operation
    }

    /// Returns the human-readable error message.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Returns the error context as key-value pairs.
    pub fn context_iter(&self) -> impl Iterator<Item = (&'static str, &str)> {
        self.context.iter().map(|(k, v)| (*k, v.as_str()))
    }

    /// Returns `true` if this error is permanent.
    pub fn is_permanent(&self) -> bool {
        self.status == ErrorStatus::Permanent
    }

    /// Returns `true` if this error is temporary (retryable).
    pub fn is_temporary(&self) -> bool {
        self.status == ErrorStatus::Temporary
    }

    /// Returns `true` if this error is persistent (was temporary, no longer retryable).
    pub fn is_persistent(&self) -> bool {
        self.status == ErrorStatus::Persistent
    }

    /// Set the error status to permanent. Consumes and returns self.
    pub fn set_permanent(mut self) -> Self {
        self.status = ErrorStatus::Permanent;
        self
    }

    /// Set the error status to temporary. Consumes and returns self.
    pub fn set_temporary(mut self) -> Self {
        self.status = ErrorStatus::Temporary;
        self
    }

    /// Set the error status to persistent. Consumes and returns self.
    pub fn set_persistent(mut self) -> Self {
        self.status = ErrorStatus::Persistent;
        self
    }

    /// Set the operation that caused this error.
    ///
    /// If the error already has an operation, the previous one is pushed
    /// into context as `("called", previous_operation)`.
    pub fn with_operation(mut self, operation: &'static str) -> Self {
        if !self.operation.is_empty() {
            self.context.push(("called", self.operation.to_string()));
        }
        self.operation = operation;
        self
    }

    /// Add a key-value context pair.
    pub fn with_context(mut self, key: &'static str, value: impl Into<String>) -> Self {
        self.context.push((key, value.into()));
        self
    }

    /// Set the underlying source error.
    pub fn set_source(mut self, source: impl std::error::Error + Send + Sync + 'static) -> Self {
        self.source = Some(Box::new(source));
        self
    }

    /// Returns the captured backtrace, if any.
    ///
    /// Backtraces are only captured for [`ErrorKind::Unexpected`] and only when
    /// `RUST_BACKTRACE=1` (or `full`) is set.
    ///
    /// To print the backtrace, use `Debug` formatting: `format!("{err:?}")`.
    pub fn backtrace(&self) -> Option<&Backtrace> {
        self.backtrace.as_deref()
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            // Struct-style debug output
            let mut d = f.debug_struct("Error");
            d.field("kind", &self.kind);
            d.field("message", &self.message);
            d.field("status", &self.status);
            d.field("operation", &self.operation);
            d.field("context", &self.context);
            d.field("source", &self.source.as_ref().map(|s| s.to_string()));
            d.finish()
        } else {
            // Multi-line debug output
            writeln!(
                f,
                "{} ({}) at {} => {}",
                self.kind, self.status, self.operation, self.message
            )?;
            if !self.context.is_empty() {
                writeln!(f)?;
                writeln!(f, "Context:")?;
                for (k, v) in &self.context {
                    writeln!(f, "   {k}: {v}")?;
                }
            }
            if let Some(source) = &self.source {
                writeln!(f)?;
                writeln!(f, "Source:")?;
                writeln!(f, "   {source}")?;
            }
            if let Some(backtrace) = &self.backtrace {
                writeln!(f)?;
                writeln!(f, "Backtrace:")?;
                writeln!(f, "{backtrace}")?;
            }
            Ok(())
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({}) at {}", self.kind, self.status, self.operation)?;
        if !self.context.is_empty() {
            write!(f, ", context: {{ ")?;
            for (i, (k, v)) in self.context.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{k}: {v}")?;
            }
            write!(f, " }}")?;
        }
        write!(f, " => {}", self.message)?;
        if let Some(source) = &self.source {
            write!(f, ", source: {source}")?;
        }
        Ok(())
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.as_ref().map(|e| e.as_ref() as _)
    }
}

impl From<Error> for std::io::Error {
    fn from(err: Error) -> std::io::Error {
        let kind = match err.kind() {
            ErrorKind::NotFound => std::io::ErrorKind::NotFound,
            ErrorKind::PermissionDenied => std::io::ErrorKind::PermissionDenied,
            ErrorKind::Interrupted => std::io::ErrorKind::Interrupted,
            ErrorKind::Unsupported => std::io::ErrorKind::Unsupported,
            _ => std::io::ErrorKind::Other,
        };
        std::io::Error::new(kind, err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error as StdError;

    #[test]
    fn test_error_kind_as_str() {
        assert_eq!(ErrorKind::Unexpected.as_str(), "Unexpected");
        assert_eq!(ErrorKind::NotFound.as_str(), "NotFound");
        assert_eq!(ErrorKind::PermissionDenied.as_str(), "PermissionDenied");
        assert_eq!(ErrorKind::ConfigInvalid.as_str(), "ConfigInvalid");
        assert_eq!(ErrorKind::SystemError.as_str(), "SystemError");
        assert_eq!(ErrorKind::CuFileError.as_str(), "CuFileError");
        assert_eq!(ErrorKind::CudaError.as_str(), "CudaError");
        assert_eq!(ErrorKind::Unsupported.as_str(), "Unsupported");
        assert_eq!(ErrorKind::Interrupted.as_str(), "Interrupted");
    }

    #[test]
    fn test_error_kind_display() {
        assert_eq!(format!("{}", ErrorKind::NotFound), "NotFound");
        assert_eq!(format!("{}", ErrorKind::CuFileError), "CuFileError");
    }

    #[test]
    fn test_error_status_display() {
        assert_eq!(format!("{}", ErrorStatus::Permanent), "permanent");
        assert_eq!(format!("{}", ErrorStatus::Temporary), "temporary");
        assert_eq!(format!("{}", ErrorStatus::Persistent), "persistent");
    }

    #[test]
    fn test_error_new_defaults() {
        let err = Error::new(ErrorKind::NotFound, "file missing");
        assert_eq!(err.kind(), ErrorKind::NotFound);
        assert_eq!(err.status(), ErrorStatus::Permanent);
        assert_eq!(err.operation(), "");
        assert_eq!(err.message(), "file missing");
        assert!(err.is_permanent());
        assert!(!err.is_temporary());
        assert!(!err.is_persistent());
    }

    #[test]
    fn test_error_builder_chain() {
        let err = Error::new(ErrorKind::CuFileError, "driver init failed")
            .with_operation("FileHandle::open")
            .with_context("path", "/data/file.bin")
            .set_temporary();

        assert_eq!(err.kind(), ErrorKind::CuFileError);
        assert_eq!(err.status(), ErrorStatus::Temporary);
        assert_eq!(err.operation(), "FileHandle::open");
        assert!(err.is_temporary());

        let ctx: Vec<_> = err.context_iter().collect();
        assert_eq!(ctx.len(), 1);
        assert_eq!(ctx[0], ("path", "/data/file.bin"));
    }

    #[test]
    fn test_error_operation_chaining() {
        let err = Error::new(ErrorKind::SystemError, "pread failed")
            .with_operation("posix_host_io")
            .with_operation("FileHandle::read_host");

        assert_eq!(err.operation(), "FileHandle::read_host");
        let ctx: Vec<_> = err.context_iter().collect();
        assert_eq!(ctx.len(), 1);
        assert_eq!(ctx[0], ("called", "posix_host_io"));
    }

    #[test]
    fn test_error_with_source() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "no such file");
        let err = Error::new(ErrorKind::SystemError, "failed to open file")
            .with_operation("FileHandle::open")
            .set_source(io_err);

        assert!(StdError::source(&err).is_some());
        let source = StdError::source(&err).unwrap();
        assert!(source.to_string().contains("no such file"));
    }

    #[test]
    fn test_error_display_format() {
        let err = Error::new(ErrorKind::NotFound, "file missing")
            .with_operation("FileHandle::open")
            .with_context("path", "/data/file.bin");

        let display = format!("{err}");
        assert!(display.contains("NotFound"));
        assert!(display.contains("permanent"));
        assert!(display.contains("FileHandle::open"));
        assert!(display.contains("path: /data/file.bin"));
        assert!(display.contains("file missing"));
    }

    #[test]
    fn test_error_debug_format() {
        let err = Error::new(ErrorKind::CuFileError, "driver failed")
            .with_operation("Cufile::new")
            .with_context("reason", "library not loaded");

        let debug = format!("{err:?}");
        assert!(debug.contains("CuFileError"));
        assert!(debug.contains("permanent"));
        assert!(debug.contains("Cufile::new"));
        assert!(debug.contains("driver failed"));
        assert!(debug.contains("reason: library not loaded"));
    }

    #[test]
    fn test_error_alternate_debug_format() {
        let err = Error::new(ErrorKind::Unexpected, "internal error");
        let debug = format!("{err:#?}");
        assert!(debug.contains("Unexpected"));
        assert!(debug.contains("internal error"));
    }

    #[test]
    fn test_error_status_transitions() {
        let err = Error::new(ErrorKind::SystemError, "transient failure");
        assert!(err.is_permanent());

        let err = err.set_temporary();
        assert!(err.is_temporary());

        let err = err.set_persistent();
        assert!(err.is_persistent());

        let err = err.set_permanent();
        assert!(err.is_permanent());
    }

    #[test]
    fn test_error_into_io_error() {
        let err = Error::new(ErrorKind::NotFound, "gone");
        let io_err: std::io::Error = err.into();
        assert_eq!(io_err.kind(), std::io::ErrorKind::NotFound);

        let err = Error::new(ErrorKind::PermissionDenied, "no access");
        let io_err: std::io::Error = err.into();
        assert_eq!(io_err.kind(), std::io::ErrorKind::PermissionDenied);

        let err = Error::new(ErrorKind::CuFileError, "cuda problem");
        let io_err: std::io::Error = err.into();
        assert_eq!(io_err.kind(), std::io::ErrorKind::Other);
    }

    #[test]
    fn test_error_display_with_source() {
        let io_err = std::io::Error::new(std::io::ErrorKind::BrokenPipe, "pipe broke");
        let err = Error::new(ErrorKind::SystemError, "write failed")
            .with_operation("FileHandle::write_host")
            .set_source(io_err);

        let display = format!("{err}");
        assert!(display.contains("source: pipe broke"));
    }

    #[test]
    fn test_error_multiple_context() {
        let err = Error::new(ErrorKind::SystemError, "pread failed")
            .with_context("path", "/data/file.bin")
            .with_context("offset", "4096")
            .with_context("size", "8192");

        let ctx: Vec<_> = err.context_iter().collect();
        assert_eq!(ctx.len(), 3);
        assert_eq!(ctx[0], ("path", "/data/file.bin"));
        assert_eq!(ctx[1], ("offset", "4096"));
        assert_eq!(ctx[2], ("size", "8192"));
    }

    #[test]
    fn test_error_is_std_error() {
        fn assert_error<E: std::error::Error>() {}
        assert_error::<Error>();
    }

    #[test]
    fn test_error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        // Error should be Send (source is Send + Sync) but not necessarily Sync
        // due to Box<dyn Error + Send + Sync> which is Send but potentially not Sync.
        assert_send_sync::<Error>();
    }

    #[test]
    fn test_backtrace_not_captured_for_non_unexpected() {
        // Non-Unexpected kinds should never capture a backtrace.
        let err = Error::new(ErrorKind::NotFound, "file missing");
        assert!(err.backtrace().is_none());

        let err = Error::new(ErrorKind::SystemError, "pread failed");
        assert!(err.backtrace().is_none());

        let err = Error::new(ErrorKind::CuFileError, "driver init failed");
        assert!(err.backtrace().is_none());
    }

    #[test]
    fn test_backtrace_captured_for_unexpected_when_enabled() {
        // When RUST_BACKTRACE is set, Unexpected errors should capture a backtrace.
        // SAFETY: test runs single-threaded; no other thread reads this env var concurrently.
        unsafe { std::env::set_var("RUST_BACKTRACE", "1") };
        let err = Error::new(ErrorKind::Unexpected, "internal error");

        // The backtrace should be captured.
        assert!(err.backtrace().is_some());

        // Debug output should contain the backtrace.
        let debug = format!("{err:?}");
        assert!(debug.contains("Backtrace:"));
    }
}
