//! Compatibility mode detection and configuration.
//!
//! Determines whether kvik-rs uses GPUDirect Storage (GDS) or falls back to POSIX I/O.
//!
//! # Detection Logic
//!
//! In `Auto` mode, kvik-rs probes for GDS availability by checking:
//! 1. Can the cuFile driver be initialized?
//! 2. Is `/run/udev` readable? (Required for GDS device detection; absent in many Docker containers.)
//! 3. Is this WSL? (GDS is not supported on WSL.)
//! 4. For a specific file: is the filesystem GDS-compatible?

use std::path::Path;

use crate::error::{Error, ErrorKind};

/// Controls whether kvik-rs uses GPUDirect Storage or POSIX I/O.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum CompatMode {
    /// Enforce GDS. Returns an error if GDS is unavailable.
    Off,
    /// Enforce POSIX I/O. Never attempts GDS.
    On,
    /// Try GDS first, fall back to POSIX if unavailable (default).
    #[default]
    Auto,
}

impl CompatMode {
    /// Parse a compatibility mode from a string.
    ///
    /// Accepts (case-insensitive):
    /// - `"on"`, `"true"`, `"yes"`, `"1"` → [`CompatMode::On`]
    /// - `"off"`, `"false"`, `"no"`, `"0"` → [`CompatMode::Off`]
    /// - `"auto"` → [`CompatMode::Auto`]
    pub fn parse(s: &str) -> crate::error::Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "on" | "true" | "yes" | "1" => Ok(CompatMode::On),
            "off" | "false" | "no" | "0" => Ok(CompatMode::Off),
            "auto" => Ok(CompatMode::Auto),
            _ => Err(Error::new(
                ErrorKind::ConfigInvalid,
                format!("unknown compatibility mode: {s:?}"),
            )
            .with_operation("CompatMode::parse")),
        }
    }

    /// Returns `true` if GDS should be used (i.e., mode is not `On`).
    ///
    /// In `Auto` mode, this returns `true` but the caller must still verify
    /// that GDS is actually available.
    pub fn is_gds_preferred(self) -> bool {
        self != CompatMode::On
    }

    /// Returns `true` if this is the POSIX-only compatibility mode.
    pub fn is_compat(self) -> bool {
        self == CompatMode::On
    }
}

impl std::fmt::Display for CompatMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompatMode::Off => f.write_str("OFF"),
            CompatMode::On => f.write_str("ON"),
            CompatMode::Auto => f.write_str("AUTO"),
        }
    }
}

/// Check whether the system is running under WSL (Windows Subsystem for Linux).
///
/// GDS is not supported on WSL. Detection checks `/proc/version` for "microsoft".
pub fn is_wsl() -> bool {
    std::fs::read_to_string("/proc/version")
        .map(|v| v.to_ascii_lowercase().contains("microsoft"))
        .unwrap_or(false)
}

/// Check whether `/run/udev` is readable.
///
/// This directory is required for GDS device detection and is absent in many
/// Docker containers.
pub fn is_udev_readable() -> bool {
    Path::new("/run/udev").is_dir()
}

/// Determine the effective compatibility mode for the runtime environment.
///
/// When `mode` is `Auto`, this function probes the system:
/// - If WSL is detected, returns `On` (POSIX fallback).
/// - If `/run/udev` is not readable, returns `On`.
/// - Otherwise returns `Off` (try GDS).
///
/// When `mode` is `On` or `Off`, it is returned unchanged.
pub fn resolve_compat_mode(mode: CompatMode) -> CompatMode {
    match mode {
        CompatMode::On | CompatMode::Off => mode,
        CompatMode::Auto => {
            if is_wsl() || !is_udev_readable() {
                CompatMode::On
            } else {
                CompatMode::Off
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Parsing tests (mirroring C++ test_defaults.cpp parse_compat_mode_str) ----

    #[test]
    fn test_parse_on_variants() {
        for s in &["ON", "on", "On", "TRUE", "true", "True", "YES", "yes", "1"] {
            assert_eq!(
                CompatMode::parse(s).unwrap(),
                CompatMode::On,
                "failed to parse {s:?} as On"
            );
        }
    }

    #[test]
    fn test_parse_off_variants() {
        for s in &[
            "OFF", "off", "Off", "FALSE", "false", "False", "NO", "no", "0",
        ] {
            assert_eq!(
                CompatMode::parse(s).unwrap(),
                CompatMode::Off,
                "failed to parse {s:?} as Off"
            );
        }
    }

    #[test]
    fn test_parse_auto_variants() {
        for s in &["AUTO", "auto", "Auto", "aUtO"] {
            assert_eq!(
                CompatMode::parse(s).unwrap(),
                CompatMode::Auto,
                "failed to parse {s:?} as Auto"
            );
        }
    }

    #[test]
    fn test_parse_invalid() {
        let err = CompatMode::parse("invalid").unwrap_err();
        assert_eq!(err.kind(), ErrorKind::ConfigInvalid);
        assert!(err.message().contains("unknown compatibility mode"));
    }

    #[test]
    fn test_parse_whitespace_trimmed() {
        assert_eq!(CompatMode::parse("  on  ").unwrap(), CompatMode::On);
        assert_eq!(CompatMode::parse("\tauto\n").unwrap(), CompatMode::Auto);
    }

    // ---- Display tests ----

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", CompatMode::Off), "OFF");
        assert_eq!(format!("{}", CompatMode::On), "ON");
        assert_eq!(format!("{}", CompatMode::Auto), "AUTO");
    }

    // ---- Default test ----

    #[test]
    fn test_default_is_auto() {
        assert_eq!(CompatMode::default(), CompatMode::Auto);
    }

    // ---- Helper method tests ----

    #[test]
    fn test_is_gds_preferred() {
        assert!(CompatMode::Off.is_gds_preferred());
        assert!(!CompatMode::On.is_gds_preferred());
        assert!(CompatMode::Auto.is_gds_preferred());
    }

    #[test]
    fn test_is_compat() {
        assert!(!CompatMode::Off.is_compat());
        assert!(CompatMode::On.is_compat());
        assert!(!CompatMode::Auto.is_compat());
    }

    // ---- Resolution tests ----

    #[test]
    fn test_resolve_explicit_modes_unchanged() {
        assert_eq!(resolve_compat_mode(CompatMode::On), CompatMode::On);
        assert_eq!(resolve_compat_mode(CompatMode::Off), CompatMode::Off);
    }

    #[test]
    fn test_resolve_auto_returns_on_or_off() {
        let resolved = resolve_compat_mode(CompatMode::Auto);
        assert!(resolved == CompatMode::On || resolved == CompatMode::Off);
    }

    // ---- Environment detection tests ----

    #[test]
    fn test_is_wsl_returns_bool() {
        // We can't control the environment, but we can verify it returns a bool
        // without panicking.
        let _ = is_wsl();
    }

    #[test]
    fn test_is_udev_readable_returns_bool() {
        let _ = is_udev_readable();
    }
}
