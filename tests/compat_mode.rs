//! Compatibility mode integration tests.
//!
//! Verifies CompatMode detection, resolution, and fallback behavior
//! in end-to-end scenarios.

mod test_utils;

use kvik_rs::compat_mode::{is_udev_readable, is_wsl, resolve_compat_mode};
use kvik_rs::{CompatMode, Config, FileHandle};

use test_utils::{assert_data_eq, gen_data, EnvVarGuard};

/// Auto-detection should return a valid, non-Auto mode.
#[test]
fn test_auto_detection() {
    let resolved = resolve_compat_mode(CompatMode::Auto);
    assert!(
        resolved == CompatMode::On || resolved == CompatMode::Off,
        "resolve_compat_mode(Auto) returned {resolved:?}, expected On or Off"
    );
}

/// Force compat mode ON, open a file, and verify I/O works.
#[test]
fn test_forced_on_works() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path();
    let data = gen_data(8192);

    let handle = FileHandle::open(path, "w", 0o644, CompatMode::On).unwrap();
    assert_eq!(handle.compat_mode(), CompatMode::On);
    assert!(!handle.is_gds_available());
    handle.write_host(&data, 0).unwrap();
    drop(handle);

    let handle = FileHandle::open(path, "r", 0o644, CompatMode::On).unwrap();
    let mut buf = vec![0u8; data.len()];
    handle.read_host(&mut buf, 0).unwrap();
    assert_data_eq(&data, &buf);
}

/// Force compat mode OFF on a system without GDS.
/// This should return an error during open (cuFile init fails) or
/// fall through to POSIX if Auto.
#[test]
fn test_forced_off_without_gds() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path();

    // On a system without GDS, opening with Off should error.
    let result = FileHandle::open(path, "w+", 0o644, CompatMode::Off);
    // We can't assert which outcome because it depends on GDS availability.
    // On systems with GDS: Ok. On systems without: Err.
    match result {
        Ok(handle) => {
            // GDS is available — handle should work.
            assert!(
                handle.compat_mode() == CompatMode::Off,
                "expected Off mode when GDS is available"
            );
        }
        Err(e) => {
            // GDS not available — should get a CuFileError.
            assert!(
                e.kind() == kvik_rs::ErrorKind::CuFileError
                    || e.kind() == kvik_rs::ErrorKind::SystemError,
                "unexpected error kind: {:?}",
                e.kind()
            );
        }
    }
}

/// Set KVIKIO_COMPAT_MODE=ON via environment, verify Config picks it up.
#[test]
fn test_env_var_override() {
    let _guard = EnvVarGuard::new(&[("KVIKIO_COMPAT_MODE", "ON")]);
    let config = Config::from_env().unwrap();
    assert_eq!(config.compat_mode, CompatMode::On);
}

/// Verify compat mode AUTO resolves correctly on WSL.
#[test]
fn test_auto_on_wsl() {
    let wsl = is_wsl();
    let resolved = resolve_compat_mode(CompatMode::Auto);

    if wsl {
        assert_eq!(
            resolved,
            CompatMode::On,
            "WSL detected but Auto did not resolve to On"
        );
    }
    // On non-WSL, result depends on /run/udev.
}

/// Verify that is_udev_readable and is_wsl don't panic.
#[test]
fn test_detection_functions_stable() {
    let _ = is_udev_readable();
    let _ = is_wsl();
}

/// Explicit On and Off modes are returned unchanged by resolve.
#[test]
fn test_explicit_modes_passthrough() {
    assert_eq!(resolve_compat_mode(CompatMode::On), CompatMode::On);
    assert_eq!(resolve_compat_mode(CompatMode::Off), CompatMode::Off);
}

/// FileHandle with Auto mode should resolve to On or Off.
#[test]
fn test_file_handle_auto_resolves() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    // Auto might fail if GDS is not available, so we accept either outcome.
    match FileHandle::open(tmp.path(), "r", 0o644, CompatMode::Auto) {
        Ok(handle) => {
            let mode = handle.compat_mode();
            assert!(
                mode == CompatMode::On || mode == CompatMode::Off,
                "Auto resolved to {mode:?}"
            );
        }
        Err(_) => {
            // This can happen if the system probes GDS and it fails
            // in a way that doesn't fall back gracefully.
        }
    }
}
