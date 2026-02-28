//! Runtime-configurable defaults for kvik-rs.
//!
//! Mirrors the C++ kvikio `defaults` namespace. Configuration is read from
//! environment variables on first access, and can be modified at runtime
//! through a global singleton.
//!
//! # Environment Variables
//!
//! | Variable | Type | Default | Description |
//! |----------|------|---------|-------------|
//! | `KVIKIO_COMPAT_MODE` | `CompatMode` | `Auto` | Compatibility mode |
//! | `KVIKIO_NTHREADS` | `usize` | `1` | Number of I/O threads |
//! | `KVIKIO_TASK_SIZE` | `usize` | `4194304` (4 MiB) | Parallel I/O chunk size |
//! | `KVIKIO_GDS_THRESHOLD` | `usize` | `1048576` (1 MiB) | Min size for GDS |
//! | `KVIKIO_BOUNCE_BUFFER_SIZE` | `usize` | `16777216` (16 MiB) | Bounce buffer size |
//! | `KVIKIO_AUTO_DIRECT_IO_READ` | `bool` | `true` | Enable Direct I/O for reads |
//! | `KVIKIO_AUTO_DIRECT_IO_WRITE` | `bool` | `true` | Enable Direct I/O for writes |

use std::sync::{OnceLock, RwLock};

use crate::compat_mode::CompatMode;
use crate::error::{Error, ErrorKind};

/// Default number of I/O threads.
const DEFAULT_NUM_THREADS: usize = 1;

/// Default parallel I/O task size: 4 MiB.
const DEFAULT_TASK_SIZE: usize = 4 * 1024 * 1024;

/// Default minimum transfer size for GDS: 1 MiB.
const DEFAULT_GDS_THRESHOLD: usize = 1024 * 1024;

/// Default bounce buffer size: 16 MiB.
const DEFAULT_BOUNCE_BUFFER_SIZE: usize = 16 * 1024 * 1024;

/// Runtime configuration for kvik-rs.
///
/// Thread-safe access through the global singleton via [`Config::get`] and [`Config::set`].
#[derive(Debug, Clone)]
pub struct Config {
    /// Compatibility mode (GDS/POSIX/Auto).
    pub compat_mode: CompatMode,
    /// Number of threads for parallel I/O.
    pub num_threads: usize,
    /// Chunk size for parallel I/O operations (bytes).
    pub task_size: usize,
    /// Minimum transfer size to use GDS instead of POSIX (bytes).
    pub gds_threshold: usize,
    /// Size of each bounce buffer (bytes).
    pub bounce_buffer_size: usize,
    /// Whether to use Direct I/O for read operations when possible.
    pub auto_direct_io_read: bool,
    /// Whether to use Direct I/O for write operations when possible.
    pub auto_direct_io_write: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            compat_mode: CompatMode::Auto,
            num_threads: DEFAULT_NUM_THREADS,
            task_size: DEFAULT_TASK_SIZE,
            gds_threshold: DEFAULT_GDS_THRESHOLD,
            bounce_buffer_size: DEFAULT_BOUNCE_BUFFER_SIZE,
            auto_direct_io_read: true,
            auto_direct_io_write: true,
        }
    }
}

/// Global configuration singleton.
static GLOBAL_CONFIG: OnceLock<RwLock<Config>> = OnceLock::new();

impl Config {
    /// Create a configuration from environment variables, falling back to defaults.
    pub fn from_env() -> crate::error::Result<Self> {
        let mut config = Config::default();

        if let Some(val) = env_var("KVIKIO_COMPAT_MODE")? {
            config.compat_mode = CompatMode::parse(&val)?;
        }

        if let Some(val) = env_var("KVIKIO_NTHREADS")? {
            config.num_threads = parse_usize(&val, "KVIKIO_NTHREADS")?;
        }
        // Also check the alias used by some C++ kvikio versions.
        if let Some(val) = env_var("KVIKIO_NUM_THREADS")? {
            config.num_threads = parse_usize(&val, "KVIKIO_NUM_THREADS")?;
        }

        if let Some(val) = env_var("KVIKIO_TASK_SIZE")? {
            config.task_size = parse_usize(&val, "KVIKIO_TASK_SIZE")?;
        }

        if let Some(val) = env_var("KVIKIO_GDS_THRESHOLD")? {
            config.gds_threshold = parse_usize(&val, "KVIKIO_GDS_THRESHOLD")?;
        }

        if let Some(val) = env_var("KVIKIO_BOUNCE_BUFFER_SIZE")? {
            config.bounce_buffer_size = parse_usize(&val, "KVIKIO_BOUNCE_BUFFER_SIZE")?;
        }

        if let Some(val) = env_var("KVIKIO_AUTO_DIRECT_IO_READ")? {
            config.auto_direct_io_read = parse_bool(&val, "KVIKIO_AUTO_DIRECT_IO_READ")?;
        }

        if let Some(val) = env_var("KVIKIO_AUTO_DIRECT_IO_WRITE")? {
            config.auto_direct_io_write = parse_bool(&val, "KVIKIO_AUTO_DIRECT_IO_WRITE")?;
        }

        Ok(config)
    }

    /// Get a read-only snapshot of the global configuration.
    ///
    /// On first call, reads from environment variables. Subsequent calls return
    /// the (possibly modified) global configuration.
    pub fn get() -> Config {
        let lock = GLOBAL_CONFIG.get_or_init(|| {
            let config = Config::from_env().unwrap_or_default();
            RwLock::new(config)
        });
        lock.read().expect("config lock poisoned").clone()
    }

    /// Replace the global configuration.
    pub fn set(config: Config) {
        let lock = GLOBAL_CONFIG.get_or_init(|| RwLock::new(Config::default()));
        let mut guard = lock.write().expect("config lock poisoned");
        *guard = config;
    }

    /// Modify the global configuration in place via a closure.
    pub fn update(f: impl FnOnce(&mut Config)) {
        let lock = GLOBAL_CONFIG.get_or_init(|| RwLock::new(Config::default()));
        let mut guard = lock.write().expect("config lock poisoned");
        f(&mut guard);
    }
}

/// Read an environment variable, returning `None` if it is unset or empty.
fn env_var(name: &str) -> crate::error::Result<Option<String>> {
    match std::env::var(name) {
        Ok(val) if val.is_empty() => Ok(None),
        Ok(val) => Ok(Some(val)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(Error::new(
            ErrorKind::ConfigInvalid,
            format!("environment variable {name} contains invalid Unicode"),
        )
        .with_operation("Config::from_env")
        .with_context("env_var", name)),
    }
}

fn parse_usize(val: &str, var_name: &str) -> crate::error::Result<usize> {
    val.trim().parse::<usize>().map_err(|e| {
        Error::new(
            ErrorKind::ConfigInvalid,
            format!("invalid value for {var_name}: {val:?}"),
        )
        .with_operation("Config::from_env")
        .with_context("env_var", var_name)
        .set_source(e)
    })
}

fn parse_bool(val: &str, var_name: &str) -> crate::error::Result<bool> {
    match val.trim().to_ascii_lowercase().as_str() {
        "true" | "on" | "yes" | "1" => Ok(true),
        "false" | "off" | "no" | "0" => Ok(false),
        _ => Err(Error::new(
            ErrorKind::ConfigInvalid,
            format!("invalid boolean value for {var_name}: {val:?}"),
        )
        .with_operation("Config::from_env")
        .with_context("env_var", var_name)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    // Helper to temporarily set env vars for a test.
    // Note: env var tests are inherently not thread-safe, but Rust's test
    // runner runs them in separate threads. We use unique var names where possible.
    struct EnvVarGuard {
        vars: Vec<(String, Option<String>)>,
    }

    impl EnvVarGuard {
        fn new(vars: &[(&str, &str)]) -> Self {
            let mut saved = Vec::new();
            for (key, value) in vars {
                saved.push((key.to_string(), env::var(key).ok()));
                // SAFETY: Tests are run with --test-threads=1 or we use unique
                // env var names. No other threads are reading these env vars.
                unsafe { env::set_var(key, value) };
            }
            EnvVarGuard { vars: saved }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            for (key, prev) in &self.vars {
                // SAFETY: See EnvVarGuard::new.
                match prev {
                    Some(val) => unsafe { env::set_var(key, val) },
                    None => unsafe { env::remove_var(key) },
                }
            }
        }
    }

    // ---- Default config tests ----

    #[test]
    fn test_config_defaults() {
        let config = Config::default();
        assert_eq!(config.compat_mode, CompatMode::Auto);
        assert_eq!(config.num_threads, 1);
        assert_eq!(config.task_size, 4 * 1024 * 1024);
        assert_eq!(config.gds_threshold, 1024 * 1024);
        assert_eq!(config.bounce_buffer_size, 16 * 1024 * 1024);
        assert!(config.auto_direct_io_read);
        assert!(config.auto_direct_io_write);
    }

    // ---- Environment variable parsing tests ----

    #[test]
    fn test_parse_usize_valid() {
        assert_eq!(parse_usize("42", "TEST").unwrap(), 42);
        assert_eq!(parse_usize("  100  ", "TEST").unwrap(), 100);
        assert_eq!(parse_usize("0", "TEST").unwrap(), 0);
        assert_eq!(parse_usize("8388608", "TEST").unwrap(), 8388608);
    }

    #[test]
    fn test_parse_usize_invalid() {
        let err = parse_usize("abc", "TEST_VAR").unwrap_err();
        assert_eq!(err.kind(), ErrorKind::ConfigInvalid);
        assert!(err.message().contains("TEST_VAR"));

        let err = parse_usize("-1", "TEST_VAR").unwrap_err();
        assert_eq!(err.kind(), ErrorKind::ConfigInvalid);
    }

    #[test]
    fn test_parse_bool_valid() {
        for s in &["true", "TRUE", "True", "on", "ON", "yes", "YES", "1"] {
            assert!(parse_bool(s, "TEST").unwrap(), "expected true for {s:?}");
        }
        for s in &["false", "FALSE", "False", "off", "OFF", "no", "NO", "0"] {
            assert!(!parse_bool(s, "TEST").unwrap(), "expected false for {s:?}");
        }
    }

    #[test]
    fn test_parse_bool_invalid() {
        let err = parse_bool("maybe", "TEST_VAR").unwrap_err();
        assert_eq!(err.kind(), ErrorKind::ConfigInvalid);
    }

    #[test]
    fn test_parse_bool_trimmed() {
        assert!(parse_bool("  true  ", "TEST").unwrap());
        assert!(!parse_bool("\tfalse\n", "TEST").unwrap());
    }

    // ---- from_env tests (mirroring C++ test_defaults.cpp) ----

    #[test]
    fn test_from_env_with_compat_mode() {
        let _guard = EnvVarGuard::new(&[("KVIKIO_COMPAT_MODE", "ON")]);
        let config = Config::from_env().unwrap();
        assert_eq!(config.compat_mode, CompatMode::On);
    }

    #[test]
    fn test_from_env_with_nthreads() {
        let _guard = EnvVarGuard::new(&[("KVIKIO_NTHREADS", "4")]);
        let config = Config::from_env().unwrap();
        assert_eq!(config.num_threads, 4);
    }

    #[test]
    fn test_from_env_with_task_size() {
        let _guard = EnvVarGuard::new(&[("KVIKIO_TASK_SIZE", "8388608")]);
        let config = Config::from_env().unwrap();
        assert_eq!(config.task_size, 8 * 1024 * 1024);
    }

    #[test]
    fn test_from_env_with_direct_io() {
        let _guard = EnvVarGuard::new(&[
            ("KVIKIO_AUTO_DIRECT_IO_READ", "false"),
            ("KVIKIO_AUTO_DIRECT_IO_WRITE", "false"),
        ]);
        let config = Config::from_env().unwrap();
        assert!(!config.auto_direct_io_read);
        assert!(!config.auto_direct_io_write);
    }

    #[test]
    fn test_from_env_empty_values_use_defaults() {
        let _guard = EnvVarGuard::new(&[("KVIKIO_NTHREADS", "")]);
        let config = Config::from_env().unwrap();
        assert_eq!(config.num_threads, DEFAULT_NUM_THREADS);
    }

    // ---- Global singleton tests ----

    #[test]
    fn test_config_set_and_get() {
        Config::set(Config {
            num_threads: 8,
            task_size: 1024,
            ..Config::default()
        });

        let retrieved = Config::get();
        assert_eq!(retrieved.num_threads, 8);
        assert_eq!(retrieved.task_size, 1024);

        // Restore default so other tests aren't affected
        Config::set(Config::default());
    }

    #[test]
    fn test_config_update() {
        Config::set(Config::default());
        Config::update(|c| {
            c.bounce_buffer_size = 42;
        });
        let config = Config::get();
        assert_eq!(config.bounce_buffer_size, 42);

        // Restore
        Config::set(Config::default());
    }
}
