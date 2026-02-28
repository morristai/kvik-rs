//! Configuration environment variable integration tests.
//!
//! Mirrors C++ kvikio's `test_defaults.cpp` for environment variable parsing,
//! alias handling, and edge cases.

mod test_utils;

use kvik_rs::{CompatMode, Config, ErrorKind};

use test_utils::EnvVarGuard;

// ---- Test all environment variables ----

#[test]
fn test_env_compat_mode() {
    let _guard = EnvVarGuard::new(&[("KVIKIO_COMPAT_MODE", "OFF")]);
    let config = Config::from_env().unwrap();
    assert_eq!(config.compat_mode, CompatMode::Off);
}

#[test]
fn test_env_nthreads() {
    let _guard = EnvVarGuard::new(&[("KVIKIO_NTHREADS", "8")]);
    let config = Config::from_env().unwrap();
    assert_eq!(config.num_threads, 8);
}

#[test]
fn test_env_num_threads_alias() {
    let _guard = EnvVarGuard::new(&[("KVIKIO_NUM_THREADS", "16")]);
    let config = Config::from_env().unwrap();
    assert_eq!(config.num_threads, 16);
}

#[test]
fn test_env_task_size() {
    let _guard = EnvVarGuard::new(&[("KVIKIO_TASK_SIZE", "8388608")]);
    let config = Config::from_env().unwrap();
    assert_eq!(config.task_size, 8 * 1024 * 1024);
}

#[test]
fn test_env_gds_threshold() {
    let _guard = EnvVarGuard::new(&[("KVIKIO_GDS_THRESHOLD", "2097152")]);
    let config = Config::from_env().unwrap();
    assert_eq!(config.gds_threshold, 2 * 1024 * 1024);
}

#[test]
fn test_env_bounce_buffer_size() {
    let _guard = EnvVarGuard::new(&[("KVIKIO_BOUNCE_BUFFER_SIZE", "33554432")]);
    let config = Config::from_env().unwrap();
    assert_eq!(config.bounce_buffer_size, 32 * 1024 * 1024);
}

#[test]
fn test_env_auto_direct_io_read() {
    let _guard = EnvVarGuard::new(&[("KVIKIO_AUTO_DIRECT_IO_READ", "false")]);
    let config = Config::from_env().unwrap();
    assert!(!config.auto_direct_io_read);
}

#[test]
fn test_env_auto_direct_io_write() {
    let _guard = EnvVarGuard::new(&[("KVIKIO_AUTO_DIRECT_IO_WRITE", "no")]);
    let config = Config::from_env().unwrap();
    assert!(!config.auto_direct_io_write);
}

// ---- Conflicting NTHREADS aliases ----

#[test]
fn test_conflicting_nthreads_aliases_same_value() {
    // Both set to the same value — should be OK, last one wins.
    let _guard = EnvVarGuard::new(&[
        ("KVIKIO_NTHREADS", "4"),
        ("KVIKIO_NUM_THREADS", "4"),
    ]);
    let config = Config::from_env().unwrap();
    assert_eq!(config.num_threads, 4);
}

#[test]
fn test_conflicting_nthreads_aliases_different_values() {
    // Different values — KVIKIO_NUM_THREADS wins (checked second, overwrites first).
    let _guard = EnvVarGuard::new(&[
        ("KVIKIO_NTHREADS", "4"),
        ("KVIKIO_NUM_THREADS", "8"),
    ]);
    let config = Config::from_env().unwrap();
    // KVIKIO_NUM_THREADS is checked after KVIKIO_NTHREADS, so it takes precedence.
    assert_eq!(config.num_threads, 8);
}

// ---- Invalid environment variable values ----

#[test]
fn test_invalid_nthreads() {
    let _guard = EnvVarGuard::new(&[("KVIKIO_NTHREADS", "abc")]);
    let err = Config::from_env().unwrap_err();
    assert_eq!(err.kind(), ErrorKind::ConfigInvalid);
}

#[test]
fn test_invalid_compat_mode() {
    let _guard = EnvVarGuard::new(&[("KVIKIO_COMPAT_MODE", "invalid")]);
    let err = Config::from_env().unwrap_err();
    assert_eq!(err.kind(), ErrorKind::ConfigInvalid);
}

#[test]
fn test_invalid_task_size_negative() {
    let _guard = EnvVarGuard::new(&[("KVIKIO_TASK_SIZE", "-1")]);
    let err = Config::from_env().unwrap_err();
    assert_eq!(err.kind(), ErrorKind::ConfigInvalid);
}

#[test]
fn test_invalid_bool_value() {
    let _guard = EnvVarGuard::new(&[("KVIKIO_AUTO_DIRECT_IO_READ", "maybe")]);
    let err = Config::from_env().unwrap_err();
    assert_eq!(err.kind(), ErrorKind::ConfigInvalid);
}

// ---- Empty values use defaults ----

#[test]
fn test_empty_env_values_use_defaults() {
    let _guard = EnvVarGuard::new(&[
        ("KVIKIO_NTHREADS", ""),
        ("KVIKIO_TASK_SIZE", ""),
    ]);
    let config = Config::from_env().unwrap();
    assert_eq!(config.num_threads, 1);
    assert_eq!(config.task_size, 4 * 1024 * 1024);
}

// ---- Config global singleton thread safety ----

#[test]
fn test_config_thread_safety() {
    let saved = Config::get();

    // Spawn multiple threads that read and write Config concurrently.
    std::thread::scope(|scope| {
        // Readers.
        for _ in 0..4 {
            scope.spawn(|| {
                for _ in 0..100 {
                    let config = Config::get();
                    // Just verify it doesn't panic and returns consistent data.
                    let _ = config.num_threads;
                    let _ = config.task_size;
                }
            });
        }

        // Writers.
        for i in 0..4 {
            scope.spawn(move || {
                for j in 0..100 {
                    Config::update(|c| {
                        c.num_threads = i * 100 + j;
                    });
                }
            });
        }
    });

    Config::set(saved);
}

// ---- Config set/get/update round-trip ----

#[test]
fn test_config_set_get_roundtrip() {
    let saved = Config::get();

    let custom = Config {
        compat_mode: CompatMode::On,
        num_threads: 42,
        task_size: 1234,
        gds_threshold: 5678,
        bounce_buffer_size: 9999,
        auto_direct_io_read: false,
        auto_direct_io_write: false,
    };

    Config::set(custom.clone());
    let got = Config::get();
    assert_eq!(got.compat_mode, CompatMode::On);
    assert_eq!(got.num_threads, 42);
    assert_eq!(got.task_size, 1234);
    assert_eq!(got.gds_threshold, 5678);
    assert_eq!(got.bounce_buffer_size, 9999);
    assert!(!got.auto_direct_io_read);
    assert!(!got.auto_direct_io_write);

    Config::set(saved);
}

// ---- Boolean parsing variants ----

#[test]
fn test_bool_parsing_variants() {
    for (value, expected) in &[
        ("true", true),
        ("TRUE", true),
        ("yes", true),
        ("YES", true),
        ("on", true),
        ("ON", true),
        ("1", true),
        ("false", false),
        ("FALSE", false),
        ("no", false),
        ("NO", false),
        ("off", false),
        ("OFF", false),
        ("0", false),
    ] {
        let _guard = EnvVarGuard::new(&[("KVIKIO_AUTO_DIRECT_IO_READ", value)]);
        let config = Config::from_env().unwrap();
        assert_eq!(
            config.auto_direct_io_read, *expected,
            "bool parsing failed for {value:?}"
        );
    }
}

// ---- CompatMode parsing variants through env ----

#[test]
fn test_compat_mode_parsing_via_env() {
    for (value, expected) in &[
        ("on", CompatMode::On),
        ("ON", CompatMode::On),
        ("true", CompatMode::On),
        ("yes", CompatMode::On),
        ("1", CompatMode::On),
        ("off", CompatMode::Off),
        ("OFF", CompatMode::Off),
        ("false", CompatMode::Off),
        ("no", CompatMode::Off),
        ("0", CompatMode::Off),
        ("auto", CompatMode::Auto),
        ("AUTO", CompatMode::Auto),
        ("Auto", CompatMode::Auto),
    ] {
        let _guard = EnvVarGuard::new(&[("KVIKIO_COMPAT_MODE", value)]);
        let config = Config::from_env().unwrap();
        assert_eq!(
            config.compat_mode, *expected,
            "compat mode parsing failed for {value:?}"
        );
    }
}
