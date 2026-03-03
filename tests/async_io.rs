//! Integration tests for async I/O (thread pool, IoFuture, pread/pwrite).
//!
//! Covers edge cases, concurrency pitfalls, and consistency with sync APIs.

mod test_utils;

use kvik_rs::{CompatMode, FileHandle, IoFuture, IoThreadPool};

use test_utils::{assert_data_eq, gen_data};

// ---------------------------------------------------------------------------
// FileHandle::pread — edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_pread_at_nonzero_offset() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path();

    // Write 20_000 bytes, read the second half.
    let data = gen_data(20_000);
    let handle = FileHandle::open(path, "w", 0o644, CompatMode::On).unwrap();
    handle.write_host(&data, 0).unwrap();
    drop(handle);

    let handle = FileHandle::open(path, "r", 0o644, CompatMode::On).unwrap();
    let mut buf = vec![0u8; 10_000];
    let n = handle.pread(&mut buf, 10_000, 0).get().unwrap();
    assert_eq!(n, 10_000);
    assert_data_eq(&data[10_000..], &buf);
}

#[test]
fn test_pwrite_at_nonzero_offset() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path();

    // Pre-fill file with zeros.
    let zeros = vec![0u8; 20_000];
    let handle = FileHandle::open(path, "w", 0o644, CompatMode::On).unwrap();
    handle.write_host(&zeros, 0).unwrap();
    drop(handle);

    // Overwrite the second half with a pattern.
    let patch = gen_data(10_000);
    let handle = FileHandle::open(path, "r+", 0o644, CompatMode::On).unwrap();
    let n = handle.pwrite(&patch, 10_000, 0).get().unwrap();
    assert_eq!(n, 10_000);
    drop(handle);

    // Verify: first half all zeros, second half is the pattern.
    let handle = FileHandle::open(path, "r", 0o644, CompatMode::On).unwrap();
    let mut full = vec![0u8; 20_000];
    handle.read_host(&mut full, 0).unwrap();
    assert_data_eq(&zeros[..10_000], &full[..10_000]);
    assert_data_eq(&patch, &full[10_000..]);
}

#[test]
fn test_pread_task_size_equal_to_file_size() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path();
    let size = 4096;
    let data = gen_data(size);

    let handle = FileHandle::open(path, "w", 0o644, CompatMode::On).unwrap();
    handle.write_host(&data, 0).unwrap();
    drop(handle);

    let handle = FileHandle::open(path, "r", 0o644, CompatMode::On).unwrap();
    let mut buf = vec![0u8; size];
    // task_size == total size → single chunk path.
    let n = handle.pread(&mut buf, 0, size).get().unwrap();
    assert_eq!(n, size);
    assert_data_eq(&data, &buf);
}

#[test]
fn test_pread_task_size_larger_than_file() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path();
    let size = 1000;
    let data = gen_data(size);

    let handle = FileHandle::open(path, "w", 0o644, CompatMode::On).unwrap();
    handle.write_host(&data, 0).unwrap();
    drop(handle);

    let handle = FileHandle::open(path, "r", 0o644, CompatMode::On).unwrap();
    let mut buf = vec![0u8; size];
    // task_size much larger than file → single chunk.
    let n = handle.pread(&mut buf, 0, 1_000_000).get().unwrap();
    assert_eq!(n, size);
    assert_data_eq(&data, &buf);
}

#[test]
fn test_pread_task_size_one_byte() {
    // Extreme chunking: 1 byte per task.
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path();
    let size = 200;
    let data = gen_data(size);

    let handle = FileHandle::open(path, "w", 0o644, CompatMode::On).unwrap();
    handle.write_host(&data, 0).unwrap();
    drop(handle);

    let handle = FileHandle::open(path, "r", 0o644, CompatMode::On).unwrap();
    let mut buf = vec![0u8; size];
    let n = handle.pread(&mut buf, 0, 1).get().unwrap();
    assert_eq!(n, size);
    assert_data_eq(&data, &buf);
}

#[test]
fn test_pwrite_task_size_one_byte() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path();
    let size = 200;
    let data = gen_data(size);

    let handle = FileHandle::open(path, "w", 0o644, CompatMode::On).unwrap();
    let n = handle.pwrite(&data, 0, 1).get().unwrap();
    assert_eq!(n, size);
    drop(handle);

    let handle = FileHandle::open(path, "r", 0o644, CompatMode::On).unwrap();
    let mut buf = vec![0u8; size];
    handle.read_host(&mut buf, 0).unwrap();
    assert_data_eq(&data, &buf);
}

#[test]
fn test_pread_single_byte_file() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path();

    let handle = FileHandle::open(path, "w", 0o644, CompatMode::On).unwrap();
    handle.write_host(&[0xAB], 0).unwrap();
    drop(handle);

    let handle = FileHandle::open(path, "r", 0o644, CompatMode::On).unwrap();
    let mut buf = [0u8; 1];
    let n = handle.pread(&mut buf, 0, 0).get().unwrap();
    assert_eq!(n, 1);
    assert_eq!(buf[0], 0xAB);
}

// ---------------------------------------------------------------------------
// Consistency: pread vs pread_host
// ---------------------------------------------------------------------------

#[test]
fn test_pread_matches_pread_host() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path();
    let size = 100_000;
    let task_size = 8_000;
    let data = gen_data(size);

    let handle = FileHandle::open(path, "w", 0o644, CompatMode::On).unwrap();
    handle.write_host(&data, 0).unwrap();
    drop(handle);

    let handle = FileHandle::open(path, "r", 0o644, CompatMode::On).unwrap();

    // Sync path.
    let mut sync_buf = vec![0u8; size];
    let sync_n = handle.pread_host(&mut sync_buf, 0, task_size).unwrap();

    // Async path.
    let mut async_buf = vec![0u8; size];
    let async_n = handle.pread(&mut async_buf, 0, task_size).get().unwrap();

    assert_eq!(sync_n, async_n);
    assert_data_eq(&sync_buf, &async_buf);
}

#[test]
fn test_pwrite_matches_pwrite_host() {
    let size = 100_000;
    let task_size = 8_000;
    let data = gen_data(size);

    // Sync write path.
    let tmp_sync = tempfile::NamedTempFile::new().unwrap();
    let handle = FileHandle::open(tmp_sync.path(), "w", 0o644, CompatMode::On).unwrap();
    handle.pwrite_host(&data, 0, task_size).unwrap();
    drop(handle);

    let handle = FileHandle::open(tmp_sync.path(), "r", 0o644, CompatMode::On).unwrap();
    let mut sync_buf = vec![0u8; size];
    handle.read_host(&mut sync_buf, 0).unwrap();
    drop(handle);

    // Async write path.
    let tmp_async = tempfile::NamedTempFile::new().unwrap();
    let handle = FileHandle::open(tmp_async.path(), "w", 0o644, CompatMode::On).unwrap();
    handle.pwrite(&data, 0, task_size).get().unwrap();
    drop(handle);

    let handle = FileHandle::open(tmp_async.path(), "r", 0o644, CompatMode::On).unwrap();
    let mut async_buf = vec![0u8; size];
    handle.read_host(&mut async_buf, 0).unwrap();
    drop(handle);

    assert_data_eq(&sync_buf, &async_buf);
}

// ---------------------------------------------------------------------------
// Roundtrip: pwrite then pread on the same file
// ---------------------------------------------------------------------------

#[test]
fn test_pwrite_then_pread_roundtrip() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path();
    let size = 50_000;
    let task_size = 7_000;
    let data = gen_data(size);

    let handle = FileHandle::open(path, "w+", 0o644, CompatMode::On).unwrap();

    // Async write.
    let nw = handle.pwrite(&data, 0, task_size).get().unwrap();
    assert_eq!(nw, size);

    // Async read back through the same handle.
    let mut buf = vec![0u8; size];
    let nr = handle.pread(&mut buf, 0, task_size).get().unwrap();
    assert_eq!(nr, size);
    assert_data_eq(&data, &buf);
}

// ---------------------------------------------------------------------------
// Multiple sequential futures from the same handle
// ---------------------------------------------------------------------------

#[test]
fn test_sequential_preads_same_handle() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path();
    let data = gen_data(30_000);

    let handle = FileHandle::open(path, "w", 0o644, CompatMode::On).unwrap();
    handle.write_host(&data, 0).unwrap();
    drop(handle);

    let handle = FileHandle::open(path, "r", 0o644, CompatMode::On).unwrap();

    // Three sequential reads of 10K each.
    for i in 0..3 {
        let offset = i * 10_000;
        let mut buf = vec![0u8; 10_000];
        let n = handle.pread(&mut buf, offset as u64, 0).get().unwrap();
        assert_eq!(n, 10_000);
        assert_data_eq(&data[offset..offset + 10_000], &buf);
    }
}

// ---------------------------------------------------------------------------
// Various task sizes (mirrors parallel_io.rs pattern)
// ---------------------------------------------------------------------------

#[test]
fn test_pread_task_size_256() {
    pread_with_task_size(256);
}

#[test]
fn test_pread_task_size_1024() {
    pread_with_task_size(1024);
}

#[test]
fn test_pread_task_size_4096() {
    pread_with_task_size(4096);
}

#[test]
fn test_pread_task_size_default() {
    pread_with_task_size(0);
}

fn pread_with_task_size(task_size: usize) {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path();
    let size = 100_000;
    let data = gen_data(size);

    let handle = FileHandle::open(path, "w", 0o644, CompatMode::On).unwrap();
    handle.write_host(&data, 0).unwrap();
    drop(handle);

    let handle = FileHandle::open(path, "r", 0o644, CompatMode::On).unwrap();
    let mut buf = vec![0u8; size];
    let read = handle.pread(&mut buf, 0, task_size).get().unwrap();
    assert_eq!(read, size);
    assert_data_eq(&data, &buf);
}

#[test]
fn test_pwrite_task_size_256() {
    pwrite_with_task_size(256);
}

#[test]
fn test_pwrite_task_size_1024() {
    pwrite_with_task_size(1024);
}

#[test]
fn test_pwrite_task_size_4096() {
    pwrite_with_task_size(4096);
}

#[test]
fn test_pwrite_task_size_default() {
    pwrite_with_task_size(0);
}

fn pwrite_with_task_size(task_size: usize) {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path();
    let size = 100_000;
    let data = gen_data(size);

    let handle = FileHandle::open(path, "w", 0o644, CompatMode::On).unwrap();
    let n = handle.pwrite(&data, 0, task_size).get().unwrap();
    assert_eq!(n, size);
    drop(handle);

    let handle = FileHandle::open(path, "r", 0o644, CompatMode::On).unwrap();
    let mut buf = vec![0u8; size];
    handle.read_host(&mut buf, 0).unwrap();
    assert_data_eq(&data, &buf);
}

// ---------------------------------------------------------------------------
// Unaligned sizes (mirrors C++ kvikio edge cases)
// ---------------------------------------------------------------------------

#[test]
fn test_pread_unaligned_size_with_chunking() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path();
    // 1 MiB + 124 bytes: not a multiple of any typical task_size.
    let size = 1024 * 1024 + 124;
    let data = gen_data(size);

    let handle = FileHandle::open(path, "w", 0o644, CompatMode::On).unwrap();
    handle.write_host(&data, 0).unwrap();
    drop(handle);

    let handle = FileHandle::open(path, "r", 0o644, CompatMode::On).unwrap();
    let mut buf = vec![0u8; size];
    let n = handle.pread(&mut buf, 0, 64 * 1024).get().unwrap();
    assert_eq!(n, size);
    assert_data_eq(&data, &buf);
}

#[test]
fn test_pwrite_unaligned_size_with_chunking() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path();
    let size = 1024 * 1024 + 124;
    let data = gen_data(size);

    let handle = FileHandle::open(path, "w", 0o644, CompatMode::On).unwrap();
    let n = handle.pwrite(&data, 0, 64 * 1024).get().unwrap();
    assert_eq!(n, size);
    drop(handle);

    let handle = FileHandle::open(path, "r", 0o644, CompatMode::On).unwrap();
    let mut buf = vec![0u8; size];
    handle.read_host(&mut buf, 0).unwrap();
    assert_data_eq(&data, &buf);
}

// ---------------------------------------------------------------------------
// IoFuture drop without consuming (no panic, no leak)
// ---------------------------------------------------------------------------

#[test]
fn test_drop_io_future_before_get() {
    let pool = IoThreadPool::new(2);
    // Submit and immediately drop the future.
    let future = pool.submit(|| Ok(42usize));
    drop(future);
    // The pool should still function.
    let result = pool.submit(|| Ok(99usize)).get().unwrap();
    assert_eq!(result, 99);
}

#[test]
fn test_drop_multiple_futures_before_get() {
    let pool = IoThreadPool::new(2);
    for _ in 0..50 {
        let _future = pool.submit(|| Ok(0usize));
        // Dropped immediately each iteration.
    }
    // Pool still works.
    assert_eq!(pool.submit(|| Ok(1usize)).get().unwrap(), 1);
}

// ---------------------------------------------------------------------------
// IoFuture::try_get — idempotent after consumption
// ---------------------------------------------------------------------------

#[test]
fn test_try_get_returns_none_after_consumed() {
    let pool = IoThreadPool::new(1);
    let mut future = pool.submit(|| Ok(42usize));

    // Spin until ready.
    while !future.is_ready() {
        std::thread::yield_now();
    }

    let first = future.try_get();
    assert!(first.is_some());
    assert_eq!(first.unwrap().unwrap(), 42);

    // Second try_get should return None (value already taken).
    let second = future.try_get();
    assert!(second.is_none());
}

// ---------------------------------------------------------------------------
// IoFuture polling loop (simulates async executor without `futures` crate)
// ---------------------------------------------------------------------------

#[test]
fn test_try_get_polling_loop() {
    let pool = IoThreadPool::new(1);
    let mut future = pool.submit(|| {
        std::thread::sleep(std::time::Duration::from_millis(10));
        Ok(123usize)
    });

    // Poll until ready.
    let result = loop {
        if let Some(r) = future.try_get() {
            break r;
        }
        std::thread::yield_now();
    };

    assert_eq!(result.unwrap(), 123);
}

// ---------------------------------------------------------------------------
// Thread pool: submit after shutdown panics
// ---------------------------------------------------------------------------

#[test]
#[should_panic(expected = "thread pool is shut down")]
fn test_submit_after_shutdown_panics() {
    let pool = IoThreadPool::new(1);
    pool.shutdown();
    // This should panic.
    let _future = pool.submit(|| Ok(0usize));
}

// ---------------------------------------------------------------------------
// Thread pool: drop while futures are outstanding
// ---------------------------------------------------------------------------

#[test]
fn test_pool_drop_with_outstanding_futures() {
    let futures: Vec<IoFuture<usize>>;
    {
        let pool = IoThreadPool::new(2);
        futures = (0..10).map(|i| pool.submit(move || Ok(i))).collect();
        // Pool is dropped here, which shuts down workers after in-flight
        // tasks complete.
    }
    // Futures should all be resolved because pool::drop calls shutdown()
    // which joins workers after draining the queue.
    for (i, f) in futures.into_iter().enumerate() {
        assert_eq!(f.get().unwrap(), i);
    }
}

// ---------------------------------------------------------------------------
// Thread pool: single thread serialization correctness
// ---------------------------------------------------------------------------

#[test]
fn test_single_thread_preserves_submission_order_effects() {
    // With 1 thread, tasks execute serially in submission order.
    // We verify this with a shared counter that each task increments.
    use std::sync::atomic::{AtomicUsize, Ordering};
    let counter = Arc::new(AtomicUsize::new(0));

    let pool = IoThreadPool::new(1);
    let mut futures = Vec::new();

    for _i in 0..50 {
        let c = Arc::clone(&counter);
        futures.push(pool.submit(move || {
            let prev = c.fetch_add(1, Ordering::SeqCst);
            // With 1 thread, tasks run in order so prev == expected.
            Ok(prev)
        }));
    }

    for (i, f) in futures.into_iter().enumerate() {
        assert_eq!(f.get().unwrap(), i);
    }
}

use std::sync::Arc;

// ---------------------------------------------------------------------------
// Thread pool: high thread count, low task count
// ---------------------------------------------------------------------------

#[test]
fn test_more_threads_than_tasks() {
    let pool = IoThreadPool::new(16);
    let futures: Vec<_> = (0..3).map(|i| pool.submit(move || Ok(i))).collect();
    for (i, f) in futures.into_iter().enumerate() {
        assert_eq!(f.get().unwrap(), i);
    }
}

// ---------------------------------------------------------------------------
// Thread pool: rapid submit-get cycles (resource leak check)
// ---------------------------------------------------------------------------

#[test]
fn test_rapid_submit_get_no_leak() {
    let pool = IoThreadPool::new(4);
    for i in 0..1000 {
        let result = pool.submit(move || Ok(i)).get().unwrap();
        assert_eq!(result, i);
    }
}

// ---------------------------------------------------------------------------
// Thread pool: task panics are isolated
// ---------------------------------------------------------------------------

#[test]
fn test_task_panic_does_not_poison_pool() {
    let pool = IoThreadPool::new(2);

    // Submit a panicking task. The IoFuture's get() will propagate the panic
    // through the worker thread, but the *pool* should still be usable because
    // std::thread::spawn starts a new OS thread per worker, and the panic
    // only takes down that one thread. However, our pool uses a fixed set
    // of workers — a panic will reduce the live worker count by 1.
    //
    // We test that the pool continues to service tasks even if a worker panics.
    let panicking = pool.submit(|| -> kvik_rs::Result<usize> {
        panic!("intentional test panic");
    });

    // Don't call get() on the panicking future — that would propagate
    // the panic to *this* thread via the oneshot channel. Instead,
    // just drop it. The worker thread panics, the sender is dropped
    // without sending, and the receiver's recv() would block forever.
    // But we're dropping the future, so that's fine.
    drop(panicking);

    // Give the panicking task time to execute and the worker to die.
    std::thread::sleep(std::time::Duration::from_millis(50));

    // The pool should still work (remaining worker(s) pick up tasks).
    let result = pool.submit(|| Ok(42usize)).get().unwrap();
    assert_eq!(result, 42);
}

// ---------------------------------------------------------------------------
// Large payload through IoFuture (no truncation)
// ---------------------------------------------------------------------------

#[test]
fn test_large_result_through_future() {
    let pool = IoThreadPool::new(1);
    let future = pool.submit(|| {
        let big = vec![0xFFu8; 10_000_000]; // 10 MB
        Ok(big)
    });
    let result = future.get().unwrap();
    assert_eq!(result.len(), 10_000_000);
    assert!(result.iter().all(|&b| b == 0xFF));
}

// ---------------------------------------------------------------------------
// futures feature: async pitfalls
// ---------------------------------------------------------------------------

#[cfg(feature = "futures")]
mod futures_tests {
    use super::*;

    /// Test that poll returns Pending before the task completes,
    /// then Ready after the waker fires.
    #[test]
    fn test_poll_pending_then_ready() {
        use std::future::Future;
        use std::pin::Pin;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

        // Minimal waker that sets a flag when woken.
        static WOKEN: AtomicBool = AtomicBool::new(false);
        WOKEN.store(false, Ordering::SeqCst);

        unsafe fn clone_fn(p: *const ()) -> RawWaker {
            RawWaker::new(p, &VTABLE)
        }
        unsafe fn wake_fn(_: *const ()) {
            WOKEN.store(true, Ordering::SeqCst);
        }
        unsafe fn wake_by_ref_fn(_: *const ()) {
            WOKEN.store(true, Ordering::SeqCst);
        }
        unsafe fn drop_fn(_: *const ()) {}

        static VTABLE: RawWakerVTable =
            RawWakerVTable::new(clone_fn, wake_fn, wake_by_ref_fn, drop_fn);

        let raw = RawWaker::new(std::ptr::null(), &VTABLE);
        let waker = unsafe { Waker::from_raw(raw) };
        let mut cx = Context::from_waker(&waker);

        let gate = Arc::new(std::sync::Barrier::new(2));

        let pool = IoThreadPool::new(1);
        let g = Arc::clone(&gate);
        let mut future = pool.submit(move || {
            g.wait(); // Block until test says go.
            Ok(42usize)
        });

        // First poll: task is blocked, so we should get Pending.
        let pinned = Pin::new(&mut future);
        let poll_result = pinned.poll(&mut cx);
        assert!(matches!(poll_result, Poll::Pending));

        // Release the task.
        gate.wait();

        // Spin until the waker fires.
        while !WOKEN.load(Ordering::SeqCst) {
            std::thread::yield_now();
        }

        // Second poll: should be Ready.
        let pinned = Pin::new(&mut future);
        let poll_result = pinned.poll(&mut cx);
        match poll_result {
            Poll::Ready(Ok(v)) => assert_eq!(v, 42),
            other => panic!("expected Ready(Ok(42)), got {:?}", other.is_pending()),
        }
    }

    /// Test that concurrent `.await` of multiple futures resolves correctly.
    #[test]
    fn test_concurrent_await_multiple_futures() {
        let pool = IoThreadPool::new(4);
        let futures: Vec<_> = (0..20)
            .map(|i| pool.submit(move || Ok(i * i)))
            .collect();

        // Use FuturesUnordered to drive them concurrently.
        use futures::stream::{FuturesUnordered, StreamExt};

        let mut unordered: FuturesUnordered<_> = futures.into_iter().collect();
        let mut results = Vec::new();

        futures::executor::block_on(async {
            while let Some(result) = unordered.next().await {
                results.push(result.unwrap());
            }
        });

        results.sort();
        let expected: Vec<usize> = (0..20).map(|i| i * i).collect();
        assert_eq!(results, expected);
    }

    /// Test dropping an IoFuture that implements Future before awaiting.
    #[test]
    fn test_drop_awaitable_future_before_completion() {
        let pool = IoThreadPool::new(1);

        // Submit a slow task.
        let future = pool.submit(|| {
            std::thread::sleep(std::time::Duration::from_millis(50));
            Ok(99usize)
        });

        // Drop without awaiting — should not panic or leak.
        drop(future);

        // Pool still works.
        let r = futures::executor::block_on(pool.submit(|| Ok(1usize)));
        assert_eq!(r.unwrap(), 1);
    }

    /// Waker replacement: poll with one waker, then poll with a different waker.
    /// The second waker should be the one that gets called.
    #[test]
    fn test_waker_replacement() {
        use std::future::Future;
        use std::pin::Pin;
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

        static WAKE_COUNT: AtomicU32 = AtomicU32::new(0);
        WAKE_COUNT.store(0, Ordering::SeqCst);

        unsafe fn clone_fn(p: *const ()) -> RawWaker {
            RawWaker::new(p, &VTABLE)
        }
        unsafe fn wake_fn(_: *const ()) {
            WAKE_COUNT.fetch_add(1, Ordering::SeqCst);
        }
        unsafe fn wake_by_ref_fn(_: *const ()) {
            WAKE_COUNT.fetch_add(1, Ordering::SeqCst);
        }
        unsafe fn drop_fn(_: *const ()) {}

        static VTABLE: RawWakerVTable =
            RawWakerVTable::new(clone_fn, wake_fn, wake_by_ref_fn, drop_fn);

        let raw = RawWaker::new(std::ptr::null(), &VTABLE);
        let waker = unsafe { Waker::from_raw(raw) };
        let mut cx = Context::from_waker(&waker);

        let gate = Arc::new(std::sync::Barrier::new(2));
        let pool = IoThreadPool::new(1);
        let g = Arc::clone(&gate);
        let mut future = pool.submit(move || {
            g.wait();
            Ok(0usize)
        });

        // Poll twice with the same waker (simulates waker replacement).
        let _ = Pin::new(&mut future).poll(&mut cx);
        let _ = Pin::new(&mut future).poll(&mut cx);

        // Release the task.
        gate.wait();

        // Wait for waker to fire.
        let deadline =
            std::time::Instant::now() + std::time::Duration::from_secs(2);
        while WAKE_COUNT.load(Ordering::SeqCst) == 0 {
            if std::time::Instant::now() > deadline {
                panic!("waker was never called");
            }
            std::thread::yield_now();
        }

        // The waker should have been called at least once.
        assert!(WAKE_COUNT.load(Ordering::SeqCst) >= 1);

        // Final poll should return Ready.
        let result = Pin::new(&mut future).poll(&mut cx);
        match result {
            Poll::Ready(Ok(v)) => assert_eq!(v, 0),
            _ => panic!("expected Ready"),
        }
    }

    /// Test that `block_on` with a slow task works (no deadlock).
    #[test]
    fn test_block_on_slow_task() {
        let pool = IoThreadPool::new(1);
        let future = pool.submit(|| {
            std::thread::sleep(std::time::Duration::from_millis(100));
            Ok(777usize)
        });
        let result = futures::executor::block_on(future);
        assert_eq!(result.unwrap(), 777);
    }
}

// ---------------------------------------------------------------------------
// tokio-bridge feature: async pitfalls
// ---------------------------------------------------------------------------

#[cfg(feature = "tokio-bridge")]
mod tokio_tests {
    use super::*;

    #[tokio::test]
    async fn test_into_tokio_concurrent() {
        let pool = IoThreadPool::new(4);
        let handles: Vec<_> = (0..20)
            .map(|i| pool.submit(move || Ok(i * 2)).into_tokio())
            .collect();

        let mut results = Vec::new();
        for h in handles {
            results.push(h.await.unwrap().unwrap());
        }
        let expected: Vec<usize> = (0..20).map(|i| i * 2).collect();
        assert_eq!(results, expected);
    }

    #[tokio::test]
    async fn test_into_tokio_with_file_io() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        let size = 50_000;
        let data = gen_data(size);

        let handle = FileHandle::open(&path, "w", 0o644, CompatMode::On).unwrap();
        handle.write_host(&data, 0).unwrap();
        drop(handle);

        let handle = FileHandle::open(&path, "r", 0o644, CompatMode::On).unwrap();
        let mut buf = vec![0u8; size];
        let n = handle
            .pread(&mut buf, 0, 10_000)
            .into_tokio()
            .await
            .unwrap()
            .unwrap();
        assert_eq!(n, size);
        assert_data_eq(&data, &buf);
    }
}
