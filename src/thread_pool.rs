//! Thread pool and zero-dependency future for parallel I/O.
//!
//! Provides [`IoThreadPool`] (a minimal persistent thread pool) and [`IoFuture`]
//! (a future that can be polled or blocked on without any async runtime).
//!
//! # Async runtime integration
//!
//! - With the `futures` feature, `IoFuture<T>` implements `std::future::Future`.
//! - With the `tokio-bridge` feature, `IoFuture<T>` gains an `into_tokio()` method.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use crate::config::Config;
use crate::error::Result;

// ---------------------------------------------------------------------------
// Internal oneshot channel
// ---------------------------------------------------------------------------

/// Shared state for the oneshot channel between sender and receiver.
struct Shared<T> {
    value: Mutex<Option<T>>,
    ready: AtomicBool,
    condvar: Condvar,
    #[cfg(feature = "futures")]
    waker: Mutex<Option<std::task::Waker>>,
}

/// Sending half of the oneshot channel.
struct Sender<T> {
    shared: Arc<Shared<T>>,
}

/// Receiving half of the oneshot channel.
struct Receiver<T> {
    shared: Arc<Shared<T>>,
}

fn oneshot<T>() -> (Sender<T>, Receiver<T>) {
    let shared = Arc::new(Shared {
        value: Mutex::new(None),
        ready: AtomicBool::new(false),
        condvar: Condvar::new(),
        #[cfg(feature = "futures")]
        waker: Mutex::new(None),
    });
    (
        Sender {
            shared: Arc::clone(&shared),
        },
        Receiver { shared },
    )
}

impl<T> Sender<T> {
    fn send(self, value: T) {
        {
            let mut slot = self.shared.value.lock().expect("oneshot lock poisoned");
            *slot = Some(value);
        }
        self.shared.ready.store(true, Ordering::Release);
        self.shared.condvar.notify_one();

        #[cfg(feature = "futures")]
        {
            if let Some(waker) = self.shared.waker.lock().expect("waker lock poisoned").take() {
                waker.wake();
            }
        }
    }
}

impl<T> Receiver<T> {
    /// Block until the value is available.
    fn recv(self) -> T {
        let mut slot = self.shared.value.lock().expect("oneshot lock poisoned");
        while slot.is_none() {
            slot = self.shared.condvar.wait(slot).expect("oneshot lock poisoned");
        }
        slot.take().expect("value consumed twice")
    }

    /// Non-blocking attempt to receive.
    fn try_recv(&self) -> Option<T> {
        if !self.shared.ready.load(Ordering::Acquire) {
            return None;
        }
        let mut slot = self.shared.value.lock().expect("oneshot lock poisoned");
        slot.take()
    }

    fn is_ready(&self) -> bool {
        self.shared.ready.load(Ordering::Acquire)
    }
}

// ---------------------------------------------------------------------------
// IoFuture<T>
// ---------------------------------------------------------------------------

/// A future representing a value being computed on the I/O thread pool.
///
/// Can be used without any async runtime:
/// - [`get()`](Self::get) blocks the calling thread until the result is ready.
/// - [`try_get()`](Self::try_get) polls without blocking.
/// - [`is_ready()`](Self::is_ready) checks completion status.
///
/// With the `futures` feature enabled, `IoFuture<T>` implements
/// [`std::future::Future`], making it `.await`-able from any async runtime.
///
/// With the `tokio-bridge` feature, [`into_tokio()`](Self::into_tokio) bridges
/// to a `tokio::task::JoinHandle`.
pub struct IoFuture<T> {
    receiver: Option<Receiver<Result<T>>>,
    #[cfg(feature = "futures")]
    shared: Arc<Shared<Result<T>>>,
}

impl<T> IoFuture<T> {
    /// Block the calling thread until the result is ready.
    pub fn get(mut self) -> Result<T> {
        self.receiver
            .take()
            .expect("IoFuture already consumed")
            .recv()
    }

    /// Non-blocking check. Returns `Some(result)` if done, `None` if pending.
    pub fn try_get(&mut self) -> Option<Result<T>> {
        self.receiver
            .as_ref()
            .and_then(|rx| rx.try_recv())
    }

    /// Returns `true` if the result is available.
    pub fn is_ready(&self) -> bool {
        self.receiver
            .as_ref()
            .map_or(false, |rx| rx.is_ready())
    }
}

#[cfg(feature = "futures")]
impl<T> std::future::Future for IoFuture<T> {
    type Output = Result<T>;

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        // Fast path: already completed.
        if let Some(value) = self.try_get() {
            return std::task::Poll::Ready(value);
        }

        // Store waker so the sender thread can wake us.
        {
            let mut waker_slot = self
                .shared
                .waker
                .lock()
                .expect("waker lock poisoned");
            *waker_slot = Some(cx.waker().clone());
        }

        // Double-check after storing waker to avoid missed notifications.
        if let Some(value) = self.try_get() {
            std::task::Poll::Ready(value)
        } else {
            std::task::Poll::Pending
        }
    }
}

#[cfg(feature = "tokio-bridge")]
impl<T: Send + 'static> IoFuture<T> {
    /// Bridge this future into the Tokio runtime.
    ///
    /// Spawns a blocking task that calls [`get()`](Self::get) and returns
    /// the result through a `JoinHandle`.
    pub fn into_tokio(self) -> tokio::task::JoinHandle<Result<T>> {
        tokio::task::spawn_blocking(move || self.get())
    }
}

// ---------------------------------------------------------------------------
// IoThreadPool
// ---------------------------------------------------------------------------

type Task = Box<dyn FnOnce() + Send>;

/// A minimal persistent thread pool for I/O parallelism.
///
/// Workers consume tasks from a shared MPSC queue. The pool is sized from
/// [`Config::num_threads`](crate::Config) (env: `KVIKIO_NTHREADS`, default: 1).
///
/// The global pool is lazily initialized via [`global_thread_pool()`].
pub struct IoThreadPool {
    sender: Mutex<Option<std::sync::mpsc::Sender<Task>>>,
    workers: Mutex<Vec<std::thread::JoinHandle<()>>>,
    num_threads: usize,
}

impl IoThreadPool {
    /// Create a new thread pool with the given number of worker threads.
    ///
    /// # Panics
    ///
    /// Panics if `num_threads` is 0.
    pub fn new(num_threads: usize) -> Self {
        assert!(num_threads > 0, "thread pool requires at least 1 thread");

        let (tx, rx) = std::sync::mpsc::channel::<Task>();
        let rx = Arc::new(Mutex::new(rx));

        let mut workers = Vec::with_capacity(num_threads);
        for _ in 0..num_threads {
            let rx = Arc::clone(&rx);
            workers.push(std::thread::spawn(move || {
                loop {
                    let task = {
                        let rx = rx.lock().expect("thread pool rx poisoned");
                        rx.recv()
                    };
                    match task {
                        Ok(task) => task(),
                        Err(_) => break, // channel closed, shut down
                    }
                }
            }));
        }

        Self {
            sender: Mutex::new(Some(tx)),
            workers: Mutex::new(workers),
            num_threads,
        }
    }

    /// Submit a task to the thread pool, returning an `IoFuture` for its result.
    pub fn submit<F, T>(&self, f: F) -> IoFuture<T>
    where
        F: FnOnce() -> Result<T> + Send + 'static,
        T: Send + 'static,
    {
        let (tx, rx) = oneshot();

        let task: Task = Box::new(move || {
            tx.send(f());
        });

        let sender = self.sender.lock().expect("thread pool sender poisoned");
        sender
            .as_ref()
            .expect("thread pool is shut down")
            .send(task)
            .expect("thread pool workers have disconnected");

        IoFuture {
            #[cfg(feature = "futures")]
            shared: Arc::clone(&rx.shared),
            receiver: Some(rx),
        }
    }

    /// Returns the number of worker threads.
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    /// Shut down the pool, waiting for all in-flight tasks to complete.
    ///
    /// After shutdown, `submit()` will panic. Multiple shutdowns are safe.
    pub fn shutdown(&self) {
        // Drop the sender to signal workers to exit.
        // Use `lock()` with poison recovery — the mutex may be poisoned if a
        // previous operation panicked (e.g., submit after shutdown in a test).
        {
            let mut sender = self
                .sender
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            sender.take();
        }
        // Join all workers.
        let mut workers = self
            .workers
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for handle in workers.drain(..) {
            handle.join().ok();
        }
    }
}

impl Drop for IoThreadPool {
    fn drop(&mut self) {
        self.shutdown();
    }
}

// ---------------------------------------------------------------------------
// Global thread pool singleton
// ---------------------------------------------------------------------------

static GLOBAL_THREAD_POOL: OnceLock<IoThreadPool> = OnceLock::new();

/// Returns a reference to the global I/O thread pool.
///
/// Lazily initialized with `Config::num_threads` worker threads.
pub fn global_thread_pool() -> &'static IoThreadPool {
    GLOBAL_THREAD_POOL.get_or_init(|| {
        let num_threads = Config::get().num_threads.max(1);
        IoThreadPool::new(num_threads)
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::{Error, ErrorKind};

    // ---- Oneshot channel tests ----

    #[test]
    fn test_oneshot_send_recv() {
        let (tx, rx) = oneshot::<i32>();
        tx.send(42);
        assert_eq!(rx.recv(), 42);
    }

    #[test]
    fn test_oneshot_send_recv_cross_thread() {
        let (tx, rx) = oneshot::<String>();
        std::thread::spawn(move || {
            tx.send("hello from thread".to_string());
        });
        assert_eq!(rx.recv(), "hello from thread");
    }

    #[test]
    fn test_oneshot_try_recv_before_send() {
        let (_tx, rx) = oneshot::<i32>();
        assert!(rx.try_recv().is_none());
    }

    #[test]
    fn test_oneshot_try_recv_after_send() {
        let (tx, rx) = oneshot::<i32>();
        tx.send(99);
        assert_eq!(rx.try_recv(), Some(99));
    }

    #[test]
    fn test_oneshot_is_ready() {
        let (tx, rx) = oneshot::<i32>();
        assert!(!rx.is_ready());
        tx.send(1);
        assert!(rx.is_ready());
    }

    // ---- IoFuture tests ----

    #[test]
    fn test_io_future_get() {
        let (tx, rx) = oneshot();
        let future = IoFuture::<usize> {
            #[cfg(feature = "futures")]
            shared: Arc::clone(&rx.shared),
            receiver: Some(rx),
        };

        std::thread::spawn(move || {
            tx.send(Ok(42));
        });

        assert_eq!(future.get().unwrap(), 42);
    }

    #[test]
    fn test_io_future_get_error() {
        let (tx, rx) = oneshot();
        let future = IoFuture::<usize> {
            #[cfg(feature = "futures")]
            shared: Arc::clone(&rx.shared),
            receiver: Some(rx),
        };

        tx.send(Err(Error::new(ErrorKind::SystemError, "test error")));
        let err = future.get().unwrap_err();
        assert_eq!(err.kind(), ErrorKind::SystemError);
    }

    #[test]
    fn test_io_future_try_get_pending() {
        let (_tx, rx) = oneshot();
        let mut future = IoFuture::<usize> {
            #[cfg(feature = "futures")]
            shared: Arc::clone(&rx.shared),
            receiver: Some(rx),
        };

        assert!(future.try_get().is_none());
    }

    #[test]
    fn test_io_future_try_get_ready() {
        let (tx, rx) = oneshot();
        let mut future = IoFuture::<usize> {
            #[cfg(feature = "futures")]
            shared: Arc::clone(&rx.shared),
            receiver: Some(rx),
        };

        tx.send(Ok(7));
        assert_eq!(future.try_get().unwrap().unwrap(), 7);
    }

    #[test]
    fn test_io_future_is_ready() {
        let (tx, rx) = oneshot();
        let future = IoFuture::<usize> {
            #[cfg(feature = "futures")]
            shared: Arc::clone(&rx.shared),
            receiver: Some(rx),
        };

        assert!(!future.is_ready());
        tx.send(Ok(0));
        assert!(future.is_ready());
    }

    // ---- IoThreadPool tests ----

    #[test]
    fn test_pool_submit_single_task() {
        let pool = IoThreadPool::new(1);
        let future = pool.submit(|| Ok(42usize));
        assert_eq!(future.get().unwrap(), 42);
    }

    #[test]
    fn test_pool_submit_multiple_tasks() {
        let pool = IoThreadPool::new(4);
        let futures: Vec<_> = (0..20)
            .map(|i| pool.submit(move || Ok(i * 2)))
            .collect();

        let results: Vec<usize> = futures
            .into_iter()
            .map(|f| f.get().unwrap())
            .collect();

        let expected: Vec<usize> = (0..20).map(|i| i * 2).collect();
        assert_eq!(results, expected);
    }

    #[test]
    fn test_pool_respects_thread_count() {
        let pool = IoThreadPool::new(3);
        assert_eq!(pool.num_threads(), 3);
    }

    #[test]
    #[should_panic(expected = "thread pool requires at least 1 thread")]
    fn test_pool_zero_threads_panics() {
        IoThreadPool::new(0);
    }

    #[test]
    fn test_pool_shutdown() {
        let pool = IoThreadPool::new(2);
        let future = pool.submit(|| Ok(1usize));
        assert_eq!(future.get().unwrap(), 1);
        pool.shutdown();
        // Double shutdown should not panic.
        pool.shutdown();
    }

    #[test]
    fn test_pool_error_propagation() {
        let pool = IoThreadPool::new(1);
        let future = pool.submit(|| -> Result<usize> {
            Err(Error::new(ErrorKind::SystemError, "simulated failure"))
        });
        let err = future.get().unwrap_err();
        assert_eq!(err.kind(), ErrorKind::SystemError);
        assert!(err.message().contains("simulated failure"));
    }

    #[test]
    fn test_pool_concurrent_results_are_correct() {
        let pool = IoThreadPool::new(4);
        let futures: Vec<_> = (0..100)
            .map(|i| {
                pool.submit(move || {
                    // Small computation to verify correctness
                    let sum: usize = (0..=i).sum();
                    Ok(sum)
                })
            })
            .collect();

        for (i, future) in futures.into_iter().enumerate() {
            let expected: usize = (0..=i).sum();
            assert_eq!(future.get().unwrap(), expected);
        }
    }

    #[test]
    fn test_global_thread_pool_accessible() {
        let pool = global_thread_pool();
        assert!(pool.num_threads() >= 1);
    }

    #[test]
    fn test_global_thread_pool_submit() {
        let pool = global_thread_pool();
        let future = pool.submit(|| Ok(99usize));
        assert_eq!(future.get().unwrap(), 99);
    }

    // ---- Oneshot edge cases ----

    #[test]
    fn test_oneshot_try_recv_consumed_returns_none() {
        let (tx, rx) = oneshot::<i32>();
        tx.send(42);
        assert_eq!(rx.try_recv(), Some(42));
        // Value already taken, is_ready still true but slot is empty.
        assert!(rx.is_ready());
        assert_eq!(rx.try_recv(), None);
    }

    #[test]
    fn test_oneshot_drop_sender_before_send() {
        let (tx, rx) = oneshot::<i32>();
        drop(tx);
        // Receiver still exists but no value will arrive.
        assert!(!rx.is_ready());
        assert!(rx.try_recv().is_none());
        // Dropping receiver should not panic.
        drop(rx);
    }

    #[test]
    fn test_oneshot_drop_receiver_before_send() {
        let (tx, _rx) = oneshot::<i32>();
        // Sending to a dropped receiver should not panic
        // (the value is just placed into the shared state).
        tx.send(99);
    }

    #[test]
    fn test_oneshot_large_payload() {
        let (tx, rx) = oneshot::<Vec<u8>>();
        let big = vec![0xAB; 10_000_000]; // 10 MB
        tx.send(big.clone());
        let received = rx.recv();
        assert_eq!(received.len(), 10_000_000);
        assert_eq!(received, big);
    }

    #[test]
    fn test_oneshot_with_zero_sized_type() {
        let (tx, rx) = oneshot::<()>();
        tx.send(());
        assert_eq!(rx.recv(), ());
    }

    // ---- IoFuture edge cases ----

    #[test]
    fn test_io_future_try_get_consumed_returns_none() {
        let pool = IoThreadPool::new(1);
        let mut future = pool.submit(|| Ok(42usize));
        // Wait for result.
        while !future.is_ready() {
            std::thread::yield_now();
        }
        let first = future.try_get();
        assert!(first.is_some());
        assert_eq!(first.unwrap().unwrap(), 42);
        // Second call returns None.
        assert!(future.try_get().is_none());
        // is_ready is still true (the flag is set) but value is consumed.
        assert!(future.is_ready());
    }

    #[test]
    fn test_io_future_drop_before_result_ready() {
        let pool = IoThreadPool::new(1);
        let gate = Arc::new(std::sync::Barrier::new(2));
        let g = Arc::clone(&gate);

        let future = pool.submit(move || {
            g.wait(); // Block until we release.
            Ok(99usize)
        });

        // Drop the future while the task is still blocked.
        drop(future);

        // Release the task.
        gate.wait();

        // Pool should still work.
        let result = pool.submit(|| Ok(1usize)).get().unwrap();
        assert_eq!(result, 1);
    }

    #[test]
    fn test_io_future_try_get_then_is_ready() {
        // After try_get consumes the value, is_ready() is still true
        // (the AtomicBool flag is set) but the value slot is empty.
        let (tx, rx) = oneshot();
        let mut future = IoFuture::<usize> {
            #[cfg(feature = "futures")]
            shared: Arc::clone(&rx.shared),
            receiver: Some(rx),
        };

        tx.send(Ok(10));
        assert!(future.is_ready());

        // Consume via try_get.
        let val = future.try_get().unwrap().unwrap();
        assert_eq!(val, 10);

        // is_ready is still true (flag was set), but value is gone.
        assert!(future.is_ready());
        assert!(future.try_get().is_none());
    }

    // ---- IoThreadPool edge cases ----

    #[test]
    fn test_pool_many_tasks_single_thread() {
        let pool = IoThreadPool::new(1);
        let futures: Vec<_> = (0..500)
            .map(|i| pool.submit(move || Ok(i)))
            .collect();
        for (i, f) in futures.into_iter().enumerate() {
            assert_eq!(f.get().unwrap(), i);
        }
    }

    #[test]
    fn test_pool_more_threads_than_tasks() {
        let pool = IoThreadPool::new(32);
        let futures: Vec<_> = (0..3)
            .map(|i| pool.submit(move || Ok(i * 10)))
            .collect();
        for (i, f) in futures.into_iter().enumerate() {
            assert_eq!(f.get().unwrap(), i * 10);
        }
    }

    #[test]
    fn test_pool_rapid_submit_get_cycles() {
        let pool = IoThreadPool::new(2);
        for i in 0..200 {
            let result = pool.submit(move || Ok(i)).get().unwrap();
            assert_eq!(result, i);
        }
    }

    #[test]
    #[should_panic(expected = "thread pool is shut down")]
    fn test_pool_submit_after_shutdown() {
        let pool = IoThreadPool::new(1);
        pool.shutdown();
        let _f = pool.submit(|| Ok(0usize));
    }

    #[test]
    fn test_pool_drop_with_pending_futures() {
        let futures: Vec<IoFuture<usize>>;
        {
            let pool = IoThreadPool::new(4);
            futures = (0..20)
                .map(|i| pool.submit(move || Ok(i)))
                .collect();
            // Pool dropped here → shutdown → join workers.
        }
        // All futures should be resolvable.
        for (i, f) in futures.into_iter().enumerate() {
            assert_eq!(f.get().unwrap(), i);
        }
    }

    #[test]
    fn test_pool_task_panic_other_tasks_still_work() {
        let pool = IoThreadPool::new(2);

        // Submit a panicking task — don't call get().
        let _bad = pool.submit(|| -> Result<usize> {
            panic!("boom");
        });

        // Give time for the panic to propagate.
        std::thread::sleep(std::time::Duration::from_millis(50));

        // Other tasks should still work (remaining worker picks them up).
        let r = pool.submit(|| Ok(7usize)).get().unwrap();
        assert_eq!(r, 7);
    }

    // ---- futures feature tests ----

    #[cfg(feature = "futures")]
    #[test]
    fn test_io_future_impl_future_ok() {
        let pool = IoThreadPool::new(1);
        let future = pool.submit(|| Ok(42usize));
        let result = futures::executor::block_on(future);
        assert_eq!(result.unwrap(), 42);
    }

    #[cfg(feature = "futures")]
    #[test]
    fn test_io_future_impl_future_err() {
        let pool = IoThreadPool::new(1);
        let future = pool.submit(|| -> Result<usize> {
            Err(Error::new(ErrorKind::SystemError, "async error"))
        });
        let result = futures::executor::block_on(future);
        assert_eq!(result.unwrap_err().kind(), ErrorKind::SystemError);
    }

    #[cfg(feature = "futures")]
    #[test]
    fn test_io_future_impl_future_multiple() {
        let pool = IoThreadPool::new(4);
        let futures: Vec<_> = (0..10)
            .map(|i| pool.submit(move || Ok(i * 3usize)))
            .collect();

        for (i, future) in futures.into_iter().enumerate() {
            let result = futures::executor::block_on(future);
            assert_eq!(result.unwrap(), i * 3);
        }
    }

    // ---- tokio-bridge feature tests ----

    #[cfg(feature = "tokio-bridge")]
    #[tokio::test]
    async fn test_io_future_into_tokio() {
        let pool = IoThreadPool::new(1);
        let future = pool.submit(|| Ok(42usize));
        let result = future.into_tokio().await.unwrap();
        assert_eq!(result.unwrap(), 42);
    }

    #[cfg(feature = "tokio-bridge")]
    #[tokio::test]
    async fn test_io_future_into_tokio_err() {
        let pool = IoThreadPool::new(1);
        let future = pool.submit(|| -> Result<usize> {
            Err(Error::new(ErrorKind::SystemError, "tokio error"))
        });
        let result = future.into_tokio().await.unwrap();
        assert_eq!(result.unwrap_err().kind(), ErrorKind::SystemError);
    }
}
