//! Bounce buffer pool for staging I/O transfers.
//!
//! Provides page-aligned (and optionally CUDA-pinned) host memory buffers
//! for use as intermediaries in Direct I/O and device memory transfers.
//!
//! # Allocator Strategies
//!
//! | Allocator | Backing | Use case |
//! |-----------|---------|----------|
//! | [`PageAligned`] | `posix_memalign` | Host-only Direct I/O |
//! | [`CudaPinned`] | CUDA `cudaMallocHost` | Device I/O without Direct I/O |
//! | [`CudaPageAlignedPinned`] | `posix_memalign` + `cudaHostRegister` | Device I/O with Direct I/O |
//!
//! # Pool Design
//!
//! - Thread-safe LIFO stack.
//! - RAII [`BounceBuffer`] guard that returns the buffer to the pool on drop.
//! - Configurable buffer size via [`Config::bounce_buffer_size`](crate::Config).
//! - Intentionally leaks on process exit (avoids CUDA teardown ordering issues).

use std::sync::Mutex;

use crate::align::page_size;

/// Trait for bounce buffer memory allocation strategies.
///
/// # Safety
///
/// Implementors must ensure that:
/// - `alloc` returns a valid, writeable pointer aligned to the allocator's requirements.
/// - `dealloc` correctly frees memory allocated by `alloc`.
/// - The returned pointer remains valid until `dealloc` is called.
pub unsafe trait Allocator: Send + Sync {
    /// Allocate a buffer of the given size.
    ///
    /// Returns a pointer to the allocated memory, or `None` on failure.
    fn alloc(&self, size: usize) -> Option<*mut u8>;

    /// Deallocate a buffer previously returned by `alloc`.
    ///
    /// # Safety
    ///
    /// `ptr` must have been returned by a prior call to `alloc` on this allocator,
    /// and must not have been deallocated already.
    unsafe fn dealloc(&self, ptr: *mut u8, size: usize);
}

/// Page-aligned allocator using `posix_memalign`.
///
/// Suitable for host-only Direct I/O.
pub struct PageAligned;

// SAFETY: PageAligned uses posix_memalign which returns page-aligned memory,
// and free() to deallocate. Both are well-defined for the sizes we use.
unsafe impl Allocator for PageAligned {
    fn alloc(&self, size: usize) -> Option<*mut u8> {
        let ps = page_size();
        let mut ptr: *mut libc::c_void = std::ptr::null_mut();
        // SAFETY: ps is a valid alignment (power of 2, multiple of sizeof(void*)),
        // size > 0 for our usage.
        let ret = unsafe { libc::posix_memalign(&mut ptr, ps, size) };
        if ret != 0 { None } else { Some(ptr as *mut u8) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, _size: usize) {
        // SAFETY: ptr was allocated by posix_memalign.
        unsafe { libc::free(ptr as *mut libc::c_void) };
    }
}

// SAFETY: PageAligned contains no state and its methods use thread-safe system calls.
unsafe impl Send for PageAligned {}
unsafe impl Sync for PageAligned {}

/// Thread-safe bounce buffer pool.
///
/// Manages a LIFO stack of pre-allocated buffers that are returned to the pool
/// when dropped via the [`BounceBuffer`] RAII guard.
pub struct BounceBufferPool<A: Allocator> {
    allocator: A,
    free_buffers: Mutex<Vec<*mut u8>>,
    buffer_size: Mutex<usize>,
}

// SAFETY: The pool's internal pointers are protected by a Mutex and only accessed
// through synchronized methods.
unsafe impl<A: Allocator> Send for BounceBufferPool<A> {}
unsafe impl<A: Allocator> Sync for BounceBufferPool<A> {}

impl<A: Allocator> BounceBufferPool<A> {
    /// Create a new pool with the given allocator and initial buffer size.
    pub fn new(allocator: A, buffer_size: usize) -> Self {
        Self {
            allocator,
            free_buffers: Mutex::new(Vec::new()),
            buffer_size: Mutex::new(buffer_size),
        }
    }

    /// Get a buffer from the pool, allocating a new one if the pool is empty.
    ///
    /// If the configured buffer size has changed since the pool was last used,
    /// all existing buffers are deallocated and new ones are created at the
    /// new size.
    pub fn get(&self) -> BounceBuffer<'_, A> {
        let current_size = *self.buffer_size.lock().expect("buffer_size lock poisoned");

        let mut free = self
            .free_buffers
            .lock()
            .expect("free_buffers lock poisoned");
        if let Some(ptr) = free.pop() {
            BounceBuffer {
                pool: self,
                ptr,
                size: current_size,
            }
        } else {
            drop(free);
            let ptr = self
                .allocator
                .alloc(current_size)
                .expect("bounce buffer allocation failed");
            BounceBuffer {
                pool: self,
                ptr,
                size: current_size,
            }
        }
    }

    /// Return a buffer to the pool.
    ///
    /// If the buffer's size doesn't match the current pool size (because the
    /// size was changed while the buffer was outstanding), the buffer is
    /// deallocated instead of being returned.
    fn put(&self, ptr: *mut u8, size: usize) {
        let current_size = *self.buffer_size.lock().expect("buffer_size lock poisoned");

        if size == current_size {
            let mut free = self
                .free_buffers
                .lock()
                .expect("free_buffers lock poisoned");
            free.push(ptr);
        } else {
            // Size mismatch: deallocate instead of returning to pool.
            // SAFETY: ptr was allocated by our allocator with the given size.
            unsafe {
                self.allocator.dealloc(ptr, size);
            }
        }
    }

    /// Returns the number of free buffers currently in the pool.
    pub fn num_free_buffers(&self) -> usize {
        self.free_buffers
            .lock()
            .expect("free_buffers lock poisoned")
            .len()
    }

    /// Returns the current buffer size.
    pub fn buffer_size(&self) -> usize {
        *self.buffer_size.lock().expect("buffer_size lock poisoned")
    }

    /// Change the buffer size.
    ///
    /// Existing buffers in the pool are deallocated. Buffers currently
    /// checked out will be deallocated when they are returned (size mismatch).
    pub fn set_buffer_size(&self, new_size: usize) {
        let mut size = self.buffer_size.lock().expect("buffer_size lock poisoned");
        if *size == new_size {
            return;
        }
        *size = new_size;
        drop(size);

        self.clear();
    }

    /// Deallocate all free buffers in the pool.
    pub fn clear(&self) {
        let current_size = *self.buffer_size.lock().expect("buffer_size lock poisoned");
        let mut free = self
            .free_buffers
            .lock()
            .expect("free_buffers lock poisoned");
        for ptr in free.drain(..) {
            // SAFETY: these pointers were allocated by our allocator.
            unsafe {
                self.allocator.dealloc(ptr, current_size);
            }
        }
    }
}

// Intentionally no Drop impl — we leak remaining buffers on process exit
// to avoid CUDA teardown ordering issues (same as C++ kvikio).

/// RAII guard for a bounce buffer.
///
/// Returns the buffer to its pool when dropped.
pub struct BounceBuffer<'a, A: Allocator> {
    pool: &'a BounceBufferPool<A>,
    ptr: *mut u8,
    size: usize,
}

impl<A: Allocator> BounceBuffer<'_, A> {
    /// Get a mutable slice view of the buffer.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: ptr is valid for size bytes and we have exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size) }
    }

    /// Get a slice view of the buffer.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr is valid for size bytes.
        unsafe { std::slice::from_raw_parts(self.ptr, self.size) }
    }

    /// Get the raw pointer to the buffer.
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// Get a mutable raw pointer to the buffer.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    /// Get the buffer size.
    pub fn size(&self) -> usize {
        self.size
    }
}

impl<A: Allocator> Drop for BounceBuffer<'_, A> {
    fn drop(&mut self) {
        self.pool.put(self.ptr, self.size);
    }
}

// BounceBuffer holds a raw pointer but access is synchronized through the pool.
// SAFETY: The underlying pointer is not shared until returned to the pool.
unsafe impl<A: Allocator> Send for BounceBuffer<'_, A> {}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- PageAligned allocator tests ----

    #[test]
    fn test_page_aligned_alloc_dealloc() {
        let alloc = PageAligned;
        let ps = page_size();
        let ptr = alloc.alloc(ps).unwrap();
        assert!(!ptr.is_null());
        assert!(crate::align::is_aligned_ptr(ptr, ps));
        // SAFETY: ptr was allocated by alloc.
        unsafe { alloc.dealloc(ptr, ps) };
    }

    #[test]
    fn test_page_aligned_alloc_large() {
        let alloc = PageAligned;
        let ps = page_size();
        let size = ps * 4;
        let ptr = alloc.alloc(size).unwrap();
        assert!(crate::align::is_aligned_ptr(ptr, ps));
        // Write to verify the memory is usable.
        unsafe {
            std::ptr::write_bytes(ptr, 0xAB, size);
        }
        unsafe { alloc.dealloc(ptr, size) };
    }

    // ---- BounceBufferPool tests (mirroring C++ test_bounce_buffer.cpp) ----

    #[test]
    fn test_buffers_returned_to_pool() {
        let pool = BounceBufferPool::new(PageAligned, 4096);
        assert_eq!(pool.num_free_buffers(), 0);

        {
            let _buf = pool.get();
            assert_eq!(pool.num_free_buffers(), 0);
        }
        // Buffer should be returned on drop.
        assert_eq!(pool.num_free_buffers(), 1);

        {
            let _buf1 = pool.get();
            assert_eq!(pool.num_free_buffers(), 0);
            let _buf2 = pool.get();
            assert_eq!(pool.num_free_buffers(), 0);
        }
        assert_eq!(pool.num_free_buffers(), 2);
    }

    #[test]
    fn test_pool_reuses_buffers() {
        let pool = BounceBufferPool::new(PageAligned, 4096);

        let ptr1;
        {
            let buf = pool.get();
            ptr1 = buf.as_ptr();
        }
        // Get again — should reuse the same buffer (LIFO).
        let buf = pool.get();
        assert_eq!(buf.as_ptr(), ptr1);
    }

    #[test]
    fn test_buffer_size_changes_clears_pool() {
        let pool = BounceBufferPool::new(PageAligned, 4096);

        // Get and return a buffer.
        {
            let _buf = pool.get();
        }
        assert_eq!(pool.num_free_buffers(), 1);
        assert_eq!(pool.buffer_size(), 4096);

        // Change size — pool should be cleared.
        pool.set_buffer_size(8192);
        assert_eq!(pool.num_free_buffers(), 0);
        assert_eq!(pool.buffer_size(), 8192);

        // New buffer should have the new size.
        let buf = pool.get();
        assert_eq!(buf.size(), 8192);
    }

    #[test]
    fn test_old_size_buffer_deallocated_not_returned() {
        let pool = BounceBufferPool::new(PageAligned, 4096);

        // Get a buffer at old size.
        let buf = pool.get();
        assert_eq!(buf.size(), 4096);

        // Change size while buffer is outstanding.
        pool.set_buffer_size(8192);

        // Drop the old-size buffer — it should be deallocated, not returned.
        drop(buf);
        assert_eq!(pool.num_free_buffers(), 0);
    }

    #[test]
    fn test_pool_clear() {
        let pool = BounceBufferPool::new(PageAligned, 4096);

        {
            let _b1 = pool.get();
            let _b2 = pool.get();
            let _b3 = pool.get();
        }
        assert_eq!(pool.num_free_buffers(), 3);

        pool.clear();
        assert_eq!(pool.num_free_buffers(), 0);
    }

    #[test]
    fn test_bounce_buffer_read_write() {
        let pool = BounceBufferPool::new(PageAligned, 4096);
        let mut buf = pool.get();

        let slice = buf.as_mut_slice();
        assert_eq!(slice.len(), 4096);

        // Write a pattern.
        for (i, byte) in slice.iter_mut().enumerate() {
            *byte = (i % 256) as u8;
        }

        // Read it back.
        let slice = buf.as_slice();
        for (i, byte) in slice.iter().enumerate() {
            assert_eq!(*byte, (i % 256) as u8);
        }
    }

    #[test]
    fn test_set_same_size_is_noop() {
        let pool = BounceBufferPool::new(PageAligned, 4096);
        {
            let _buf = pool.get();
        }
        assert_eq!(pool.num_free_buffers(), 1);

        pool.set_buffer_size(4096);
        // Pool should NOT be cleared when size is unchanged.
        assert_eq!(pool.num_free_buffers(), 1);
    }

    #[test]
    fn test_multiple_get_returns() {
        let pool = BounceBufferPool::new(PageAligned, 4096);

        let b1 = pool.get();
        let b2 = pool.get();
        let b3 = pool.get();
        assert_eq!(pool.num_free_buffers(), 0);

        drop(b1);
        assert_eq!(pool.num_free_buffers(), 1);
        drop(b2);
        assert_eq!(pool.num_free_buffers(), 2);
        drop(b3);
        assert_eq!(pool.num_free_buffers(), 3);
    }
}
