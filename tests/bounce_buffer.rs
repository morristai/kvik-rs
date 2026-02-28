//! Bounce buffer pool integration tests.
//!
//! Mirrors C++ kvikio's `test_bounce_buffer.cpp` with end-to-end scenarios
//! including concurrent access and actual file I/O using bounce buffers.

mod test_utils;

use kvik_rs::align::{is_aligned_ptr, page_size};
use kvik_rs::bounce_buffer::{BounceBufferPool, PageAligned};
use kvik_rs::{CompatMode, FileHandle};

use test_utils::{assert_data_eq, gen_data};

// ---- Pool under concurrent load ----

#[test]
fn test_pool_under_concurrent_load() {
    let pool = BounceBufferPool::new(PageAligned, 4096);

    std::thread::scope(|scope| {
        // Spawn multiple threads, each getting and returning buffers.
        for _ in 0..8 {
            scope.spawn(|| {
                for _ in 0..50 {
                    let mut buf = pool.get();
                    // Write to the buffer to ensure it's usable.
                    let slice = buf.as_mut_slice();
                    slice[0] = 0xAA;
                    slice[slice.len() - 1] = 0xBB;
                    // Buffer returned on drop.
                }
            });
        }
    });

    // All 8 threads * 50 iterations = 400 get/put cycles.
    // Pool should have some buffers back (exact count depends on timing).
    let free = pool.num_free_buffers();
    assert!(
        free > 0,
        "expected some free buffers after concurrent load, got {free}"
    );
}

// ---- Pool with actual file I/O ----

#[test]
fn test_pool_with_io() {
    let pool = BounceBufferPool::new(PageAligned, 8192);
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path();

    let data = gen_data(8192);

    // Use a bounce buffer as the intermediate for writing.
    {
        let mut bounce = pool.get();
        let ps = page_size();
        assert!(is_aligned_ptr(bounce.as_ptr(), ps));

        let slice = bounce.as_mut_slice();
        slice[..data.len()].copy_from_slice(&data);
    }

    // Actually write the data through FileHandle.
    let handle = FileHandle::open(path, "w", 0o644, CompatMode::On).unwrap();
    handle.write_host(&data, 0).unwrap();
    drop(handle);

    // Read back using a bounce buffer as the destination.
    {
        let mut bounce = pool.get();
        let handle = FileHandle::open(path, "r", 0o644, CompatMode::On).unwrap();
        let n = handle.read_host(bounce.as_mut_slice(), 0).unwrap();
        assert_eq!(n, data.len());
        assert_data_eq(&data, &bounce.as_slice()[..n]);
    }

    // After all buffers are returned.
    assert!(pool.num_free_buffers() >= 1);
}

// ---- Pool size change during outstanding buffers ----

#[test]
fn test_pool_size_change_during_io() {
    let pool = BounceBufferPool::new(PageAligned, 4096);

    // Get a buffer at the original size.
    let buf1 = pool.get();
    assert_eq!(buf1.size(), 4096);

    // Change the pool size while buf1 is outstanding.
    pool.set_buffer_size(8192);
    assert_eq!(pool.buffer_size(), 8192);

    // Get a new buffer — should be at the new size.
    let buf2 = pool.get();
    assert_eq!(buf2.size(), 8192);

    // Drop buf1 (old size) — should be deallocated, not returned.
    drop(buf1);
    let free = pool.num_free_buffers();
    assert_eq!(free, 0, "old-size buffer should not be returned to pool");

    // Drop buf2 (current size) — should be returned.
    drop(buf2);
    assert_eq!(pool.num_free_buffers(), 1);
}

// ---- RAII lifecycle: buffers returned to pool ----

#[test]
fn test_buffers_returned_to_pool() {
    let pool = BounceBufferPool::new(PageAligned, 4096);
    assert_eq!(pool.num_free_buffers(), 0);

    {
        let _b1 = pool.get();
        let _b2 = pool.get();
        assert_eq!(pool.num_free_buffers(), 0);
    }
    // Both returned.
    assert_eq!(pool.num_free_buffers(), 2);

    // Get them back (reuse).
    {
        let _b1 = pool.get();
        assert_eq!(pool.num_free_buffers(), 1);
        let _b2 = pool.get();
        assert_eq!(pool.num_free_buffers(), 0);
    }
    assert_eq!(pool.num_free_buffers(), 2);
}

// ---- Pool reuse verifies same pointer (LIFO) ----

#[test]
fn test_pool_lifo_reuse() {
    let pool = BounceBufferPool::new(PageAligned, 4096);

    let ptr1;
    {
        let buf = pool.get();
        ptr1 = buf.as_ptr();
    }

    // Get again — should reuse the same buffer (LIFO).
    let buf = pool.get();
    assert_eq!(buf.as_ptr(), ptr1, "expected LIFO reuse of buffer pointer");
}

// ---- Buffer content survives pool round-trip ----

#[test]
fn test_buffer_content_survives_roundtrip() {
    let pool = BounceBufferPool::new(PageAligned, 256);

    let ptr;
    {
        let mut buf = pool.get();
        ptr = buf.as_mut_ptr();
        // Write a pattern.
        for (i, byte) in buf.as_mut_slice().iter_mut().enumerate() {
            *byte = (i % 256) as u8;
        }
    }

    // Get the buffer back (same pointer due to LIFO).
    let buf = pool.get();
    assert_eq!(buf.as_ptr(), ptr as *const u8);
    // Content should still be there (pool doesn't zero on return).
    for (i, byte) in buf.as_slice().iter().enumerate() {
        assert_eq!(*byte, (i % 256) as u8, "content mismatch at byte {i}");
    }
}

// ---- Alignment guarantee ----

#[test]
fn test_bounce_buffer_alignment() {
    let pool = BounceBufferPool::new(PageAligned, 4096);
    let ps = page_size();

    for _ in 0..10 {
        let buf = pool.get();
        assert!(
            is_aligned_ptr(buf.as_ptr(), ps),
            "bounce buffer not page-aligned: {:?}",
            buf.as_ptr()
        );
    }
}

// ---- Multiple concurrent pools ----

#[test]
fn test_multiple_pools_independent() {
    let pool_small = BounceBufferPool::new(PageAligned, 1024);
    let pool_large = BounceBufferPool::new(PageAligned, 8192);

    let buf_small = pool_small.get();
    let buf_large = pool_large.get();

    assert_eq!(buf_small.size(), 1024);
    assert_eq!(buf_large.size(), 8192);

    drop(buf_small);
    drop(buf_large);

    assert_eq!(pool_small.num_free_buffers(), 1);
    assert_eq!(pool_large.num_free_buffers(), 1);

    // Changing one pool's size should not affect the other.
    pool_small.set_buffer_size(2048);
    assert_eq!(pool_small.num_free_buffers(), 0); // Cleared.
    assert_eq!(pool_large.num_free_buffers(), 1); // Unchanged.
}

// ---- Clear operation ----

#[test]
fn test_pool_clear() {
    let pool = BounceBufferPool::new(PageAligned, 4096);

    // Fill pool.
    {
        let _b1 = pool.get();
        let _b2 = pool.get();
        let _b3 = pool.get();
    }
    assert_eq!(pool.num_free_buffers(), 3);

    pool.clear();
    assert_eq!(pool.num_free_buffers(), 0);

    // Can still allocate after clear.
    let buf = pool.get();
    assert_eq!(buf.size(), 4096);
}
