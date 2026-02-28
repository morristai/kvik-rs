//! Alignment utilities for Direct I/O operations.
//!
//! Direct I/O requires page-aligned offsets and sizes. These helpers perform
//! the necessary alignment arithmetic using bit operations (alignment must be
//! a power of two).

/// Returns the page size of the system (typically 4096).
pub fn page_size() -> usize {
    // SAFETY: _SC_PAGESIZE is always valid on Linux.
    let ps = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    debug_assert!(ps > 0, "sysconf(_SC_PAGESIZE) returned {ps}");
    ps as usize
}

/// Round `value` up to the next multiple of `alignment`.
///
/// `alignment` must be a power of two.
///
/// # Examples
///
/// ```
/// use kvik_rs::align::align_up;
/// assert_eq!(align_up(4095, 4096), 4096);
/// assert_eq!(align_up(4096, 4096), 4096);
/// assert_eq!(align_up(4097, 4096), 8192);
/// ```
#[inline]
pub fn align_up(value: usize, alignment: usize) -> usize {
    debug_assert!(alignment.is_power_of_two(), "alignment must be a power of two");
    (value + alignment - 1) & !(alignment - 1)
}

/// Round `value` down to the previous multiple of `alignment`.
///
/// `alignment` must be a power of two.
///
/// # Examples
///
/// ```
/// use kvik_rs::align::align_down;
/// assert_eq!(align_down(4095, 4096), 0);
/// assert_eq!(align_down(4096, 4096), 4096);
/// assert_eq!(align_down(4097, 4096), 4096);
/// ```
#[inline]
pub fn align_down(value: usize, alignment: usize) -> usize {
    debug_assert!(alignment.is_power_of_two(), "alignment must be a power of two");
    value & !(alignment - 1)
}

/// Check if `value` is aligned to `alignment`.
///
/// `alignment` must be a power of two.
///
/// # Examples
///
/// ```
/// use kvik_rs::align::is_aligned;
/// assert!(is_aligned(4096, 4096));
/// assert!(!is_aligned(4095, 4096));
/// ```
#[inline]
pub fn is_aligned(value: usize, alignment: usize) -> bool {
    debug_assert!(alignment.is_power_of_two(), "alignment must be a power of two");
    (value & (alignment - 1)) == 0
}

/// Round a pointer address up to the next multiple of `alignment`.
///
/// `alignment` must be a power of two.
#[inline]
pub fn align_up_ptr(ptr: *const u8, alignment: usize) -> *const u8 {
    align_up(ptr as usize, alignment) as *const u8
}

/// Round a pointer address down to the previous multiple of `alignment`.
///
/// `alignment` must be a power of two.
#[inline]
pub fn align_down_ptr(ptr: *const u8, alignment: usize) -> *const u8 {
    align_down(ptr as usize, alignment) as *const u8
}

/// Check if a pointer address is aligned to `alignment`.
///
/// `alignment` must be a power of two.
#[inline]
pub fn is_aligned_ptr(ptr: *const u8, alignment: usize) -> bool {
    is_aligned(ptr as usize, alignment)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- align_up tests (mirroring C++ test_detail_utils.cpp) ----

    #[test]
    fn test_align_up_already_aligned() {
        assert_eq!(align_up(0, 4096), 0);
        assert_eq!(align_up(4096, 4096), 4096);
        assert_eq!(align_up(8192, 4096), 8192);
    }

    #[test]
    fn test_align_up_not_aligned() {
        assert_eq!(align_up(1, 4096), 4096);
        assert_eq!(align_up(4095, 4096), 4096);
        assert_eq!(align_up(4097, 4096), 8192);
        assert_eq!(align_up(100, 4096), 4096);
    }

    #[test]
    fn test_align_up_various_alignments() {
        assert_eq!(align_up(3, 2), 4);
        assert_eq!(align_up(5, 4), 8);
        assert_eq!(align_up(7, 8), 8);
        assert_eq!(align_up(9, 16), 16);
        assert_eq!(align_up(0, 1), 0);
        assert_eq!(align_up(1, 1), 1);
    }

    // ---- align_down tests ----

    #[test]
    fn test_align_down_already_aligned() {
        assert_eq!(align_down(0, 4096), 0);
        assert_eq!(align_down(4096, 4096), 4096);
        assert_eq!(align_down(8192, 4096), 8192);
    }

    #[test]
    fn test_align_down_not_aligned() {
        assert_eq!(align_down(1, 4096), 0);
        assert_eq!(align_down(4095, 4096), 0);
        assert_eq!(align_down(4097, 4096), 4096);
        assert_eq!(align_down(8191, 4096), 4096);
    }

    #[test]
    fn test_align_down_various_alignments() {
        assert_eq!(align_down(3, 2), 2);
        assert_eq!(align_down(5, 4), 4);
        assert_eq!(align_down(7, 8), 0);
        assert_eq!(align_down(17, 16), 16);
    }

    // ---- is_aligned tests ----

    #[test]
    fn test_is_aligned_true() {
        assert!(is_aligned(0, 4096));
        assert!(is_aligned(4096, 4096));
        assert!(is_aligned(8192, 4096));
        assert!(is_aligned(0, 1));
        assert!(is_aligned(1, 1));
        assert!(is_aligned(4, 2));
    }

    #[test]
    fn test_is_aligned_false() {
        assert!(!is_aligned(1, 4096));
        assert!(!is_aligned(4095, 4096));
        assert!(!is_aligned(4097, 4096));
        assert!(!is_aligned(3, 2));
        assert!(!is_aligned(5, 4));
    }

    // ---- Pointer alignment tests ----

    #[test]
    fn test_align_up_ptr() {
        let base = 0x1000 as *const u8;
        assert_eq!(align_up_ptr(base, 4096), 0x1000 as *const u8);

        let unaligned = 0x1001 as *const u8;
        assert_eq!(align_up_ptr(unaligned, 4096), 0x2000 as *const u8);
    }

    #[test]
    fn test_align_down_ptr() {
        let base = 0x1000 as *const u8;
        assert_eq!(align_down_ptr(base, 4096), 0x1000 as *const u8);

        let unaligned = 0x1FFF as *const u8;
        assert_eq!(align_down_ptr(unaligned, 4096), 0x1000 as *const u8);
    }

    #[test]
    fn test_is_aligned_ptr() {
        assert!(is_aligned_ptr(0x1000 as *const u8, 4096));
        assert!(!is_aligned_ptr(0x1001 as *const u8, 4096));
    }

    // ---- page_size test ----

    #[test]
    fn test_page_size() {
        let ps = page_size();
        assert!(ps > 0);
        assert!(ps.is_power_of_two());
        // On most systems this is 4096
        assert!(ps >= 4096);
    }

    // ---- Property-based alignment tests ----

    #[test]
    fn test_align_up_ge_value() {
        for v in 0..10000 {
            assert!(align_up(v, 4096) >= v);
        }
    }

    #[test]
    fn test_align_down_le_value() {
        for v in 0..10000 {
            assert!(align_down(v, 4096) <= v);
        }
    }

    #[test]
    fn test_align_up_result_is_aligned() {
        for v in 0..10000 {
            assert!(is_aligned(align_up(v, 4096), 4096));
        }
    }

    #[test]
    fn test_align_down_result_is_aligned() {
        for v in 0..10000 {
            assert!(is_aligned(align_down(v, 4096), 4096));
        }
    }

    #[test]
    fn test_roundtrip_aligned_values() {
        // For already-aligned values, align_up and align_down should be identity.
        for multiple in 0..100 {
            let v = multiple * 4096;
            assert_eq!(align_up(v, 4096), v);
            assert_eq!(align_down(v, 4096), v);
            assert!(is_aligned(v, 4096));
        }
    }
}
