//! Per-call scratch rows without a heap allocation.

use std::mem::MaybeUninit;

/// Runs `f` on a zeroed scratch slice of `len` elements, kept on the stack
/// when `len <= N` and on the heap beyond. Only the `len` elements handed to
/// `f` are zeroed, so a large `N` costs no extra work.
#[inline(always)]
pub(crate) fn with_scratch<T: Copy + Default, const N: usize, R>(
    len: usize,
    f: impl FnOnce(&mut [T]) -> R,
) -> R {
    if len > N {
        return f(&mut vec![T::default(); len]);
    }
    let mut stack = [MaybeUninit::<T>::uninit(); N];
    for slot in &mut stack[..len] {
        slot.write(T::default());
    }
    // SAFETY: the first `len <= N` elements were just initialised, and
    // `MaybeUninit<T>` has the layout of `T`.
    let buf = unsafe { std::slice::from_raw_parts_mut(stack.as_mut_ptr().cast::<T>(), len) };
    f(buf)
}

#[cfg(test)]
mod tests {
    use super::with_scratch;

    #[test]
    fn stack_and_heap_paths_are_zeroed() {
        for len in [0usize, 1, 7, 8, 9, 100] {
            let sum: u32 = with_scratch::<u32, 8, _>(len, |buf| {
                assert_eq!(buf.len(), len);
                let s = buf.iter().sum();
                buf.fill(7);
                s
            });
            assert_eq!(sum, 0, "len {len}");
        }
    }
}
