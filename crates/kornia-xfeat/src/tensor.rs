//! NHWC f32 tensor with 64-byte (cache-line) aligned storage and a fixed-size
//! ping-pong arena that owns every intermediate the model produces.
//!
//! The model is non-batched, so the rank is always three: `(H, W, C)`. NHWC
//! is chosen over NCHW because the inner loop of every conv kernel walks the
//! channel axis, which means the contiguous axis (C) maps directly onto SIMD
//! lanes — `f32x4` on NEON, `f32x8` on AVX2, `f32x16` on AVX-512.
//!
//! All allocations happen at model construction time. `extract()` runs against
//! pre-allocated slots in the arena; the only growable allocation during
//! inference is the sparse keypoint output, which itself caps at `top_k`.

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::ptr::NonNull;
use std::slice;

/// 64-byte alignment so every row start lands on a cache-line boundary and the
/// SIMD load pipes stay at peak throughput (the NEON skill calls this out
/// explicitly for Cortex-A78AE).
const ALIGN: usize = 64;

/// Owned, cache-line-aligned NHWC f32 tensor.
///
/// Layout: `data[h * w * c + w_idx * c + c_idx]` is the value at `(h, w_idx, c_idx)`.
/// `len = h * w * c` and the buffer is allocated rounded up to a multiple of 16
/// f32s (64 B) so SIMD loads at the tail don't need a special mask.
#[derive(Debug)]
pub struct NhwcTensor {
    ptr: NonNull<f32>,
    /// Spatial height.
    h: usize,
    /// Spatial width.
    w: usize,
    /// Channel count.
    c: usize,
    /// Allocated capacity in f32 elements (`>= h*w*c`, rounded up).
    capacity: usize,
}

// Safety: NhwcTensor owns a heap-allocated buffer of plain f32 with no thread-local state.
unsafe impl Send for NhwcTensor {}
unsafe impl Sync for NhwcTensor {}

impl NhwcTensor {
    /// Allocate an `(h, w, c)` tensor, zero-initialized.
    pub fn zeros(h: usize, w: usize, c: usize) -> Self {
        let logical = h
            .checked_mul(w)
            .and_then(|v| v.checked_mul(c))
            .expect("NhwcTensor shape overflows usize");
        // Round capacity up to a multiple of 16 f32s so a trailing SIMD load
        // never reads uninitialized memory beyond the allocation.
        let capacity = logical.div_ceil(16) * 16;
        let bytes = capacity
            .checked_mul(std::mem::size_of::<f32>())
            .expect("NhwcTensor allocation overflows usize");
        let layout = Layout::from_size_align(bytes.max(ALIGN), ALIGN)
            .expect("NhwcTensor alignment is valid");
        // SAFETY: layout is non-zero (>= ALIGN bytes) with a valid power-of-two
        // alignment; alloc_zeroed returns either null or a valid pointer.
        let raw = unsafe { alloc_zeroed(layout) } as *mut f32;
        let ptr = NonNull::new(raw).expect("NhwcTensor allocation failed");
        Self {
            ptr,
            h,
            w,
            c,
            capacity,
        }
    }

    /// `(height, width, channels)`.
    #[inline]
    pub fn shape(&self) -> (usize, usize, usize) {
        (self.h, self.w, self.c)
    }
    /// Spatial height.
    #[inline]
    pub fn h(&self) -> usize {
        self.h
    }
    /// Spatial width.
    #[inline]
    pub fn w(&self) -> usize {
        self.w
    }
    /// Channel count.
    #[inline]
    pub fn c(&self) -> usize {
        self.c
    }
    /// Number of logical f32 elements (`h * w * c`).
    #[inline]
    pub fn len(&self) -> usize {
        self.h * self.w * self.c
    }
    /// `true` if any dimension is zero.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read-only slice view over the logical extent. Length = `h*w*c`.
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        // SAFETY: ptr is non-null, points to `capacity` valid f32, and `len <= capacity`.
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len()) }
    }

    /// Mutable slice view.
    #[inline]
    pub fn as_slice_mut(&mut self) -> &mut [f32] {
        // SAFETY: same invariants as `as_slice`; exclusive borrow guarantees no aliasing.
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len()) }
    }

    /// Raw const pointer. Only use under `unsafe` SIMD paths.
    #[inline]
    pub fn as_ptr(&self) -> *const f32 {
        self.ptr.as_ptr()
    }
    /// Raw mut pointer. Only use under `unsafe` SIMD paths.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self.ptr.as_ptr()
    }

    /// Zero the logical extent. Used by the arena between ping-pong reuses to
    /// avoid carrying stale tails into a kernel that writes only part of the
    /// buffer.
    #[inline]
    pub fn zero(&mut self) {
        // Zero the *capacity* not just the logical len: kernels are allowed
        // to read up to the rounded tail for free SIMD tails.
        // SAFETY: ptr/capacity invariants hold; f32 with all-bits-zero is a
        // valid 0.0.
        unsafe {
            std::ptr::write_bytes(self.ptr.as_ptr(), 0, self.capacity);
        }
    }
}

impl Drop for NhwcTensor {
    fn drop(&mut self) {
        let bytes = self.capacity * std::mem::size_of::<f32>();
        let layout = Layout::from_size_align(bytes.max(ALIGN), ALIGN).expect("layout");
        // SAFETY: pointer was returned by `alloc_zeroed` with this exact layout.
        unsafe { dealloc(self.ptr.as_ptr() as *mut u8, layout) }
    }
}

/// Two-buffer arena: every layer reads from one buffer and writes to the other,
/// then the next layer flips. The two slots are sized at construction to the
/// largest intermediate the model produces.
///
/// Single image, single thread of inference at a time. The model's mutable
/// borrow on `XFeat` (via `&mut self` on `extract`) prevents two simultaneous
/// `extract` calls on the same instance, so the arena does not need interior
/// mutability.
#[derive(Debug)]
pub struct PingPongArena {
    a: NhwcTensor,
    b: NhwcTensor,
    /// Tracks which slot was most recently written so callers don't have to
    /// pass `front` / `back` around explicitly.
    front_is_a: bool,
}

impl PingPongArena {
    /// Allocate two `(max_h, max_w, max_c)` slots.
    pub fn new(max_h: usize, max_w: usize, max_c: usize) -> Self {
        Self {
            a: NhwcTensor::zeros(max_h, max_w, max_c),
            b: NhwcTensor::zeros(max_h, max_w, max_c),
            front_is_a: true,
        }
    }

    /// `(input_slot, output_slot)` for the next layer; flip on the call after.
    pub fn pair_mut(&mut self) -> (&NhwcTensor, &mut NhwcTensor) {
        if self.front_is_a {
            (&self.a, &mut self.b)
        } else {
            (&self.b, &mut self.a)
        }
    }

    /// Advance the ping-pong cursor.
    pub fn flip(&mut self) {
        self.front_is_a = !self.front_is_a;
    }

    /// Read the most recently *written* slot.
    pub fn front(&self) -> &NhwcTensor {
        if self.front_is_a {
            &self.a
        } else {
            &self.b
        }
    }

    /// Reset both buffers to zero. Called once per `extract()` before the first
    /// layer fires; protects against a kernel that writes only a sub-region.
    pub fn reset(&mut self) {
        self.a.zero();
        self.b.zero();
        self.front_is_a = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocates_aligned_to_64() {
        let t = NhwcTensor::zeros(7, 13, 5);
        assert_eq!(t.as_ptr() as usize % ALIGN, 0);
    }

    #[test]
    fn shape_and_len_consistent() {
        let t = NhwcTensor::zeros(4, 5, 3);
        assert_eq!(t.shape(), (4, 5, 3));
        assert_eq!(t.len(), 60);
        assert_eq!(t.as_slice().len(), 60);
    }

    #[test]
    fn zeros_returns_zero() {
        let t = NhwcTensor::zeros(2, 2, 4);
        assert!(t.as_slice().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn ping_pong_alternates() {
        let mut arena = PingPongArena::new(2, 2, 2);
        let a_ptr = arena.front().as_ptr() as usize;
        arena.flip();
        let b_ptr = arena.front().as_ptr() as usize;
        assert_ne!(a_ptr, b_ptr);
        arena.flip();
        assert_eq!(arena.front().as_ptr() as usize, a_ptr);
    }
}
