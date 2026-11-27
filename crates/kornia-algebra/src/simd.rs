//! Lane-batched `f32` types for structure-of-arrays kernels.
//!
//! [`F32x4`] holds four `f32` lanes and [`Mask32x4`] four boolean lanes. Both wrap glam's
//! `Vec4` and `BVec4A`, so they lower to SSE2 on x86-64 and NEON on aarch64 with no `unsafe`
//! here and no extra dependency. Each is a `#[repr(transparent)]` newtype that stores the glam
//! register itself, so operators act on it directly with no per-op conversion.
//!
//! [`F32x8`] and [`Mask32x8`] are eight lanes with the same API and NaN rules. On x86-64
//! built with the `avx2` target feature they are one 256-bit AVX register; elsewhere (SSE2,
//! NEON) two four-lane halves, lanes 0-3 then 4-7. The backend is fixed at compile time, not
//! dispatched at run time, and both give the same bits. Even as two 128-bit ops per op they
//! cost less per point than one [`F32x4`] op does, because a kernel's per-chunk fixed costs
//! (loop, bitmask branch, splats) are paid once per eight lanes.
//!
//! # NaN
//!
//! The comparisons are ordered: a lane is `false` when either operand is NaN, `cmpeq`
//! included. A gate such as `d2.cmple(max_sq)` therefore rejects NaN lanes, and
//! `!a.cmpeq(b)` is `true` for them. [`F32x4::min`] and [`F32x4::max`] return `rhs` when either
//! lane is NaN, on every target (see their docs).
//!
//! # Example
//!
//! ```
//! use kornia_algebra::simd::F32x4;
//!
//! let d2 = F32x4::new(0.5, 2.0, f32::NAN, 1.0);
//! let inlier = d2.cmple(F32x4::splat(1.0));
//! assert_eq!(inlier.bitmask(), 0b1001);
//! let w = F32x4::select(inlier, F32x4::ONE, F32x4::ZERO);
//! assert_eq!(w.element_sum(), 2.0);
//! ```

use glam::{BVec4A, Vec4};
use std::ops::{Add, AddAssign, BitAnd, BitOr, Div, Mul, MulAssign, Neg, Not, Sub, SubAssign};

/// Four `f32` lanes. Lane `i` is element `i` of the array or slice it was loaded from.
#[derive(Debug, Clone, Copy, Default)]
#[repr(transparent)]
pub struct F32x4(Vec4);

/// Four boolean lanes, the result of an [`F32x4`] comparison.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct Mask32x4(BVec4A);

impl F32x4 {
    /// All lanes `0.0`.
    pub const ZERO: Self = Self(Vec4::ZERO);
    /// All lanes `1.0`.
    pub const ONE: Self = Self(Vec4::ONE);
    /// Number of lanes.
    pub const LANES: usize = 4;

    /// Creates a value from its four lanes.
    #[inline]
    pub const fn new(l0: f32, l1: f32, l2: f32, l3: f32) -> Self {
        Self(Vec4::new(l0, l1, l2, l3))
    }

    /// Sets every lane to `v`.
    #[inline]
    pub const fn splat(v: f32) -> Self {
        Self(Vec4::splat(v))
    }

    /// Loads four lanes from an array; lane `i` is `src[i]`.
    #[inline]
    pub const fn from_array(src: [f32; 4]) -> Self {
        Self(Vec4::from_array(src))
    }

    /// Returns the lanes as an array; element `i` is lane `i`.
    #[inline]
    pub const fn to_array(self) -> [f32; 4] {
        self.0.to_array()
    }

    /// Loads the first four elements of `src`.
    ///
    /// # Returns
    ///
    /// `None` if `src` holds fewer than four elements.
    #[inline]
    pub fn from_slice(src: &[f32]) -> Option<Self> {
        src.first_chunk::<4>().map(|a| Self::from_array(*a))
    }

    /// Stores the lanes into the first four elements of `dst`.
    ///
    /// # Returns
    ///
    /// `None`, with `dst` untouched, if `dst` holds fewer than four elements.
    #[inline]
    pub fn write_to_slice(self, dst: &mut [f32]) -> Option<()> {
        *dst.first_chunk_mut::<4>()? = self.to_array();
        Some(())
    }

    /// `self * a + b`. Fused (one rounding) where the target has an FMA instruction, that is
    /// aarch64 and x86 built with the `fma` target feature; elsewhere a multiply then an add,
    /// because the software `fma` fallback is many times slower.
    #[inline]
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        #[cfg(any(target_arch = "aarch64", target_feature = "fma"))]
        {
            Self(self.0.mul_add(a.0, b.0))
        }
        #[cfg(not(any(target_arch = "aarch64", target_feature = "fma")))]
        {
            Self(self.0 * a.0 + b.0)
        }
    }

    /// Lane-wise square root; negative lanes give NaN.
    #[inline]
    pub fn sqrt(self) -> Self {
        Self(self.0.sqrt())
    }

    /// Lane-wise absolute value; clears the sign bit, so NaN stays NaN.
    #[inline]
    pub fn abs(self) -> Self {
        Self(self.0.abs())
    }

    /// Lane-wise `if self < rhs { self } else { rhs }`.
    ///
    /// A NaN in either lane gives `rhs`, as does `min(-0.0, 0.0)`. This is x86 `minps` and is
    /// kept on aarch64, whose `fmin` would propagate the NaN, so results match across targets.
    /// A possibly-NaN value in `self` is replaced by `rhs`; in `rhs` the NaN is kept.
    #[inline]
    pub fn min(self, rhs: Self) -> Self {
        // per-lane form: LLVM folds it to one `minps` on x86, whereas glam's and/andnot/or
        // `select` stays four instructions
        let (a, b) = (self.to_array(), rhs.to_array());
        Self::from_array(std::array::from_fn(
            |i| if a[i] < b[i] { a[i] } else { b[i] },
        ))
    }

    /// Lane-wise `if self > rhs { self } else { rhs }`; NaN in either lane gives `rhs`, as for
    /// [`F32x4::min`].
    #[inline]
    pub fn max(self, rhs: Self) -> Self {
        let (a, b) = (self.to_array(), rhs.to_array());
        Self::from_array(std::array::from_fn(
            |i| if a[i] > b[i] { a[i] } else { b[i] },
        ))
    }

    /// Sum of the four lanes, `(l0 + l1) + (l2 + l3)` on x86; the grouping may differ on other
    /// targets.
    #[inline]
    pub fn element_sum(self) -> f32 {
        self.0.element_sum()
    }

    /// Lane-wise `self < rhs`; `false` where either lane is NaN.
    #[inline]
    pub fn cmplt(self, rhs: Self) -> Mask32x4 {
        Mask32x4(self.0.cmplt(rhs.0))
    }

    /// Lane-wise `self <= rhs`; `false` where either lane is NaN.
    #[inline]
    pub fn cmple(self, rhs: Self) -> Mask32x4 {
        Mask32x4(self.0.cmple(rhs.0))
    }

    /// Lane-wise `self > rhs`; `false` where either lane is NaN.
    #[inline]
    pub fn cmpgt(self, rhs: Self) -> Mask32x4 {
        Mask32x4(self.0.cmpgt(rhs.0))
    }

    /// Lane-wise `self >= rhs`; `false` where either lane is NaN.
    #[inline]
    pub fn cmpge(self, rhs: Self) -> Mask32x4 {
        Mask32x4(self.0.cmpge(rhs.0))
    }

    /// Lane-wise `self == rhs`; `false` where either lane is NaN, `true` for `-0.0 == 0.0`.
    #[inline]
    pub fn cmpeq(self, rhs: Self) -> Mask32x4 {
        Mask32x4(self.0.cmpeq(rhs.0))
    }

    /// Takes each lane from `if_true` where `mask` is set, else from `if_false`.
    #[inline]
    pub fn select(mask: Mask32x4, if_true: Self, if_false: Self) -> Self {
        Self(Vec4::select(mask.0, if_true.0, if_false.0))
    }
}

impl Mask32x4 {
    /// Sets every lane to `v`.
    #[inline]
    pub const fn splat(v: bool) -> Self {
        Self(BVec4A::splat(v))
    }

    /// Builds a mask from its lanes; lane `i` is `lanes[i]`.
    #[inline]
    pub const fn from_array(lanes: [bool; 4]) -> Self {
        Self(BVec4A::from_array(lanes))
    }

    /// Bit `i` is lane `i`; the upper 28 bits are zero.
    #[inline]
    pub fn bitmask(self) -> u32 {
        self.0.bitmask()
    }

    /// `true` if any lane is set.
    #[inline]
    pub fn any(self) -> bool {
        self.0.any()
    }

    /// `true` if every lane is set.
    #[inline]
    pub fn all(self) -> bool {
        self.0.all()
    }
}

impl From<Vec4> for F32x4 {
    #[inline]
    fn from(v: Vec4) -> Self {
        Self(v)
    }
}

impl From<F32x4> for Vec4 {
    #[inline]
    fn from(v: F32x4) -> Self {
        v.0
    }
}

macro_rules! impl_binop {
    ($op:ident, $f:ident, $op_assign:ident, $f_assign:ident) => {
        impl $op for F32x4 {
            type Output = Self;
            #[inline]
            fn $f(self, rhs: Self) -> Self {
                Self(self.0.$f(rhs.0))
            }
        }
        impl $op_assign for F32x4 {
            #[inline]
            fn $f_assign(&mut self, rhs: Self) {
                self.0.$f_assign(rhs.0);
            }
        }
    };
}

impl_binop!(Add, add, AddAssign, add_assign);
impl_binop!(Sub, sub, SubAssign, sub_assign);
impl_binop!(Mul, mul, MulAssign, mul_assign);

impl Div for F32x4 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        Self(self.0 / rhs.0)
    }
}

/// Multiplies every lane by a scalar: glam's `Vec4 * f32`, which is one `fmul` by element on
/// aarch64 (`vmulq_n_f32`) and a broadcast plus `mulps` on x86, where a loop-invariant scalar's
/// broadcast is hoisted out of the loop.
impl Mul<f32> for F32x4 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f32) -> Self {
        Self(self.0 * rhs)
    }
}

impl Neg for F32x4 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl BitAnd for Mask32x4 {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}

impl BitOr for Mask32x4 {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl Not for Mask32x4 {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}

// The backend is chosen at compile time: a value type used inside hot loops has no call boundary
// where a runtime `cpu_features()` check could sit without costing more than the op it picks.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use x8_avx2 as x8;
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
use x8_halves as x8;

/// Two [`F32x4`] halves, lanes 0-3 then 4-7: the portable path (SSE2, NEON).
#[cfg(any(test, not(all(target_arch = "x86_64", target_feature = "avx2"))))]
mod x8_halves {
    use super::{F32x4, Mask32x4};
    use std::ops::{Add, Div, Mul, Neg, Sub};

    pub(super) type F = [F32x4; 2];
    pub(super) type M = [Mask32x4; 2];

    #[inline(always)]
    pub(super) const fn from_array(s: [f32; 8]) -> F {
        [
            F32x4::new(s[0], s[1], s[2], s[3]),
            F32x4::new(s[4], s[5], s[6], s[7]),
        ]
    }

    #[inline(always)]
    pub(super) const fn to_array(v: F) -> [f32; 8] {
        let (a, b) = (v[0].to_array(), v[1].to_array());
        [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]]
    }

    #[inline(always)]
    pub(super) const fn mask_from_array(l: [bool; 8]) -> M {
        [
            Mask32x4::from_array([l[0], l[1], l[2], l[3]]),
            Mask32x4::from_array([l[4], l[5], l[6], l[7]]),
        ]
    }

    macro_rules! per_half {
        ($($f:ident -> $out:ty),*) => {$(
            #[inline(always)]
            pub(super) fn $f(a: F, b: F) -> $out {
                [a[0].$f(b[0]), a[1].$f(b[1])]
            }
        )*};
    }
    per_half!(add -> F, sub -> F, mul -> F, div -> F, min -> F, max -> F);
    per_half!(cmplt -> M, cmple -> M, cmpgt -> M, cmpge -> M, cmpeq -> M);

    #[inline(always)]
    pub(super) fn neg(a: F) -> F {
        [a[0].neg(), a[1].neg()]
    }

    #[inline(always)]
    pub(super) fn abs(a: F) -> F {
        [a[0].abs(), a[1].abs()]
    }

    #[inline(always)]
    pub(super) fn sqrt(a: F) -> F {
        [a[0].sqrt(), a[1].sqrt()]
    }

    #[inline(always)]
    pub(super) fn scale(a: F, s: f32) -> F {
        [a[0] * s, a[1] * s]
    }

    #[inline(always)]
    pub(super) fn mul_add(a: F, b: F, c: F) -> F {
        [a[0].mul_add(b[0], c[0]), a[1].mul_add(b[1], c[1])]
    }

    #[inline(always)]
    pub(super) fn element_sum(a: F) -> f32 {
        (a[0] + a[1]).element_sum()
    }

    #[inline(always)]
    pub(super) fn select(m: M, t: F, f: F) -> F {
        [
            F32x4::select(m[0], t[0], f[0]),
            F32x4::select(m[1], t[1], f[1]),
        ]
    }

    #[inline(always)]
    pub(super) fn bitmask(m: M) -> u32 {
        m[0].bitmask() | (m[1].bitmask() << 4)
    }

    #[inline(always)]
    pub(super) fn any(m: M) -> bool {
        (m[0] | m[1]).any()
    }

    #[inline(always)]
    pub(super) fn all(m: M) -> bool {
        (m[0] & m[1]).all()
    }

    #[inline(always)]
    pub(super) fn and(a: M, b: M) -> M {
        [a[0] & b[0], a[1] & b[1]]
    }

    #[inline(always)]
    pub(super) fn or(a: M, b: M) -> M {
        [a[0] | b[0], a[1] | b[1]]
    }

    #[inline(always)]
    pub(super) fn not(a: M) -> M {
        [!a[0], !a[1]]
    }
}

/// One `__m256` register per value. Every intrinsic call is `unsafe` only because it is a
/// `#[target_feature]` fn; the `cfg` below guarantees the build enables that feature.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
mod x8_avx2 {
    use super::F32x4;
    use std::arch::x86_64::*;

    pub(super) type F = __m256;
    /// Each lane is all ones (set) or all zeros, as `_mm256_cmp_ps` writes it.
    pub(super) type M = __m256;

    const ALL_LANES: u32 = 0xff;

    #[inline(always)]
    pub(super) const fn from_array(s: [f32; 8]) -> F {
        // SAFETY: `[f32; 8]` and `__m256` are both 32 bytes of plain data, every bit pattern is
        // valid for both, and lane `i` of the register is element `i` of the array.
        unsafe { std::mem::transmute::<[f32; 8], __m256>(s) }
    }

    #[inline(always)]
    pub(super) const fn to_array(v: F) -> [f32; 8] {
        // SAFETY: as in `from_array`, in the other direction.
        unsafe { std::mem::transmute::<__m256, [f32; 8]>(v) }
    }

    #[inline(always)]
    pub(super) const fn mask_from_array(l: [bool; 8]) -> M {
        let mut bits = [0u32; 8];
        let mut i = 0;
        while i < 8 {
            bits[i] = if l[i] { u32::MAX } else { 0 };
            i += 1;
        }
        // SAFETY: `[u32; 8]` and `__m256` are both 32 bytes of plain data with no invalid bit
        // patterns; lane `i` gets `bits[i]`.
        unsafe { std::mem::transmute::<[u32; 8], __m256>(bits) }
    }

    macro_rules! binop {
        ($($f:ident => $op:ident -> $out:ty),*) => {$(
            #[inline(always)]
            pub(super) fn $f(a: F, b: F) -> $out {
                // SAFETY: this module compiles only with `avx2` enabled for the whole build.
                unsafe { $op(a, b) }
            }
        )*};
    }
    // `minps`/`maxps` return their second operand, `b` = `rhs`, when either lane is NaN, as
    // `F32x4::min`/`max` do; the operand order here is what keeps that rule
    binop!(add => _mm256_add_ps -> F, sub => _mm256_sub_ps -> F, mul => _mm256_mul_ps -> F);
    binop!(div => _mm256_div_ps -> F, min => _mm256_min_ps -> F, max => _mm256_max_ps -> F);
    binop!(and => _mm256_and_ps -> M, or => _mm256_or_ps -> M);

    macro_rules! cmpop {
        ($($f:ident => $pred:ident),*) => {$(
            /// Ordered and quiet: `false` where either lane is NaN.
            #[inline(always)]
            pub(super) fn $f(a: F, b: F) -> M {
                // SAFETY: this module compiles only with `avx2` enabled for the whole build.
                unsafe { _mm256_cmp_ps::<$pred>(a, b) }
            }
        )*};
    }
    cmpop!(cmplt => _CMP_LT_OQ, cmple => _CMP_LE_OQ, cmpgt => _CMP_GT_OQ);
    cmpop!(cmpge => _CMP_GE_OQ, cmpeq => _CMP_EQ_OQ);

    #[inline(always)]
    pub(super) fn neg(a: F) -> F {
        // SAFETY: `avx2` is enabled for the whole build (this module's cfg).
        unsafe { _mm256_xor_ps(a, _mm256_set1_ps(-0.0)) }
    }

    #[inline(always)]
    pub(super) fn abs(a: F) -> F {
        // SAFETY: `avx2` is enabled for the whole build (this module's cfg).
        unsafe { _mm256_andnot_ps(_mm256_set1_ps(-0.0), a) }
    }

    #[inline(always)]
    pub(super) fn sqrt(a: F) -> F {
        // SAFETY: `avx2` is enabled for the whole build (this module's cfg).
        unsafe { _mm256_sqrt_ps(a) }
    }

    #[inline(always)]
    pub(super) fn scale(a: F, s: f32) -> F {
        // SAFETY: `avx2` is enabled for the whole build (this module's cfg).
        unsafe { _mm256_mul_ps(a, _mm256_set1_ps(s)) }
    }

    #[inline(always)]
    pub(super) fn mul_add(a: F, b: F, c: F) -> F {
        #[cfg(target_feature = "fma")]
        {
            // SAFETY: `avx2` (this module's cfg) and `fma` (this block's cfg) are enabled.
            unsafe { _mm256_fmadd_ps(a, b, c) }
        }
        #[cfg(not(target_feature = "fma"))]
        {
            // SAFETY: `avx2` is enabled for the whole build (this module's cfg).
            unsafe { _mm256_add_ps(_mm256_mul_ps(a, b), c) }
        }
    }

    /// Lanes 0-3 plus lanes 4-7, then the [`F32x4::element_sum`] reduction: the grouping of
    /// the two-halves path, so both give the same bits.
    #[inline(always)]
    pub(super) fn element_sum(a: F) -> f32 {
        // SAFETY: this module compiles only with `avx2` enabled for the whole build.
        let h = unsafe { _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps::<1>(a)) };
        F32x4::from(glam::Vec4::from(h)).element_sum()
    }

    #[inline(always)]
    pub(super) fn select(m: M, t: F, f: F) -> F {
        // SAFETY: `avx2` is enabled for the whole build (this module's cfg).
        unsafe { _mm256_blendv_ps(f, t, m) }
    }

    #[inline(always)]
    pub(super) fn bitmask(m: M) -> u32 {
        // SAFETY: `avx2` is enabled for the whole build (this module's cfg).
        unsafe { _mm256_movemask_ps(m) as u32 }
    }

    #[inline(always)]
    pub(super) fn any(m: M) -> bool {
        bitmask(m) != 0
    }

    #[inline(always)]
    pub(super) fn all(m: M) -> bool {
        bitmask(m) == ALL_LANES
    }

    #[inline(always)]
    pub(super) fn not(m: M) -> M {
        // SAFETY: `avx2` is enabled for the whole build (this module's cfg).
        unsafe { _mm256_xor_ps(m, _mm256_castsi256_ps(_mm256_set1_epi32(-1))) }
    }
}

/// Eight `f32` lanes; lane `i` is element `i` of its source.
///
/// Every op has the semantics of the [`F32x4`] op of the same name. On x86-64 built with the
/// `avx2` target feature (for example `-C target-cpu=x86-64-v3`) a value is one `__m256`
/// register; elsewhere it is two [`F32x4`] halves. Both give the same bits for every op,
/// [`F32x8::element_sum`] included.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct F32x8(x8::F);

/// Eight boolean lanes, the result of an [`F32x8`] comparison.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Mask32x8(x8::M);

impl F32x8 {
    /// All lanes `0.0`.
    pub const ZERO: Self = Self::splat(0.0);
    /// All lanes `1.0`.
    pub const ONE: Self = Self::splat(1.0);
    /// Number of lanes.
    pub const LANES: usize = 8;

    /// Creates a value from its eight lanes.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub const fn new(
        l0: f32,
        l1: f32,
        l2: f32,
        l3: f32,
        l4: f32,
        l5: f32,
        l6: f32,
        l7: f32,
    ) -> Self {
        Self::from_array([l0, l1, l2, l3, l4, l5, l6, l7])
    }

    /// Sets every lane to `v`.
    #[inline]
    pub const fn splat(v: f32) -> Self {
        Self::from_array([v; 8])
    }

    /// Loads eight lanes from an array; lane `i` is `src[i]`.
    #[inline]
    pub const fn from_array(src: [f32; 8]) -> Self {
        Self(x8::from_array(src))
    }

    /// Returns the lanes as an array; element `i` is lane `i`.
    #[inline]
    pub const fn to_array(self) -> [f32; 8] {
        x8::to_array(self.0)
    }

    /// Loads the first eight elements of `src`.
    ///
    /// # Returns
    ///
    /// `None` if `src` holds fewer than eight elements.
    #[inline]
    pub fn from_slice(src: &[f32]) -> Option<Self> {
        src.first_chunk::<8>().map(|a| Self::from_array(*a))
    }

    /// Stores the lanes into the first eight elements of `dst`.
    ///
    /// # Returns
    ///
    /// `None`, with `dst` untouched, if `dst` holds fewer than eight elements.
    #[inline]
    pub fn write_to_slice(self, dst: &mut [f32]) -> Option<()> {
        *dst.first_chunk_mut::<8>()? = self.to_array();
        Some(())
    }

    /// `self * a + b`, fused where [`F32x4::mul_add`] is.
    #[inline]
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        Self(x8::mul_add(self.0, a.0, b.0))
    }

    /// Lane-wise square root; negative lanes give NaN.
    #[inline]
    pub fn sqrt(self) -> Self {
        Self(x8::sqrt(self.0))
    }

    /// Lane-wise absolute value; NaN stays NaN.
    #[inline]
    pub fn abs(self) -> Self {
        Self(x8::abs(self.0))
    }

    /// Lane-wise minimum; NaN in either lane gives `rhs`, as for [`F32x4::min`].
    #[inline]
    pub fn min(self, rhs: Self) -> Self {
        Self(x8::min(self.0, rhs.0))
    }

    /// Lane-wise maximum; NaN in either lane gives `rhs`, as for [`F32x4::max`].
    #[inline]
    pub fn max(self, rhs: Self) -> Self {
        Self(x8::max(self.0, rhs.0))
    }

    /// Sum of the eight lanes: lanes 0-3 and 4-7 are added lane-wise, then that four-lane sum
    /// is reduced as [`F32x4::element_sum`]. The grouping is the same on both backends.
    #[inline]
    pub fn element_sum(self) -> f32 {
        x8::element_sum(self.0)
    }

    /// Lane-wise `self < rhs`; `false` where either lane is NaN.
    #[inline]
    pub fn cmplt(self, rhs: Self) -> Mask32x8 {
        Mask32x8(x8::cmplt(self.0, rhs.0))
    }

    /// Lane-wise `self <= rhs`; `false` where either lane is NaN.
    #[inline]
    pub fn cmple(self, rhs: Self) -> Mask32x8 {
        Mask32x8(x8::cmple(self.0, rhs.0))
    }

    /// Lane-wise `self > rhs`; `false` where either lane is NaN.
    #[inline]
    pub fn cmpgt(self, rhs: Self) -> Mask32x8 {
        Mask32x8(x8::cmpgt(self.0, rhs.0))
    }

    /// Lane-wise `self >= rhs`; `false` where either lane is NaN.
    #[inline]
    pub fn cmpge(self, rhs: Self) -> Mask32x8 {
        Mask32x8(x8::cmpge(self.0, rhs.0))
    }

    /// Lane-wise `self == rhs`; `false` where either lane is NaN, `true` for `-0.0 == 0.0`.
    #[inline]
    pub fn cmpeq(self, rhs: Self) -> Mask32x8 {
        Mask32x8(x8::cmpeq(self.0, rhs.0))
    }

    /// Takes each lane from `if_true` where `mask` is set, else from `if_false`.
    #[inline]
    pub fn select(mask: Mask32x8, if_true: Self, if_false: Self) -> Self {
        Self(x8::select(mask.0, if_true.0, if_false.0))
    }
}

impl Default for F32x8 {
    #[inline]
    fn default() -> Self {
        Self::ZERO
    }
}

// by lanes, so the output does not depend on the backend
impl std::fmt::Debug for F32x8 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("F32x8").field(&self.to_array()).finish()
    }
}

impl Mask32x8 {
    /// Sets every lane to `v`.
    #[inline]
    pub const fn splat(v: bool) -> Self {
        Self::from_array([v; 8])
    }

    /// Builds a mask from its lanes; lane `i` is `lanes[i]`.
    #[inline]
    pub const fn from_array(lanes: [bool; 8]) -> Self {
        Self(x8::mask_from_array(lanes))
    }

    /// Bit `i` is lane `i`; the upper 24 bits are zero.
    #[inline]
    pub fn bitmask(self) -> u32 {
        x8::bitmask(self.0)
    }

    /// `true` if any lane is set.
    #[inline]
    pub fn any(self) -> bool {
        x8::any(self.0)
    }

    /// `true` if every lane is set.
    #[inline]
    pub fn all(self) -> bool {
        x8::all(self.0)
    }
}

impl Default for Mask32x8 {
    #[inline]
    fn default() -> Self {
        Self::splat(false)
    }
}

// a lane is all ones or all zeros, so equal bitmasks mean equal masks
impl PartialEq for Mask32x8 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.bitmask() == other.bitmask()
    }
}

impl Eq for Mask32x8 {}

impl std::fmt::Debug for Mask32x8 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bits = self.bitmask();
        let lanes: [bool; 8] = std::array::from_fn(|i| bits >> i & 1 != 0);
        f.debug_tuple("Mask32x8").field(&lanes).finish()
    }
}

macro_rules! impl_binop_x8 {
    ($op:ident, $f:ident, $op_assign:ident, $f_assign:ident) => {
        impl $op for F32x8 {
            type Output = Self;
            #[inline]
            fn $f(self, rhs: Self) -> Self {
                Self(x8::$f(self.0, rhs.0))
            }
        }
        impl $op_assign for F32x8 {
            #[inline]
            fn $f_assign(&mut self, rhs: Self) {
                self.0 = x8::$f(self.0, rhs.0);
            }
        }
    };
}

impl_binop_x8!(Add, add, AddAssign, add_assign);
impl_binop_x8!(Sub, sub, SubAssign, sub_assign);
impl_binop_x8!(Mul, mul, MulAssign, mul_assign);

impl Div for F32x8 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        Self(x8::div(self.0, rhs.0))
    }
}

/// Multiplies every lane by a scalar: a broadcast plus one multiply per register.
impl Mul<f32> for F32x8 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f32) -> Self {
        Self(x8::scale(self.0, rhs))
    }
}

impl Neg for F32x8 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self(x8::neg(self.0))
    }
}

impl BitAnd for Mask32x8 {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(x8::and(self.0, rhs.0))
    }
}

impl BitOr for Mask32x8 {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(x8::or(self.0, rhs.0))
    }
}

impl Not for Mask32x8 {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self(x8::not(self.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const INF: f32 = f32::INFINITY;
    const A: [f32; 4] = [1.5, -2.0, INF, f32::NAN];
    const B: [f32; 4] = [0.5, 3.0, -INF, 1.0];

    /// Lane-wise result equals `f` on the scalar lanes, NaN matching NaN.
    fn assert_lanes(got: F32x4, f: impl Fn(f32, f32) -> f32) {
        for (i, g) in got.to_array().into_iter().enumerate() {
            let want = f(A[i], B[i]);
            assert!(
                g == want || (g.is_nan() && want.is_nan()),
                "lane {i}: {g} vs {want}"
            );
        }
    }

    fn mask_bits(f: impl Fn(f32, f32) -> bool) -> u32 {
        (0..4).map(|i| (f(A[i], B[i]) as u32) << i).sum()
    }

    #[test]
    fn arithmetic_matches_scalar() {
        let (a, b) = (F32x4::from_array(A), F32x4::from_array(B));
        assert_lanes(a + b, |x, y| x + y);
        assert_lanes(a - b, |x, y| x - y);
        assert_lanes(a * b, |x, y| x * y);
        assert_lanes(a / b, |x, y| x / y);
        assert_lanes(-a, |x, _| -x);
        assert_lanes(a * 0.5, |x, _| x * 0.5);
        assert_lanes(a.abs(), |x, _| x.abs());
        assert_lanes(b.sqrt(), |_, y| y.sqrt());
        assert_lanes(a.sqrt(), |x, _| x.sqrt());
        // exact products, so fused and unfused agree
        assert_lanes(a.mul_add(b, b), |x, y| x * y + y);
        let mut c = a;
        c += b;
        c -= b;
        c *= F32x4::ONE;
        assert_lanes(c, |x, _| if x.is_infinite() { f32::NAN } else { x });
    }

    #[test]
    fn min_max_return_rhs_on_nan() {
        let (a, b) = (F32x4::from_array(A), F32x4::from_array(B));
        assert_lanes(a.min(b), |x, y| if x < y { x } else { y });
        assert_lanes(a.max(b), |x, y| if x > y { x } else { y });
        // NaN in lhs gives rhs, NaN in rhs gives NaN
        assert_eq!(a.max(b).to_array()[3], 1.0);
        assert!(b.max(a).to_array()[3].is_nan());
        let z = F32x4::splat(0.0);
        assert!(F32x4::splat(-0.0).min(z).to_array()[0].is_sign_positive());
    }

    #[test]
    fn comparisons_are_false_on_nan() {
        let (a, b) = (F32x4::from_array(A), F32x4::from_array(B));
        assert_eq!(a.cmplt(b).bitmask(), mask_bits(|x, y| x < y));
        assert_eq!(a.cmple(b).bitmask(), mask_bits(|x, y| x <= y));
        assert_eq!(a.cmpgt(b).bitmask(), mask_bits(|x, y| x > y));
        assert_eq!(a.cmpge(b).bitmask(), mask_bits(|x, y| x >= y));
        assert_eq!(a.cmpeq(b).bitmask(), mask_bits(|x, y| x == y));
        let nan = F32x4::splat(f32::NAN);
        for m in [
            nan.cmplt(nan),
            nan.cmple(nan),
            nan.cmpgt(nan),
            nan.cmpge(nan),
            nan.cmpeq(nan),
        ] {
            assert_eq!(m.bitmask(), 0);
        }
        assert!((!nan.cmpeq(nan)).all());
        assert!(F32x4::splat(-0.0).cmpeq(F32x4::ZERO).all());
    }

    #[test]
    fn lane_order_of_load_store_and_bitmask() {
        let src = [1.0, 2.0, 3.0, 4.0, 5.0];
        let v = F32x4::from_slice(&src).unwrap_or_default();
        assert_eq!(v.to_array(), [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(F32x4::new(1.0, 2.0, 3.0, 4.0).to_array(), v.to_array());
        assert!(F32x4::from_slice(&src[..3]).is_none());
        let mut dst = [0.0; 5];
        assert_eq!(v.write_to_slice(&mut dst), Some(()));
        assert_eq!(dst, [1.0, 2.0, 3.0, 4.0, 0.0]);
        assert!(v.write_to_slice(&mut dst[..3]).is_none());
        assert_eq!(dst, [1.0, 2.0, 3.0, 4.0, 0.0]);
        // bit i is lane i: lanes 1..3 exceed 1.5, lanes 0..1 are below 2.5
        assert_eq!(v.cmpgt(F32x4::splat(1.5)).bitmask(), 0b1110);
        assert_eq!(v.cmplt(F32x4::splat(2.5)).bitmask(), 0b0011);
        assert_eq!(
            Mask32x4::from_array([true, false, false, true]).bitmask(),
            0b1001
        );
    }

    #[test]
    fn mask_logic_select_and_sum() {
        let m = Mask32x4::from_array([true, true, false, false]);
        let n = Mask32x4::from_array([true, false, true, false]);
        assert_eq!((m & n).bitmask(), 0b0001);
        assert_eq!((m | n).bitmask(), 0b0111);
        assert_eq!((!m).bitmask(), 0b1100);
        assert!(m.any() && !m.all() && !Mask32x4::splat(false).any());
        assert!(Mask32x4::splat(true).all());
        let s = F32x4::select(n, F32x4::from_array(A), F32x4::from_array(B));
        assert_eq!(s.to_array()[..3], [1.5, 3.0, INF]);
        assert_eq!(s.to_array()[3], 1.0);
        assert_eq!(F32x4::new(1.0, 2.0, 4.0, 8.0).element_sum(), 15.0);
        assert!(F32x4::from_array(A).element_sum().is_nan());
        assert_eq!(F32x4::new(INF, 1.0, 0.0, 0.0).element_sum(), INF);
    }

    // lanes 4-7 differ from 0-3, so a swapped or duplicated half shows
    const A8: [f32; 8] = [1.5, -2.0, INF, f32::NAN, -0.0, 4.0, f32::NAN, 9.0];
    const B8: [f32; 8] = [0.5, 3.0, -INF, 1.0, 0.0, f32::NAN, 2.0, -9.0];

    fn assert_lanes8(got: F32x8, f: impl Fn(f32, f32) -> f32) {
        for (i, g) in got.to_array().into_iter().enumerate() {
            let want = f(A8[i], B8[i]);
            assert!(
                g.to_bits() == want.to_bits() || (g.is_nan() && want.is_nan()),
                "lane {i}: {g} vs {want}"
            );
        }
    }

    fn mask_bits8(f: impl Fn(f32, f32) -> bool) -> u32 {
        (0..8).map(|i| (f(A8[i], B8[i]) as u32) << i).sum()
    }

    #[test]
    fn x8_ops_match_scalar() {
        let (a, b) = (F32x8::from_array(A8), F32x8::from_array(B8));
        assert_lanes8(a + b, |x, y| x + y);
        assert_lanes8(a - b, |x, y| x - y);
        assert_lanes8(a * b, |x, y| x * y);
        assert_lanes8(a / b, |x, y| x / y);
        assert_lanes8(-a, |x, _| -x);
        assert_lanes8(a * 0.5, |x, _| x * 0.5);
        assert_lanes8(a.abs(), |x, _| x.abs());
        assert_lanes8(a.sqrt(), |x, _| x.sqrt());
        assert_lanes8(a.mul_add(b, b), |x, y| x * y + y);
        // NaN in either lane gives rhs; min(-0.0, 0.0) is 0.0 (lane 4)
        assert_lanes8(a.min(b), |x, y| if x < y { x } else { y });
        assert_lanes8(a.max(b), |x, y| if x > y { x } else { y });
        let mut c = a;
        c += b;
        c -= b;
        c *= F32x8::ONE;
        assert_lanes8(c, |x, y| (x + y - y) * 1.0);
        assert_eq!(a.cmplt(b).bitmask(), mask_bits8(|x, y| x < y));
        assert_eq!(a.cmple(b).bitmask(), mask_bits8(|x, y| x <= y));
        assert_eq!(a.cmpgt(b).bitmask(), mask_bits8(|x, y| x > y));
        assert_eq!(a.cmpge(b).bitmask(), mask_bits8(|x, y| x >= y));
        assert_eq!(a.cmpeq(b).bitmask(), mask_bits8(|x, y| x == y));
        let nan = F32x8::splat(f32::NAN);
        assert_eq!(nan.cmpeq(nan).bitmask(), 0);
        assert!((!nan.cmple(nan)).all());
    }

    #[test]
    fn x8_lane_order_across_halves() {
        let src: [f32; 9] = std::array::from_fn(|i| i as f32);
        let v = F32x8::from_slice(&src).unwrap_or_default();
        assert_eq!(v.to_array(), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        assert_eq!(
            F32x8::new(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0).to_array(),
            v.to_array()
        );
        assert!(F32x8::from_slice(&src[..7]).is_none());
        let mut dst = [-1.0; 9];
        assert_eq!(v.write_to_slice(&mut dst), Some(()));
        assert_eq!(dst[..8], src[..8]);
        assert_eq!(dst[8], -1.0);
        assert!(v.write_to_slice(&mut dst[..7]).is_none());
        // bit i is lane i, both halves
        assert_eq!(v.cmpgt(F32x8::splat(2.5)).bitmask(), 0b1111_1000);
        assert_eq!(v.cmplt(F32x8::splat(5.5)).bitmask(), 0b0011_1111);
        let m = Mask32x8::from_array([true, false, false, false, false, true, false, true]);
        assert_eq!(m.bitmask(), 0b1010_0001);
        let lo = Mask32x8::from_array([true, false, false, false, false, false, false, false]);
        let hi = Mask32x8::from_array([false, false, false, false, false, false, false, true]);
        assert_eq!((lo | hi).bitmask(), 0b1000_0001);
        assert_eq!((m & hi).bitmask(), 0b1000_0000);
        assert_eq!((!m).bitmask(), 0b0101_1110);
        assert!(lo.any() && hi.any() && !Mask32x8::splat(false).any());
        assert!(!(lo | hi).all() && Mask32x8::splat(true).all());
        let s = F32x8::select(m, v, -v).to_array();
        assert_eq!(s, [0.0, -1.0, -2.0, -3.0, -4.0, 5.0, -6.0, 7.0]);
        assert_eq!(v.element_sum(), 28.0);
        assert!(F32x8::from_array(A8).element_sum().is_nan());
    }

    #[test]
    fn x8_min_max_keep_rhs_on_nan() {
        let (a, b) = (F32x8::from_array(A8), F32x8::from_array(B8));
        // lane 3: NaN in lhs gives rhs; lane 5: NaN in rhs is kept
        assert_eq!(a.min(b).to_array()[3], 1.0);
        assert_eq!(a.max(b).to_array()[3], 1.0);
        assert!(a.min(b).to_array()[5].is_nan() && a.max(b).to_array()[5].is_nan());
        assert!(b.min(a).to_array()[3].is_nan());
        assert_eq!(b.max(a).to_array()[5], 4.0);
        // min(0.0, -0.0) is the rhs -0.0
        assert!(b.min(a).to_array()[4].is_sign_negative());
    }

    /// The AVX2 backend and the two-halves backend give the same bits, NaN matching NaN.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[test]
    fn x8_avx2_matches_halves() {
        use super::{x8_avx2 as v, x8_halves as h};
        let same = |got: [f32; 8], want: [f32; 8], op: &str| {
            for (i, (g, w)) in got.into_iter().zip(want).enumerate() {
                assert!(
                    g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan()),
                    "{op} lane {i}: {g} vs {w}"
                );
            }
        };
        // finite values with inexact sums and products, so rounding order shows
        let mut seed = 0x1234_5678u32;
        let mut rand8 = || {
            std::array::from_fn(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (seed >> 8) as f32 / (1 << 20) as f32 - 8.0
            })
        };
        let (r0, r1) = (rand8(), rand8());
        for (a, b) in [(A8, B8), (B8, A8), (r0, r1), (r1, A8)] {
            let (va, vb, ha, hb) = (
                v::from_array(a),
                v::from_array(b),
                h::from_array(a),
                h::from_array(b),
            );
            macro_rules! check {
                ($($f:ident),*) => {$(
                    same(v::to_array(v::$f(va, vb)), h::to_array(h::$f(ha, hb)), stringify!($f));
                )*};
            }
            check!(add, sub, mul, div, min, max);
            same(v::to_array(v::neg(va)), h::to_array(h::neg(ha)), "neg");
            same(v::to_array(v::abs(va)), h::to_array(h::abs(ha)), "abs");
            same(v::to_array(v::sqrt(va)), h::to_array(h::sqrt(ha)), "sqrt");
            same(
                v::to_array(v::scale(va, 0.3)),
                h::to_array(h::scale(ha, 0.3)),
                "scale",
            );
            let (vm, hm) = (v::mul_add(va, vb, va), h::mul_add(ha, hb, ha));
            same(v::to_array(vm), h::to_array(hm), "mul_add");
            let (vs, hs) = (v::element_sum(va), h::element_sum(ha));
            assert!(vs.to_bits() == hs.to_bits() || (vs.is_nan() && hs.is_nan()));
            macro_rules! check_mask {
                ($($f:ident),*) => {$(
                    let (vm, hm) = (v::$f(va, vb), h::$f(ha, hb));
                    assert_eq!(v::bitmask(vm), h::bitmask(hm), stringify!($f));
                    assert_eq!(v::bitmask(v::not(vm)), h::bitmask(h::not(hm)));
                    assert_eq!((v::any(vm), v::all(vm)), (h::any(hm), h::all(hm)));
                    let (vn, hn) = (v::cmplt(vb, va), h::cmplt(hb, ha));
                    assert_eq!(v::bitmask(v::and(vm, vn)), h::bitmask(h::and(hm, hn)));
                    assert_eq!(v::bitmask(v::or(vm, vn)), h::bitmask(h::or(hm, hn)));
                    same(v::to_array(v::select(vm, va, vb)), h::to_array(h::select(hm, ha, hb)), "select");
                )*};
            }
            check_mask!(cmplt, cmple, cmpgt, cmpge, cmpeq);
        }
        let l = [true, false, true, true, false, false, true, false];
        assert_eq!(
            v::bitmask(v::mask_from_array(l)),
            h::bitmask(h::mask_from_array(l))
        );
    }
}
