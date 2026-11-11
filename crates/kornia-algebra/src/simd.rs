//! Lane-batched `f32` types for structure-of-arrays kernels.
//!
//! [`F32x4`] holds four `f32` lanes and [`Mask32x4`] four boolean lanes. Both wrap glam's
//! `Vec4` and `BVec4A`, so they lower to SSE2 on x86-64 and NEON on aarch64 with no `unsafe`
//! here and no extra dependency. Each is a `#[repr(transparent)]` newtype that stores the glam
//! register itself, so operators act on it directly with no per-op conversion.
//!
//! [`F32x8`] and [`Mask32x8`] are eight lanes held as two four-lane halves, lanes 0-3 then
//! 4-7, with the same API and NaN rules. Two 128-bit ops per op cost less per point than one
//! [`F32x4`] op does, because a kernel's per-chunk fixed costs (loop, bitmask branch, splats)
//! are paid once per eight lanes.
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

/// Eight `f32` lanes as two [`F32x4`] halves; lane `i` is element `i` of its source.
///
/// Every op runs on both halves, with the semantics of the [`F32x4`] op of the same name.
#[derive(Debug, Clone, Copy, Default)]
#[repr(transparent)]
pub struct F32x8([F32x4; 2]);

/// Eight boolean lanes, the result of an [`F32x8`] comparison.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct Mask32x8([Mask32x4; 2]);

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
        Self([F32x4::new(l0, l1, l2, l3), F32x4::new(l4, l5, l6, l7)])
    }

    /// Sets every lane to `v`.
    #[inline]
    pub const fn splat(v: f32) -> Self {
        Self([F32x4::splat(v); 2])
    }

    /// Loads eight lanes from an array; lane `i` is `src[i]`.
    #[inline]
    pub const fn from_array(s: [f32; 8]) -> Self {
        Self::new(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7])
    }

    /// Returns the lanes as an array; element `i` is lane `i`.
    #[inline]
    pub const fn to_array(self) -> [f32; 8] {
        let (a, b) = (self.0[0].to_array(), self.0[1].to_array());
        [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]]
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

    #[inline(always)]
    fn map(self, f: impl Fn(F32x4) -> F32x4) -> Self {
        Self([f(self.0[0]), f(self.0[1])])
    }

    #[inline(always)]
    fn zip(self, rhs: Self, f: impl Fn(F32x4, F32x4) -> F32x4) -> Self {
        Self([f(self.0[0], rhs.0[0]), f(self.0[1], rhs.0[1])])
    }

    #[inline(always)]
    fn cmp(self, rhs: Self, f: impl Fn(F32x4, F32x4) -> Mask32x4) -> Mask32x8 {
        Mask32x8([f(self.0[0], rhs.0[0]), f(self.0[1], rhs.0[1])])
    }

    /// `self * a + b`, fused where [`F32x4::mul_add`] is.
    #[inline]
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        Self([
            self.0[0].mul_add(a.0[0], b.0[0]),
            self.0[1].mul_add(a.0[1], b.0[1]),
        ])
    }

    /// Lane-wise square root; negative lanes give NaN.
    #[inline]
    pub fn sqrt(self) -> Self {
        self.map(F32x4::sqrt)
    }

    /// Lane-wise absolute value; NaN stays NaN.
    #[inline]
    pub fn abs(self) -> Self {
        self.map(F32x4::abs)
    }

    /// Lane-wise minimum; NaN in either lane gives `rhs`, as for [`F32x4::min`].
    #[inline]
    pub fn min(self, rhs: Self) -> Self {
        self.zip(rhs, F32x4::min)
    }

    /// Lane-wise maximum; NaN in either lane gives `rhs`, as for [`F32x4::max`].
    #[inline]
    pub fn max(self, rhs: Self) -> Self {
        self.zip(rhs, F32x4::max)
    }

    /// Sum of the eight lanes: the halves are added lane-wise, then that four-lane sum is
    /// reduced as [`F32x4::element_sum`].
    #[inline]
    pub fn element_sum(self) -> f32 {
        (self.0[0] + self.0[1]).element_sum()
    }

    /// Lane-wise `self < rhs`; `false` where either lane is NaN.
    #[inline]
    pub fn cmplt(self, rhs: Self) -> Mask32x8 {
        self.cmp(rhs, F32x4::cmplt)
    }

    /// Lane-wise `self <= rhs`; `false` where either lane is NaN.
    #[inline]
    pub fn cmple(self, rhs: Self) -> Mask32x8 {
        self.cmp(rhs, F32x4::cmple)
    }

    /// Lane-wise `self > rhs`; `false` where either lane is NaN.
    #[inline]
    pub fn cmpgt(self, rhs: Self) -> Mask32x8 {
        self.cmp(rhs, F32x4::cmpgt)
    }

    /// Lane-wise `self >= rhs`; `false` where either lane is NaN.
    #[inline]
    pub fn cmpge(self, rhs: Self) -> Mask32x8 {
        self.cmp(rhs, F32x4::cmpge)
    }

    /// Lane-wise `self == rhs`; `false` where either lane is NaN, `true` for `-0.0 == 0.0`.
    #[inline]
    pub fn cmpeq(self, rhs: Self) -> Mask32x8 {
        self.cmp(rhs, F32x4::cmpeq)
    }

    /// Takes each lane from `if_true` where `mask` is set, else from `if_false`.
    #[inline]
    pub fn select(mask: Mask32x8, if_true: Self, if_false: Self) -> Self {
        Self([
            F32x4::select(mask.0[0], if_true.0[0], if_false.0[0]),
            F32x4::select(mask.0[1], if_true.0[1], if_false.0[1]),
        ])
    }
}

impl Mask32x8 {
    /// Sets every lane to `v`.
    #[inline]
    pub const fn splat(v: bool) -> Self {
        Self([Mask32x4::splat(v); 2])
    }

    /// Builds a mask from its lanes; lane `i` is `lanes[i]`.
    #[inline]
    pub const fn from_array(l: [bool; 8]) -> Self {
        Self([
            Mask32x4::from_array([l[0], l[1], l[2], l[3]]),
            Mask32x4::from_array([l[4], l[5], l[6], l[7]]),
        ])
    }

    /// Bit `i` is lane `i`; the upper 24 bits are zero.
    #[inline]
    pub fn bitmask(self) -> u32 {
        self.0[0].bitmask() | (self.0[1].bitmask() << 4)
    }

    /// `true` if any lane is set.
    #[inline]
    pub fn any(self) -> bool {
        (self.0[0] | self.0[1]).any()
    }

    /// `true` if every lane is set.
    #[inline]
    pub fn all(self) -> bool {
        (self.0[0] & self.0[1]).all()
    }
}

macro_rules! impl_binop_x8 {
    ($op:ident, $f:ident, $op_assign:ident, $f_assign:ident) => {
        impl $op for F32x8 {
            type Output = Self;
            #[inline]
            fn $f(self, rhs: Self) -> Self {
                self.zip(rhs, F32x4::$f)
            }
        }
        impl $op_assign for F32x8 {
            #[inline]
            fn $f_assign(&mut self, rhs: Self) {
                self.0[0].$f_assign(rhs.0[0]);
                self.0[1].$f_assign(rhs.0[1]);
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
        self.zip(rhs, F32x4::div)
    }
}

/// Multiplies every lane by a scalar, each half as `F32x4 * f32`.
impl Mul<f32> for F32x8 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f32) -> Self {
        self.map(|h| h * rhs)
    }
}

impl Neg for F32x8 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        self.map(F32x4::neg)
    }
}

impl BitAnd for Mask32x8 {
    type Output = Self;
    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self([self.0[0] & rhs.0[0], self.0[1] & rhs.0[1]])
    }
}

impl BitOr for Mask32x8 {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self([self.0[0] | rhs.0[0], self.0[1] | rhs.0[1]])
    }
}

impl Not for Mask32x8 {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self([!self.0[0], !self.0[1]])
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
}
