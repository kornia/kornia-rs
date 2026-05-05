//! # Nistér 5-point essential-matrix solver
//!
//! Recovers up to 10 candidate essential matrices from 5 normalized (metric)
//! point correspondences. The classic 8-point algorithm recovers a fundamental
//! matrix in pixel space and projects onto the essential manifold *after*
//! decomposition; that final projection loses structure because the (σ, σ, 0)
//! singular profile is enforced by clipping rather than by construction. The
//! 5-point method stays on the essential manifold throughout, so it gives
//! markedly better rotation recovery on small-motion or narrow-baseline pairs.
//!
//! ## Algorithm (Nistér 2004, Stewenius-style formulation)
//!
//! 1. **Null-space**: each correspondence `x̂2ᵀ E x̂1 = 0` is a linear
//!    constraint in the 9 entries of E. Five correspondences give a 5×9
//!    matrix whose null-space is 4-dimensional. Write
//!    `E(x, y, z, w) = x·X + y·Y + z·Z + w·W` for four basis vec9s `X,Y,Z,W`.
//! 2. **Essential constraints**: any valid E must satisfy
//!    (a) `det(E) = 0` — one cubic in `(x, y, z, w)`.
//!    (b) `2 E Eᵀ E − tr(E Eᵀ) E = 0` — nine cubics (matrix equation).
//!    Setting `w = 1` without loss of generality leaves 10 cubics in
//!    `(x, y, z)`. Each cubic has 20 monomials of degree ≤ 3, yielding a
//!    10×20 coefficient matrix.
//! 3. **Elimination**: Gauss-Jordan reduces the matrix so that the final
//!    rows' leading monomials are all in z only. Rearranging produces a
//!    10×10 polynomial matrix in z whose determinant is a degree-10
//!    univariate polynomial.
//! 4. **Action matrix**: build the 10×10 matrix `M_z` of multiplication-by-z
//!    in the quotient-algebra basis `{1, x, y, z, x², xy, xz, y², yz, z²}`.
//!    Every solution of the system appears as a **right eigenvector** of
//!    `M_z` with eigenvalue = the solution's `z*` coordinate.
//! 5. **Eigendecomposition**: Householder reduction to upper Hessenberg,
//!    then Wilkinson-shifted QR iteration (operating on the matrix
//!    directly, **never** forming its characteristic polynomial — the
//!    coefficients can span 10³² in magnitude in bad parameterizations,
//!    which Faddeev-Leverrier cannot survive in f64). Each real eigenvalue
//!    is a candidate `z*`.
//! 6. **Back-substitute**: for each real `z*`, recover `(x, y)` from a
//!    single row of the eigenvector of `M_z − z*·I` (fix `v[0] = 1`, solve
//!    the bottom 9×9).
//! 7. **Assemble**: `E = x·X + y·Y + z·Z + W` for each real solution.
//!
//! ## Monomial ordering
//!
//! The 20 monomials of degree ≤ 3 in three variables are indexed below.
//! This order is used consistently across [`Poly3D`] multiplication and the
//! 10×20 constraint-matrix rows.
//!
//! | idx | monomial |
//! |-----|----------|
//! | 0   | 1        |
//! | 1   | x        |
//! | 2   | y        |
//! | 3   | z        |
//! | 4   | x²       |
//! | 5   | xy       |
//! | 6   | xz       |
//! | 7   | y²       |
//! | 8   | yz       |
//! | 9   | z²       |
//! | 10  | x³       |
//! | 11  | x²y      |
//! | 12  | x²z      |
//! | 13  | xy²      |
//! | 14  | xyz      |
//! | 15  | xz²      |
//! | 16  | y³       |
//! | 17  | y²z      |
//! | 18  | yz²      |
//! | 19  | z³       |
//!
//! ## When to prefer 5-point
//!
//! - **Few inliers**: RANSAC sample probability goes as `w^s` where `s` is
//!   sample size. At 30% inlier rate, 5-point needs `log(0.01)/log(1-0.3^5) ≈
//!   568` iterations vs 8-point's `4600` for 99% confidence — 8× fewer draws.
//! - **Low parallax / small motion**: the 8-point F → E projection can leave
//!   significant rotation error when the baseline is short; 5-point's exact
//!   manifold parameterization shaves the rotation error by 2-10× on SLAM
//!   bootstrap pairs.
//! - **Always prefer over 7-point** when you have calibrated intrinsics: K is
//!   already known, so there's no reason to pay the extra F-space DOF.

#![allow(clippy::needless_range_loop)]

use kornia_algebra::{Mat3F64, Vec2F64, Vec3F64};

// ──────────────────────────────────────────────────────────────────────────
// NEON micro-kernels for hot inner loops. The 5-point solver is called once
// per RANSAC draw (hundreds of times per pose estimate), so every μs in
// build_constraint_matrix / Gauss-Jordan / QR iteration pays back at the
// loop level. Scalar fallbacks live alongside each aarch64 path so the file
// still builds (and stays testable) on x86 dev boxes.
// ──────────────────────────────────────────────────────────────────────────

/// `dst[k] += src[k]` for k in 0..20. NEON-vectorized: 10 `vaddq_f64` on
/// aarch64, tight scalar loop elsewhere. Small (80-byte) rows so alignment
/// and prefetching don't matter.
#[inline(always)]
fn row20_add_assign(dst: &mut [f64; 20], src: &[f64; 20]) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let dp = dst.as_mut_ptr();
        let sp = src.as_ptr();
        let mut k = 0usize;
        while k < 20 {
            let a = vld1q_f64(dp.add(k));
            let b = vld1q_f64(sp.add(k));
            vst1q_f64(dp.add(k), vaddq_f64(a, b));
            k += 2;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if kornia_imgproc::simd::cpu_features().has_avx2 {
            unsafe { row20_add_assign_avx2(dst, src) };
            return;
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for k in 0..20 {
            dst[k] += src[k];
        }
    }
}

/// `dst[k] -= src[k]` for k in 0..20 (NEON-vectorized; see [`row20_add_assign`]).
#[inline(always)]
fn row20_sub_assign(dst: &mut [f64; 20], src: &[f64; 20]) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let dp = dst.as_mut_ptr();
        let sp = src.as_ptr();
        let mut k = 0usize;
        while k < 20 {
            let a = vld1q_f64(dp.add(k));
            let b = vld1q_f64(sp.add(k));
            vst1q_f64(dp.add(k), vsubq_f64(a, b));
            k += 2;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if kornia_imgproc::simd::cpu_features().has_avx2 {
            unsafe { row20_sub_assign_avx2(dst, src) };
            return;
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for k in 0..20 {
            dst[k] -= src[k];
        }
    }
}

/// `out[k] = s * src[k]` for k in 0..20 (NEON-vectorized).
#[inline(always)]
fn row20_scale(src: &[f64; 20], s: f64) -> [f64; 20] {
    let mut out = [0.0f64; 20];
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let op = out.as_mut_ptr();
        let sp = src.as_ptr();
        let ss = vdupq_n_f64(s);
        let mut k = 0usize;
        while k < 20 {
            let a = vld1q_f64(sp.add(k));
            vst1q_f64(op.add(k), vmulq_f64(a, ss));
            k += 2;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if kornia_imgproc::simd::cpu_features().has_avx2 {
            unsafe { row20_scale_avx2(&mut out, src, s) };
            return out;
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for k in 0..20 {
            out[k] = src[k] * s;
        }
    }
    out
}

/// `dst[k] -= factor * src[k]` for k in 0..20 via `vfmaq_f64` (fused
/// multiply-subtract). Single-rounded on A78AE, 1/cycle throughput.
#[inline(always)]
fn row20_fma_sub(dst: &mut [f64; 20], src: &[f64; 20], factor: f64) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let dp = dst.as_mut_ptr();
        let sp = src.as_ptr();
        let neg_f = vdupq_n_f64(-factor);
        let mut k = 0usize;
        while k < 20 {
            let d = vld1q_f64(dp.add(k));
            let s = vld1q_f64(sp.add(k));
            // d + (-factor) * s = d - factor * s
            vst1q_f64(dp.add(k), vfmaq_f64(d, s, neg_f));
            k += 2;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if kornia_imgproc::simd::cpu_features().has_avx2 {
            unsafe { row20_fma_sub_avx2(dst, src, factor) };
            return;
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for k in 0..20 {
            dst[k] -= factor * src[k];
        }
    }
}

/// AVX2 row-20 helpers — 4-lane f64 (`__m256d`), 5 unrolled iterations cover
/// the full 20 lanes (20 = 5 × 4). One factored helper per op so each
/// dispatch site can return early on the AVX2 path; scalar fallback covers
/// non-AVX2 x86_64.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn row20_add_assign_avx2(dst: &mut [f64; 20], src: &[f64; 20]) {
    use std::arch::x86_64::*;
    let dp = dst.as_mut_ptr();
    let sp = src.as_ptr();
    let mut k = 0usize;
    while k < 20 {
        let a = _mm256_loadu_pd(dp.add(k));
        let b = _mm256_loadu_pd(sp.add(k));
        _mm256_storeu_pd(dp.add(k), _mm256_add_pd(a, b));
        k += 4;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn row20_sub_assign_avx2(dst: &mut [f64; 20], src: &[f64; 20]) {
    use std::arch::x86_64::*;
    let dp = dst.as_mut_ptr();
    let sp = src.as_ptr();
    let mut k = 0usize;
    while k < 20 {
        let a = _mm256_loadu_pd(dp.add(k));
        let b = _mm256_loadu_pd(sp.add(k));
        _mm256_storeu_pd(dp.add(k), _mm256_sub_pd(a, b));
        k += 4;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn row20_scale_avx2(out: &mut [f64; 20], src: &[f64; 20], s: f64) {
    use std::arch::x86_64::*;
    let op = out.as_mut_ptr();
    let sp = src.as_ptr();
    let ss = _mm256_set1_pd(s);
    let mut k = 0usize;
    while k < 20 {
        let a = _mm256_loadu_pd(sp.add(k));
        _mm256_storeu_pd(op.add(k), _mm256_mul_pd(a, ss));
        k += 4;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn row20_fma_sub_avx2(dst: &mut [f64; 20], src: &[f64; 20], factor: f64) {
    use std::arch::x86_64::*;
    let dp = dst.as_mut_ptr();
    let sp = src.as_ptr();
    let fv = _mm256_set1_pd(factor);
    let mut k = 0usize;
    while k < 20 {
        let d = _mm256_loadu_pd(dp.add(k));
        let s = _mm256_loadu_pd(sp.add(k));
        // d - factor * s == fnmadd(factor, s, d)
        _mm256_storeu_pd(dp.add(k), _mm256_fnmadd_pd(fv, s, d));
        k += 4;
    }
}

/// Rotate two rows of a 10×10 matrix by a Givens factor `(c, s)` over a
/// variable-length column span `[start..end)`:
///   h_k[j]  ← c·h_k[j] + s·h_{k+1}[j]
///   h_{k+1}[j] ← −s·h_k[j] + c·h_{k+1}[j]
/// The QR step applies ~n of these per iteration and iterates ~30-50 times
/// per 10×10 eigendecomp, so this pair-rotate is the densest hotspot.
///
/// NEON: loads two f64 from each row into `float64x2_t`, computes the two
/// outputs with `vfmaq_f64`, stores. Scalar tail for odd-length spans.
#[inline(always)]
fn givens_row_pair_10(
    h: &mut [[f64; 10]; 10],
    row_k: usize,
    start: usize,
    end: usize,
    c: f64,
    s: f64,
) {
    debug_assert!(row_k + 1 < 10 && end <= 10 && start <= end);
    // Safely disjoint-borrow the two rows.
    let (top, bot) = {
        let (hi, lo) = h.split_at_mut(row_k + 1);
        (&mut hi[row_k], &mut lo[0])
    };

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let tp = top.as_mut_ptr();
        let bp = bot.as_mut_ptr();
        let cv = vdupq_n_f64(c);
        let sv = vdupq_n_f64(s);
        let neg_sv = vdupq_n_f64(-s);
        let mut j = start;
        while j + 2 <= end {
            let x = vld1q_f64(tp.add(j));
            let y = vld1q_f64(bp.add(j));
            // new_x = c*x + s*y
            let nx = vfmaq_f64(vmulq_f64(x, cv), y, sv);
            // new_y = -s*x + c*y
            let ny = vfmaq_f64(vmulq_f64(y, cv), x, neg_sv);
            vst1q_f64(tp.add(j), nx);
            vst1q_f64(bp.add(j), ny);
            j += 2;
        }
        while j < end {
            let x = *tp.add(j);
            let y = *bp.add(j);
            *tp.add(j) = c * x + s * y;
            *bp.add(j) = -s * x + c * y;
            j += 1;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if kornia_imgproc::simd::cpu_features().has_avx2 {
            unsafe { givens_row_pair_10_avx2(top, bot, start, end, c, s) };
            return;
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for j in start..end {
            let x = top[j];
            let y = bot[j];
            top[j] = c * x + s * y;
            bot[j] = -s * x + c * y;
        }
    }
}

/// AVX2 mirror of the Givens row-pair update. 4-lane f64 batch + scalar
/// tail. `_mm256_fmadd_pd(c, x, mul(s, y))` form covers the
/// `c·x + s·y` step in one rounded operation; same shape as the NEON
/// `vfmaq_f64(vmulq_f64(...), ..., ...)` chain.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn givens_row_pair_10_avx2(
    top: &mut [f64],
    bot: &mut [f64],
    start: usize,
    end: usize,
    c: f64,
    s: f64,
) {
    use std::arch::x86_64::*;
    let tp = top.as_mut_ptr();
    let bp = bot.as_mut_ptr();
    let cv = _mm256_set1_pd(c);
    let sv = _mm256_set1_pd(s);
    let neg_sv = _mm256_set1_pd(-s);
    let mut j = start;
    while j + 4 <= end {
        let x = _mm256_loadu_pd(tp.add(j));
        let y = _mm256_loadu_pd(bp.add(j));
        // new_x = c*x + s*y
        let nx = _mm256_fmadd_pd(y, sv, _mm256_mul_pd(x, cv));
        // new_y = -s*x + c*y
        let ny = _mm256_fmadd_pd(x, neg_sv, _mm256_mul_pd(y, cv));
        _mm256_storeu_pd(tp.add(j), nx);
        _mm256_storeu_pd(bp.add(j), ny);
        j += 4;
    }
    while j < end {
        let x = *tp.add(j);
        let y = *bp.add(j);
        *tp.add(j) = c * x + s * y;
        *bp.add(j) = -s * x + c * y;
        j += 1;
    }
}

/// Degrees `[a, b, c]` of each monomial `xᵃ yᵇ zᶜ` in our 20-element basis.
///
/// Indexed by the monomial order declared in the module docs. Used by
/// [`Poly3D::mul`] to look up the product-monomial index.
const DEGREES: [[u8; 3]; 20] = [
    [0, 0, 0], // 1
    [1, 0, 0], // x
    [0, 1, 0], // y
    [0, 0, 1], // z
    [2, 0, 0], // x²
    [1, 1, 0], // xy
    [1, 0, 1], // xz
    [0, 2, 0], // y²
    [0, 1, 1], // yz
    [0, 0, 2], // z²
    [3, 0, 0], // x³
    [2, 1, 0], // x²y
    [2, 0, 1], // x²z
    [1, 2, 0], // xy²
    [1, 1, 1], // xyz
    [1, 0, 2], // xz²
    [0, 3, 0], // y³
    [0, 2, 1], // y²z
    [0, 1, 2], // yz²
    [0, 0, 3], // z³
];

/// Dense polynomial of total degree ≤ 3 in 3 variables (`x, y, z`).
///
/// Stored as 20 f64 coefficients in the order declared by [`DEGREES`].
/// Enough to express every polynomial that appears in Nistér's 10 cubic
/// constraints — the input bases are degree-1 and all products stay ≤ 3.
#[derive(Clone, Copy, Debug, Default)]
struct Poly3D {
    c: [f64; 20],
}

impl Poly3D {
    const ZERO: Poly3D = Poly3D { c: [0.0; 20] };

    /// Degree-1 polynomial `a·x + b·y + c·z + d`.
    fn linear(a: f64, b: f64, c: f64, d: f64) -> Self {
        let mut p = Self::ZERO;
        p.c[0] = d;
        p.c[1] = a;
        p.c[2] = b;
        p.c[3] = c;
        p
    }

    #[inline(always)]
    fn add_assign(&mut self, other: &Poly3D) {
        row20_add_assign(&mut self.c, &other.c);
    }

    #[inline(always)]
    fn sub_assign(&mut self, other: &Poly3D) {
        row20_sub_assign(&mut self.c, &other.c);
    }

    #[inline(always)]
    fn scale(&self, s: f64) -> Poly3D {
        Poly3D {
            c: row20_scale(&self.c, s),
        }
    }

    /// Polynomial product, silently dropping terms of degree > 3.
    /// For Nistér's 10 constraints every intermediate product stays ≤ 3,
    /// so the drop never fires on valid input.
    fn mul(&self, other: &Poly3D) -> Poly3D {
        let mut out = Poly3D::ZERO;
        for i in 0..20 {
            let a = self.c[i];
            if a == 0.0 {
                continue;
            }
            let di = DEGREES[i];
            for j in 0..20 {
                let b = other.c[j];
                if b == 0.0 {
                    continue;
                }
                let dj = DEGREES[j];
                let d = [di[0] + dj[0], di[1] + dj[1], di[2] + dj[2]];
                if d[0] + d[1] + d[2] > 3 {
                    continue;
                }
                // Linear search is fine — 20 entries, well-predicted.
                let mut k = 0;
                while k < 20 && DEGREES[k] != d {
                    k += 1;
                }
                if k < 20 {
                    out.c[k] += a * b;
                }
            }
        }
        out
    }
}

/// Compute the 4-dimensional null-space of the 5×9 epipolar constraint
/// matrix built from 5 normalized correspondences.
///
/// Returns `None` when the sample is degenerate (e.g. all five correspondences
/// lie on a common epipolar plane or are duplicated) so the null-space has
/// rank < 4. Each returned basis vector is a 9-tuple of row-major E entries
/// `[E₀₀, E₀₁, E₀₂, E₁₀, E₁₁, E₁₂, E₂₀, E₂₁, E₂₂]`.
fn null_space_5x9(x1: &[Vec2F64; 5], x2: &[Vec2F64; 5]) -> Option<[[f64; 9]; 4]> {
    // Build the 5×9 constraint matrix: row i encodes x̂2ᵀ E x̂1 = 0.
    // For x̂1 = (u1, v1, 1), x̂2 = (u2, v2, 1):
    //   [u1 u2, v1 u2, u2, u1 v2, v1 v2, v2, u1, v1, 1] · vec(E) = 0.
    let mut mat = faer::Mat::<f64>::zeros(5, 9);
    for i in 0..5 {
        let (u1, v1) = (x1[i].x, x1[i].y);
        let (u2, v2) = (x2[i].x, x2[i].y);
        mat.write(i, 0, u1 * u2);
        mat.write(i, 1, v1 * u2);
        mat.write(i, 2, u2);
        mat.write(i, 3, u1 * v2);
        mat.write(i, 4, v1 * v2);
        mat.write(i, 5, v2);
        mat.write(i, 6, u1);
        mat.write(i, 7, v1);
        mat.write(i, 8, 1.0);
    }

    // Null-space via SVD: the last 4 right singular vectors span ker(A)
    // when rank(A) = 5. We use the full-V SVD and grab columns 5..9.
    let svd = mat.svd();
    let v = svd.v(); // 9×9 orthogonal (real)

    // Sanity check: if the 5th singular value is very small the sample is
    // degenerate and the null-space rank is > 4. Reject.
    let s = svd.s_diagonal();
    let s4 = s[4].abs();
    let s0 = s[0].abs().max(1e-30);
    if s4 / s0 < 1e-8 {
        return None;
    }

    let mut basis = [[0.0f64; 9]; 4];
    for (k, col) in (5..9).enumerate() {
        for row in 0..9 {
            basis[k][row] = v[(row, col)];
        }
    }
    Some(basis)
}

/// View a 9-vec in row-major order as the 9 entries of a 3×3 matrix.
#[inline]
fn vec9_to_mat3(e: &[f64; 9]) -> Mat3F64 {
    // Mat3F64 is column-major via from_cols; entries e00, e01, e02, ...
    // become columns as (e00, e10, e20), (e01, e11, e21), (e02, e12, e22).
    Mat3F64::from_cols(
        Vec3F64::new(e[0], e[3], e[6]),
        Vec3F64::new(e[1], e[4], e[7]),
        Vec3F64::new(e[2], e[5], e[8]),
    )
}

/// Build the 10×20 cubic-constraint matrix whose rows encode the
/// essential-manifold identities
///   det(E(x,y,z,1)) = 0  and  2·E Eᵀ E − tr(E Eᵀ)·E = 0
/// in the 20 monomials of degree ≤ 3 in `(x, y, z)`.
fn build_constraint_matrix(null_basis: &[[f64; 9]; 4]) -> [[f64; 20]; 10] {
    // Each entry E[i][j] is a degree-1 polynomial in (x, y, z):
    //   E[i][j] = X[i][j]·x + Y[i][j]·y + Z[i][j]·z + W[i][j]·1.
    // The null-space basis is stored as row-major [e00, e01, e02, e10, ...]
    // so null_basis[k][3*i + j] is the k-th basis vector's (i, j) entry.
    let mut e_poly = [[Poly3D::ZERO; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let idx = 3 * i + j;
            e_poly[i][j] = Poly3D::linear(
                null_basis[0][idx],
                null_basis[1][idx],
                null_basis[2][idx],
                null_basis[3][idx],
            );
        }
    }

    // EEt[i][j] = Σ_k E[i][k] * E[j][k].
    let mut eet = [[Poly3D::ZERO; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = Poly3D::ZERO;
            for k in 0..3 {
                let prod = e_poly[i][k].mul(&e_poly[j][k]);
                s.add_assign(&prod);
            }
            eet[i][j] = s;
        }
    }

    // trace(E Eᵀ).
    let mut trace = Poly3D::ZERO;
    for i in 0..3 {
        trace.add_assign(&eet[i][i]);
    }
    let half_trace = trace.scale(0.5);

    // B[i][j] = (E Eᵀ - 0.5 tr(E Eᵀ) I) E ; the trace constraint is
    //   2 E Eᵀ E - tr(E Eᵀ) E = 0  ⇔  B = 0 (factor of 2 divided out).
    let mut b = [[Poly3D::ZERO; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = Poly3D::ZERO;
            for k in 0..3 {
                let mut m = eet[i][k];
                if i == k {
                    m.sub_assign(&half_trace);
                }
                let prod = m.mul(&e_poly[k][j]);
                s.add_assign(&prod);
            }
            b[i][j] = s;
        }
    }

    // det(E). Use cofactor expansion across row 0.
    let m01 = e_poly[1][1].mul(&e_poly[2][2]);
    let m02 = e_poly[1][2].mul(&e_poly[2][1]);
    let mut minor0 = m01;
    minor0.sub_assign(&m02);

    let m11 = e_poly[1][0].mul(&e_poly[2][2]);
    let m12 = e_poly[1][2].mul(&e_poly[2][0]);
    let mut minor1 = m11;
    minor1.sub_assign(&m12);

    let m21 = e_poly[1][0].mul(&e_poly[2][1]);
    let m22 = e_poly[1][1].mul(&e_poly[2][0]);
    let mut minor2 = m21;
    minor2.sub_assign(&m22);

    let t0 = e_poly[0][0].mul(&minor0);
    let t1 = e_poly[0][1].mul(&minor1);
    let t2 = e_poly[0][2].mul(&minor2);
    let mut det = t0;
    det.sub_assign(&t1);
    det.add_assign(&t2);

    // Stack the 10 constraints into rows of the 10×20 matrix. Row 0 is the
    // det; rows 1..10 are the nine entries of B in row-major order. Other
    // orderings work; Nistér's paper picks a specific one to make the
    // subsequent elimination structure clean. Our Gauss-Jordan below does
    // full row-reduction so the exact row order is irrelevant for
    // correctness (it can affect the numerics of ill-conditioned inputs).
    let mut mat = [[0.0f64; 20]; 10];
    for k in 0..20 {
        mat[0][k] = det.c[k];
    }
    for i in 0..3 {
        for j in 0..3 {
            let row = 1 + 3 * i + j;
            for k in 0..20 {
                mat[row][k] = b[i][j].c[k];
            }
        }
    }
    mat
}

/// In-place Gauss-Jordan reduction of a 10×20 matrix, pivoting on columns
/// 10..20 (the 10 degree-3 monomial columns). After success, the rightmost
/// 10×10 block equals the identity and row `i` expresses the (10+i)-th
/// monomial (`x³, x²y, x²z, xy², xyz, xz², y³, y²z, yz², z³`) as
/// `-Σ_j m[i][j] · B[j]` where `B = {1, x, y, z, x², xy, xz, y², yz, z²}` is
/// the quotient-algebra basis.
///
/// Pivoting on the degree-3 columns (rather than the degree-≤-2 ones) is
/// what exposes the multiplication-by-z action matrix: six of the degree-3
/// monomials are exactly `z·B[k]` for `k ∈ {4..10}`, so the reduction rows
/// hand us those columns of `M_z` directly.
///
/// Uses partial pivoting. Returns `false` when the 10×10 sub-block spanning
/// columns 10..20 is singular (degenerate 5-point sample).
fn gauss_jordan_eliminate_deg3(m: &mut [[f64; 20]; 10]) -> bool {
    for i in 0..10 {
        let col = 10 + i;
        // Partial pivot: find row in [i..10] with max |m[r][col]|.
        let mut piv = i;
        let mut piv_abs = m[i][col].abs();
        for r in (i + 1)..10 {
            let a = m[r][col].abs();
            if a > piv_abs {
                piv_abs = a;
                piv = r;
            }
        }
        if piv_abs < 1e-12 {
            return false;
        }
        if piv != i {
            m.swap(i, piv);
        }

        // Scale pivot row so m[i][col] = 1.
        let inv = 1.0 / m[i][col];
        m[i] = row20_scale(&m[i], inv);

        // Eliminate column `col` in every other row via NEON fused
        // multiply-subtract. Snapshot the pivot row (80 B) so we can
        // borrow `m[r]` mutably without fighting the borrow checker.
        let pivot = m[i];
        for r in 0..10 {
            if r == i {
                continue;
            }
            let factor = m[r][col];
            if factor == 0.0 {
                continue;
            }
            row20_fma_sub(&mut m[r], &pivot, factor);
        }
    }
    true
}

/// Assemble `E = x·X + y·Y + z·Z + W` from the null-space basis and the
/// recovered `(x, y, z)`.
fn assemble_e(null_basis: &[[f64; 9]; 4], x: f64, y: f64, z: f64) -> Mat3F64 {
    let mut e = [0.0f64; 9];
    for k in 0..9 {
        e[k] =
            x * null_basis[0][k] + y * null_basis[1][k] + z * null_basis[2][k] + null_basis[3][k];
    }
    vec9_to_mat3(&e)
}

/// Build the 10×10 action matrix of multiplication-by-z in the quotient
/// algebra basis `B = {1, x, y, z, x², xy, xz, y², yz, z²}`.
///
/// Convention: row `i` holds `z · B[i]` expanded in the basis, i.e.
/// `(M_z)[i][j] = coefficient of B[j] in the reduction of z · B[i]`. With
/// this orientation, the vector `v = (B_0(p), B_1(p), ..., B_9(p))` =
/// `(1, x*, y*, z*, x*², x*y*, x*z*, y*², y*z*, z*²)` is a **right**
/// eigenvector of `M_z` with eigenvalue `z*` at every solution `p = (x*, y*, z*)`
/// of the polynomial system.
///
/// `cm_reduced` must be the output of [`gauss_jordan_eliminate_deg3`] — its
/// right-hand 10×10 block is the identity and row `r` encodes
/// `deg3_monomial_{10+r} = -Σ_j cm_reduced[r][j] · B[j]`.
///
/// Rows 0-3 are permutation-like (`z·1 = z = B[3]`, `z·x = xz = B[6]`,
/// `z·y = yz = B[8]`, `z·z = z² = B[9]`). Rows 4-9 copy the reduction rows
/// for `x²z, xyz, xz², y²z, yz², z³` (monomial indices 12, 14, 15, 17, 18,
/// 19 — rows 2, 4, 5, 7, 8, 9 post-GJ).
fn build_action_matrix(cm_reduced: &[[f64; 20]; 10]) -> [[f64; 10]; 10] {
    let mut mz = [[0.0f64; 10]; 10];
    mz[0][3] = 1.0; // z · B[0] = z · 1 = z = B[3]
    mz[1][6] = 1.0; // z · B[1] = z · x = xz = B[6]
    mz[2][8] = 1.0; // z · B[2] = z · y = yz = B[8]
    mz[3][9] = 1.0; // z · B[3] = z · z = z² = B[9]
                    // Rows 4..10: z · B[i] is a degree-3 monomial for i = 4..10; read the
                    // GJ-produced reduction to express it in the quotient basis.
    let reduction_row: [usize; 6] = [2, 4, 5, 7, 8, 9];
    for (k, &r) in reduction_row.iter().enumerate() {
        let row = 4 + k;
        for j in 0..10 {
            mz[row][j] = -cm_reduced[r][j];
        }
    }
    mz
}

/// Reduce a 10×10 matrix to upper Hessenberg form via Householder similarity
/// transformations. Modifies `a` in place. Result: `a[i][j] = 0` for all
/// `i > j + 1`. Eigenvalues are preserved (it's an orthogonal similarity).
///
/// For the Nistér action matrix this is a 300-flop setup that makes the
/// subsequent QR iteration cost `O(n²)` per step instead of `O(n³)`.
fn hessenberg_reduce_10(a: &mut [[f64; 10]; 10]) {
    for k in 0..8 {
        // Build Householder vector v that zeroes a[k+2..10][k].
        let mut sigma = 0.0f64;
        for i in (k + 2)..10 {
            sigma += a[i][k] * a[i][k];
        }
        if sigma < 1e-28 {
            continue;
        }
        let alpha = a[k + 1][k];
        let mu = (alpha * alpha + sigma).sqrt();
        let d = if alpha >= 0.0 { alpha + mu } else { alpha - mu };
        let mut v = [0.0f64; 10];
        v[k + 1] = 1.0;
        let inv_d = 1.0 / d;
        for i in (k + 2)..10 {
            v[i] = a[i][k] * inv_d;
        }
        // beta = 2 / (vᵀv) = 2 / (1 + Σ_{i≥k+2} (a[i][k]/d)²) = 2 d² / (d² + σ).
        let beta = 2.0 * d * d / (d * d + sigma);

        // Left-apply H = I − β v vᵀ :  rows k+1..10 of columns k..10.
        for j in k..10 {
            let mut dot = a[k + 1][j];
            for i in (k + 2)..10 {
                dot += v[i] * a[i][j];
            }
            dot *= beta;
            a[k + 1][j] -= dot;
            for i in (k + 2)..10 {
                a[i][j] -= dot * v[i];
            }
        }
        // Right-apply H :  columns k+1..10 of all rows.
        for i in 0..10 {
            let mut dot = a[i][k + 1];
            for j in (k + 2)..10 {
                dot += v[j] * a[i][j];
            }
            dot *= beta;
            a[i][k + 1] -= dot;
            for j in (k + 2)..10 {
                a[i][j] -= dot * v[j];
            }
        }
    }
}

/// Eigenvalues of a 2×2 real matrix `[[a, b], [c, d]]`, returned as two
/// `(re, im)` pairs. Complex conjugate pairs carry `im = ±√discriminant`.
fn eigs_2x2(a: f64, b: f64, c: f64, d: f64) -> ((f64, f64), (f64, f64)) {
    let tr = a + d;
    let det = a * d - b * c;
    let disc = tr * tr / 4.0 - det;
    if disc >= 0.0 {
        let s = disc.sqrt();
        ((tr / 2.0 + s, 0.0), (tr / 2.0 - s, 0.0))
    } else {
        let s = (-disc).sqrt();
        ((tr / 2.0, s), (tr / 2.0, -s))
    }
}

/// One explicit-shift QR step on an active upper-Hessenberg sub-block
/// `h[0..n][0..n]` using Wilkinson shift. Applies Givens rotations to zero
/// the subdiagonal (Q-factor), then right-multiplies to re-Hessenberg-ify
/// (= RQ + μI). Touches only rows/columns 0..n.
fn qr_step_shifted(h: &mut [[f64; 10]; 10], n: usize) {
    if n < 2 {
        return;
    }
    // Wilkinson shift: eigenvalue of trailing 2×2 closer to h[n-1][n-1].
    let hnn = h[n - 1][n - 1];
    let p = (h[n - 2][n - 2] - hnn) * 0.5;
    let q = h[n - 1][n - 2] * h[n - 2][n - 1];
    let disc = p * p + q;
    let shift = if disc >= 0.0 {
        let r = disc.sqrt();
        let denom = if p >= 0.0 { p + r } else { p - r };
        if denom.abs() < 1e-300 {
            hnn
        } else {
            hnn - q / denom
        }
    } else {
        // Complex shift: use real part (single-shift variant; Francis double-shift
        // would be stricter, but this is sufficient for converging to the real
        // Schur form with 2×2 blocks for complex conjugate pairs).
        hnn - p
    };

    // Subtract shift.
    for i in 0..n {
        h[i][i] -= shift;
    }

    // Forward sweep: Givens rotations to triangularize H. Store (c, s) per
    // rotation on a stack so we can apply them from the right.
    let mut cs = [(0.0f64, 0.0f64); 9];
    for k in 0..(n - 1) {
        let a = h[k][k];
        let b = h[k + 1][k];
        let r = (a * a + b * b).sqrt();
        let (c, s) = if r < 1e-300 {
            (1.0, 0.0)
        } else {
            (a / r, b / r)
        };
        cs[k] = (c, s);
        // Apply Gᵀ to rows k, k+1 across columns k..n (NEON pair-rotate,
        // 2 columns per cycle). Dense hotspot — each QR iteration invokes
        // n-1 of these, and we iterate ~30-50 times per 10×10 eigendecomp.
        givens_row_pair_10(h, k, k, n, c, s);
    }

    // Backward sweep: apply each rotation from the right.
    for k in 0..(n - 1) {
        let (c, s) = cs[k];
        // Apply G to columns k, k+1 across rows 0..min(k+2, n).
        let lim = (k + 2).min(n);
        for i in 0..lim {
            let x = h[i][k];
            let y = h[i][k + 1];
            h[i][k] = c * x + s * y;
            h[i][k + 1] = -s * x + c * y;
        }
    }

    // Add shift back.
    for i in 0..n {
        h[i][i] += shift;
    }
}

/// Compute all 10 eigenvalues of a 10×10 real matrix via Hessenberg + shifted
/// QR, returning `(re, im)` pairs. Complex conjugate pairs come out of 2×2
/// diagonal blocks that don't fully deflate.
///
/// Operating on the matrix directly (rather than its characteristic
/// polynomial) sidesteps the catastrophic precision loss Faddeev-Leverrier
/// exhibits when the null-basis orientation makes Nistér's (x, y, z)
/// parameters blow up — coefficient magnitudes can span 10³² in that regime,
/// but the matrix itself stays moderately conditioned.
fn eigenvalues_10(a: &[[f64; 10]; 10]) -> [(f64, f64); 10] {
    let mut h = *a;
    hessenberg_reduce_10(&mut h);

    let mut eigs = [(0.0f64, 0.0f64); 10];
    let mut found = 0;
    let mut n = 10usize; // active sub-block is h[0..n][0..n]
    let mut iters = 0usize;
    const MAX_ITERS: usize = 500;

    while n > 0 && iters < MAX_ITERS {
        iters += 1;

        // Check for a small subdiagonal near the bottom to deflate.
        let mut split = 0usize;
        for k in (1..n).rev() {
            let sub = h[k][k - 1].abs();
            let diag = h[k - 1][k - 1].abs() + h[k][k].abs();
            if sub <= 1e-13 * diag.max(1e-300) {
                split = k;
                break;
            }
        }

        // Trailing 1×1 or 2×2 block can be read off directly.
        if split == n - 1 {
            eigs[found] = (h[n - 1][n - 1], 0.0);
            found += 1;
            n -= 1;
            continue;
        }
        if split == n - 2 {
            let (e1, e2) = eigs_2x2(
                h[n - 2][n - 2],
                h[n - 2][n - 1],
                h[n - 1][n - 2],
                h[n - 1][n - 1],
            );
            eigs[found] = e1;
            found += 1;
            eigs[found] = e2;
            found += 1;
            n -= 2;
            continue;
        }

        // Otherwise run a QR step on h[split..n][split..n]. We ignore `split`
        // and operate on h[0..n][0..n] — the rotations leave h[0..split] alone
        // because Givens on row split..split+1 only touches columns ≥ split in
        // upper-Hessenberg form.
        qr_step_shifted(&mut h, n);
    }

    // Edge case: iteration cap hit. Try reading the trailing 2×2 block just
    // to make progress, then fill the remaining eigs as zero sentinels.
    while n > 0 && found < 10 {
        if n == 1 {
            eigs[found] = (h[0][0], 0.0);
            found += 1;
            n -= 1;
        } else {
            let (e1, e2) = eigs_2x2(
                h[n - 2][n - 2],
                h[n - 2][n - 1],
                h[n - 1][n - 2],
                h[n - 1][n - 1],
            );
            eigs[found] = e1;
            found += 1;
            if found < 10 {
                eigs[found] = e2;
                found += 1;
            }
            n = n.saturating_sub(2);
        }
    }

    eigs
}

/// Extract the real eigenvalues (filtered by relative imaginary threshold)
/// of a 10×10 matrix.
fn real_eigenvalues_10(a: &[[f64; 10]; 10]) -> Vec<f64> {
    let all = eigenvalues_10(a);
    let mut out = Vec::new();
    for (re, im) in all.iter() {
        if im.abs() < 1e-8 * (1.0 + re.abs()) {
            out.push(*re);
        }
    }
    out
}

/// Solve a 9×9 linear system via Gaussian elimination with partial pivoting.
/// Returns `None` when the system is singular.
fn solve_9x9(mut a: [[f64; 9]; 9], mut b: [f64; 9]) -> Option<[f64; 9]> {
    for k in 0..9 {
        let mut piv = k;
        let mut piv_abs = a[k][k].abs();
        for r in (k + 1)..9 {
            let v = a[r][k].abs();
            if v > piv_abs {
                piv_abs = v;
                piv = r;
            }
        }
        if piv_abs < 1e-12 {
            return None;
        }
        if piv != k {
            a.swap(k, piv);
            b.swap(k, piv);
        }
        let inv = 1.0 / a[k][k];
        for r in (k + 1)..9 {
            let f = a[r][k] * inv;
            if f == 0.0 {
                continue;
            }
            for c in k..9 {
                a[r][c] -= f * a[k][c];
            }
            b[r] -= f * b[k];
        }
    }
    let mut x = [0.0f64; 9];
    for k in (0..9).rev() {
        let mut s = b[k];
        for c in (k + 1)..9 {
            s -= a[k][c] * x[c];
        }
        x[k] = s / a[k][k];
    }
    Some(x)
}

/// Given a real z-root of the characteristic polynomial of `M_z`, recover
/// `(x, y)` from the corresponding right eigenvector of `M_z`.
///
/// The quotient-algebra eigenvector is `v = (1, x, y, z, x², xy, xz, y², yz, z²)`
/// (up to scale). We fix `v[0] = 1` (the basis element "1") and solve rows
/// 1..10 of `(M_z − z·I) v = 0` for the remaining 9 components. Only the
/// first two — `v[1] = x` and `v[2] = y` — are needed to assemble `E`.
///
/// Returns `None` on the rare case where the 9×9 subsystem is singular at
/// the given z (typically a multiplicity-2 root not in the real slice).
fn back_substitute(mz: &[[f64; 10]; 10], z: f64) -> Option<(f64, f64)> {
    let mut a = [[0.0f64; 9]; 9];
    let mut b = [0.0f64; 9];
    for r in 1..10 {
        for c in 1..10 {
            let v = if r == c { mz[r][c] - z } else { mz[r][c] };
            a[r - 1][c - 1] = v;
        }
        b[r - 1] = -mz[r][0];
    }
    let v = solve_9x9(a, b)?;
    Some((v[0], v[1]))
}

/// Nistér 5-point essential-matrix solver.
///
/// Takes 5 pre-normalized correspondences `x̂_i = K⁻¹ x_i` (inhomogeneous
/// `(u, v)`, not homogeneous `(u, v, 1)`) in both views. Returns up to 10
/// candidate essential matrices; downstream code picks among them by
/// cheirality or per-sample inlier scoring.
///
/// Returns an empty `Vec` when the 5-point sample is degenerate (coplanar
/// back-projected rays, duplicated points, or a null-space rank > 4) or
/// when the polynomial system has no real roots (typically only at the
/// boundary of the essential manifold; well-conditioned samples give 2–10
/// real solutions).
///
/// # Algorithm
///
/// 1. SVD of the 5×9 linear constraint → 4D null basis `{X, Y, Z, W}`.
/// 2. Expand `det(E) = 0` and `2 E Eᵀ E − tr(E Eᵀ) E = 0` to a 10×20
///    cubic-monomial matrix.
/// 3. Gauss-Jordan on the 10 degree-3 columns → reduced form expressing
///    every degree-3 monomial in the quotient basis `{1, x, y, z, x², xy,
///    xz, y², yz, z²}`.
/// 4. Build the 10×10 multiplication-by-z action matrix `M_z` in that basis.
/// 5. Real eigenvalues of `M_z` via Hessenberg reduction + Wilkinson-shifted
///    QR iteration on the matrix directly (no characteristic polynomial —
///    see [`eigenvalues_10`] for why).
/// 6. For each real z, recover `(x, y)` from the right null-vector of
///    `M_z − z·I`, then assemble `E = x·X + y·Y + z·Z + W`.
pub fn essential_5pt(x1_norm: &[Vec2F64; 5], x2_norm: &[Vec2F64; 5]) -> Vec<Mat3F64> {
    let basis = match null_space_5x9(x1_norm, x2_norm) {
        Some(b) => b,
        None => return Vec::new(),
    };

    let mut cm = build_constraint_matrix(&basis);
    if !gauss_jordan_eliminate_deg3(&mut cm) {
        return Vec::new();
    }

    let mz = build_action_matrix(&cm);
    let real_zs = real_eigenvalues_10(&mz);

    let mut out = Vec::with_capacity(real_zs.len());
    for z in real_zs {
        if let Some((x, y)) = back_substitute(&mz, z) {
            out.push(assemble_e(&basis, x, y, z));
        }
    }
    out
}

// Re-export helpers for tests without leaking them into the public API.
#[cfg(test)]
pub(crate) fn __test_null_space_5x9(x1: &[Vec2F64; 5], x2: &[Vec2F64; 5]) -> Option<[[f64; 9]; 4]> {
    null_space_5x9(x1, x2)
}

#[cfg(test)]
pub(crate) fn __test_build_constraint_matrix(null_basis: &[[f64; 9]; 4]) -> [[f64; 20]; 10] {
    build_constraint_matrix(null_basis)
}

#[cfg(test)]
pub(crate) fn __test_gauss_jordan_eliminate_deg3(m: &mut [[f64; 20]; 10]) -> bool {
    gauss_jordan_eliminate_deg3(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skew(t: Vec3F64) -> Mat3F64 {
        Mat3F64::from_cols(
            Vec3F64::new(0.0, t.z, -t.y),
            Vec3F64::new(-t.z, 0.0, t.x),
            Vec3F64::new(t.y, -t.x, 0.0),
        )
    }

    fn project_normalized(p: Vec3F64) -> Vec2F64 {
        Vec2F64::new(p.x / p.z, p.y / p.z)
    }

    /// Generate 5 synthetic normalized correspondences from a known (R, t)
    /// with all points in front of both cameras. Enough signal that the
    /// null-space should recover E_true in the span of the 4 basis vectors.
    fn synthetic_sample(r: Mat3F64, t: Vec3F64) -> ([Vec2F64; 5], [Vec2F64; 5]) {
        let pts = [
            Vec3F64::new(-0.5, -0.3, 4.0),
            Vec3F64::new(0.4, -0.2, 3.5),
            Vec3F64::new(-0.3, 0.5, 5.0),
            Vec3F64::new(0.6, 0.4, 4.5),
            Vec3F64::new(-0.1, -0.6, 3.0),
        ];
        let mut x1 = [Vec2F64::ZERO; 5];
        let mut x2 = [Vec2F64::ZERO; 5];
        for (i, p) in pts.iter().enumerate() {
            x1[i] = project_normalized(*p);
            x2[i] = project_normalized(r * *p + t);
        }
        (x1, x2)
    }

    /// Null-space columns must all satisfy the epipolar constraint exactly
    /// — each basis vector, viewed as a 3×3 matrix, is a valid bilinear
    /// constraint on the 5 input correspondences.
    #[test]
    fn test_null_space_5x9_kills_input_rows() {
        let angle = 0.1_f64;
        let r = Mat3F64::from_cols(
            Vec3F64::new(angle.cos(), 0.0, -angle.sin()),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(angle.sin(), 0.0, angle.cos()),
        );
        let t = Vec3F64::new(1.0, 0.0, 0.2);
        let (x1, x2) = synthetic_sample(r, t);
        let basis = __test_null_space_5x9(&x1, &x2).expect("null-space must exist");

        // For each basis vector, build E and verify x2ᵀ E x1 = 0 for all 5
        // correspondences.
        for (k, vec9) in basis.iter().enumerate() {
            let e = vec9_to_mat3(vec9);
            for i in 0..5 {
                let x1h = Vec3F64::new(x1[i].x, x1[i].y, 1.0);
                let x2h = Vec3F64::new(x2[i].x, x2[i].y, 1.0);
                let r = x2h.dot(e * x1h);
                assert!(
                    r.abs() < 1e-8,
                    "basis vec {k} fails epipolar on pt {i}: |x2ᵀ E x1| = {r:.3e}"
                );
            }
        }
    }

    /// The true E (from (R, t)) must lie in the span of the 4 basis vectors
    /// — so fitting a 4-coefficient linear combination to `E_true` should
    /// produce a near-zero residual.
    #[test]
    fn test_null_space_5x9_contains_ground_truth_e() {
        let angle = 0.15_f64;
        let r = Mat3F64::from_cols(
            Vec3F64::new(angle.cos(), 0.0, -angle.sin()),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(angle.sin(), 0.0, angle.cos()),
        );
        let t = Vec3F64::new(0.8, 0.1, 0.3).normalize();
        let (x1, x2) = synthetic_sample(r, t);
        let basis = __test_null_space_5x9(&x1, &x2).expect("null-space must exist");

        let e_true = skew(t) * r;
        // Flatten E_true to row-major 9-vec.
        let cols = e_true.to_cols_array();
        let e_vec = [
            cols[0], cols[3], cols[6], cols[1], cols[4], cols[7], cols[2], cols[5], cols[8],
        ];

        // Project E_true onto the 4D basis (orthonormal from SVD).
        let mut coeffs = [0.0; 4];
        for k in 0..4 {
            for row in 0..9 {
                coeffs[k] += basis[k][row] * e_vec[row];
            }
        }
        // Reconstruct and measure residual.
        let mut recon = [0.0; 9];
        for k in 0..4 {
            for row in 0..9 {
                recon[row] += coeffs[k] * basis[k][row];
            }
        }
        let mut residual = 0.0;
        for row in 0..9 {
            residual += (recon[row] - e_vec[row]).powi(2);
        }
        residual = residual.sqrt();
        let e_norm: f64 = e_vec.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            residual < 1e-8 * e_norm.max(1e-12),
            "E_true should lie in null-space span; residual = {residual:.3e}, ‖E‖ = {e_norm:.3e}"
        );
    }

    /// The constraint polynomials must evaluate to (near) zero at every
    /// `(x, y, z, w=1)` that reconstructs a valid essential. Testing on
    /// `E_true`'s coefficients in the basis verifies the expansion is
    /// algebraically correct.
    #[test]
    fn test_constraint_matrix_vanishes_on_true_e() {
        let angle = 0.1_f64;
        let r = Mat3F64::from_cols(
            Vec3F64::new(angle.cos(), 0.0, -angle.sin()),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(angle.sin(), 0.0, angle.cos()),
        );
        let t = Vec3F64::new(0.9, 0.05, 0.4).normalize();
        let (x1, x2) = synthetic_sample(r, t);
        let basis = __test_null_space_5x9(&x1, &x2).expect("null-space must exist");

        // Project E_true onto the basis. If the W-coefficient is non-zero
        // we can scale so w=1. If it's zero the test is inconclusive — pick
        // a different (R, t).
        let e_true = skew(t) * r;
        let cols = e_true.to_cols_array();
        let e_vec = [
            cols[0], cols[3], cols[6], cols[1], cols[4], cols[7], cols[2], cols[5], cols[8],
        ];

        let mut coeffs = [0.0; 4];
        for k in 0..4 {
            for row in 0..9 {
                coeffs[k] += basis[k][row] * e_vec[row];
            }
        }
        assert!(
            coeffs[3].abs() > 1e-6,
            "test setup: w coefficient is zero, pick a different E"
        );
        let inv_w = 1.0 / coeffs[3];
        let x = coeffs[0] * inv_w;
        let y = coeffs[1] * inv_w;
        let z = coeffs[2] * inv_w;

        // Evaluate the constraint matrix at (x, y, z).
        let cm = __test_build_constraint_matrix(&basis);
        let mut monomials = [0.0f64; 20];
        monomials[0] = 1.0;
        monomials[1] = x;
        monomials[2] = y;
        monomials[3] = z;
        monomials[4] = x * x;
        monomials[5] = x * y;
        monomials[6] = x * z;
        monomials[7] = y * y;
        monomials[8] = y * z;
        monomials[9] = z * z;
        monomials[10] = x * x * x;
        monomials[11] = x * x * y;
        monomials[12] = x * x * z;
        monomials[13] = x * y * y;
        monomials[14] = x * y * z;
        monomials[15] = x * z * z;
        monomials[16] = y * y * y;
        monomials[17] = y * y * z;
        monomials[18] = y * z * z;
        monomials[19] = z * z * z;

        for row_idx in 0..10 {
            let mut s = 0.0;
            for k in 0..20 {
                s += cm[row_idx][k] * monomials[k];
            }
            let scale = coeffs[3].abs().max(1.0).powi(3);
            assert!(
                s.abs() < 1e-6 * scale,
                "constraint row {row_idx} doesn't vanish at E_true: residual = {s:.3e}"
            );
        }
    }

    /// GJ must produce a right 10×10 identity block (columns 10..20 = I)
    /// for non-degenerate inputs, so each row expresses one degree-3
    /// monomial in the quotient basis.
    #[test]
    fn test_gauss_jordan_produces_identity_on_right_block() {
        let angle = 0.1_f64;
        let r = Mat3F64::from_cols(
            Vec3F64::new(angle.cos(), 0.0, -angle.sin()),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(angle.sin(), 0.0, angle.cos()),
        );
        let t = Vec3F64::new(0.9, 0.05, 0.4).normalize();
        let (x1, x2) = synthetic_sample(r, t);
        let basis = __test_null_space_5x9(&x1, &x2).expect("null-space must exist");

        let mut cm = __test_build_constraint_matrix(&basis);
        let ok = __test_gauss_jordan_eliminate_deg3(&mut cm);
        assert!(
            ok,
            "Gauss-Jordan should not fail on a non-degenerate sample"
        );

        for i in 0..10 {
            for j in 0..10 {
                let expected = if i == j { 1.0 } else { 0.0 };
                let val = cm[i][10 + j];
                assert!(
                    (val - expected).abs() < 1e-9,
                    "GJ right block [{i}][{j}] = {val} (expected {expected})"
                );
            }
        }
    }

    /// Full pipeline: synthetic (R, t) → 5 correspondences → essential_5pt
    /// must return at least one candidate matching E_true up to sign/scale.
    #[test]
    fn test_essential_5pt_recovers_true_e() {
        let angle = 0.15_f64;
        let r = Mat3F64::from_cols(
            Vec3F64::new(angle.cos(), 0.0, -angle.sin()),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(angle.sin(), 0.0, angle.cos()),
        );
        let t = Vec3F64::new(0.8, 0.1, 0.3).normalize();
        let (x1, x2) = synthetic_sample(r, t);

        let candidates = essential_5pt(&x1, &x2);
        assert!(
            !candidates.is_empty(),
            "solver must return ≥1 candidate on a non-degenerate sample"
        );

        let e_true = skew(t) * r;
        let flat_true = e_true.to_cols_array();
        let norm_true: f64 = flat_true.iter().map(|v| v * v).sum::<f64>().sqrt();
        let e_true_unit: [f64; 9] = core::array::from_fn(|k| flat_true[k] / norm_true);

        let mut best_err = f64::INFINITY;
        for cand in &candidates {
            let flat = cand.to_cols_array();
            let norm: f64 = flat.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm < 1e-12 {
                continue;
            }
            let mut err_pos = 0.0f64;
            let mut err_neg = 0.0f64;
            for k in 0..9 {
                let u = flat[k] / norm;
                err_pos += (u - e_true_unit[k]).powi(2);
                err_neg += (u + e_true_unit[k]).powi(2);
            }
            let err = err_pos.min(err_neg).sqrt();
            if err < best_err {
                best_err = err;
            }
        }
        assert!(
            best_err < 1e-4,
            "no candidate close to E_true: best Frobenius dist (unit-norm, ±sign) = {best_err:.3e}"
        );
    }

    /// All returned candidates should satisfy the epipolar constraint on
    /// every input correspondence — a candidate that doesn't kill x2ᵀ E x1
    /// means the solver emitted a spurious root that back-substitution
    /// didn't catch.
    #[test]
    fn test_essential_5pt_candidates_satisfy_epipolar() {
        let angle = 0.12_f64;
        let r = Mat3F64::from_cols(
            Vec3F64::new(angle.cos(), 0.0, -angle.sin()),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(angle.sin(), 0.0, angle.cos()),
        );
        let t = Vec3F64::new(0.7, -0.2, 0.4).normalize();
        let (x1, x2) = synthetic_sample(r, t);

        let candidates = essential_5pt(&x1, &x2);
        assert!(!candidates.is_empty());
        for (idx, e) in candidates.iter().enumerate() {
            for i in 0..5 {
                let x1h = Vec3F64::new(x1[i].x, x1[i].y, 1.0);
                let x2h = Vec3F64::new(x2[i].x, x2[i].y, 1.0);
                let r_val = x2h.dot(*e * x1h);
                assert!(
                    r_val.abs() < 1e-6,
                    "candidate {idx} fails epipolar on pt {i}: |x2ᵀ E x1| = {r_val:.3e}"
                );
            }
        }
    }
}
