//! # Fundamental Matrix
//!
//! The fundamental matrix F encodes epipolar geometry in **pixel space**: for corresponding
//! points x1, x2 across two views, `x2ᵀ F x1 = 0`.
//!
//! - 7 DOF: 3×3 matrix (9) minus scale (1) minus rank-2 constraint (1).
//! - Singular values: (σ₁, σ₂, 0). The zero enforces rank 2.
//! - Related to the essential matrix by camera intrinsics: `E = K2ᵀ F K1`,
//!   equivalently `F = K2⁻ᵀ E K1⁻¹`.
//!
//! ## 8-point algorithm
//!
//! Build a design matrix A from ≥8 correspondences, take the null vector of A (last
//! column of V from SVD), reshape into 3×3, enforce rank 2 by zeroing the smallest
//! singular value. Hartley normalization (translate centroid to origin, scale mean
//! distance to √2) is critical for numerical stability.
//!
//! ## Relationship to the essential manifold
//!
//! F and E encode the same geometry at different levels. F works in pixel space (depends
//! on camera intrinsics K), E works in metric space (intrinsics removed). The conversion
//! `E = K2ᵀ F K1` is invertible when K is known.
//!
//! F has 7 DOF; E has only 5 DOF because the essential manifold is a constrained subset
//! (the (σ,σ,0) singular value structure) — those 2 extra DOF in F encode the intrinsics.
//! The essential manifold is itself a quotient of SE(3), losing translation scale and sign
//! (see the [`essential`](super::essential) module).
//!
//! ## Pitfalls
//!
//! - **Transpose bugs are silent**: F and Fᵀ both satisfy rank-2, both have the same
//!   singular values. Only asymmetric tests (epipolar constraint direction, Sampson
//!   distance) catch them.
//! - **Column-major storage**: the SVD null vector is in row-major order. When using
//!   `from_cols`, consecutive 3 elements give columns, not rows — grouping wrong gives Fᵀ.

#![allow(clippy::needless_range_loop)]

use kornia_algebra::{linalg::svd::svd3_f64, Mat3F64, Vec2F64, Vec3F64};

/// Error type for fundamental matrix estimation.
#[derive(thiserror::Error, Debug)]
pub enum FundamentalError {
    /// Input correspondences are invalid or insufficient.
    #[error("Need at least 8 correspondences and equal lengths")]
    InvalidInput,
    /// SVD failed or produced an invalid result.
    #[error("SVD failed to produce a valid fundamental matrix")]
    SvdFailure,
}

/// Estimate the fundamental matrix using the normalized 8-point algorithm.
///
/// - `x1`: points in image 1 as `&[Vec2F64]` (length >= 8)
/// - `x2`: corresponding points in image 2 as `&[Vec2F64]` (same length)
pub fn fundamental_8point(x1: &[Vec2F64], x2: &[Vec2F64]) -> Result<Mat3F64, FundamentalError> {
    if x1.len() != x2.len() || x1.len() < 8 {
        return Err(FundamentalError::InvalidInput);
    }

    let (x1n, t1) = normalize_points_2d(x1);
    let (x2n, t2) = normalize_points_2d(x2);

    // Null-vector of A (the n×9 design matrix). Householder on Aᵀ (9×n) wins
    // for small n because it skips the fixed 9×9 eigen cost; M = AᵀA + LDLᵀ
    // amortizes better for large n. Crossover at n ≈ 64.
    let fvec = if x1n.len() <= 64 {
        null_vector_householder(&x1n, &x2n)
    } else {
        null_vector_mtm(&x1n, &x2n)
    };

    let f = Mat3F64::from_cols(
        Vec3F64::new(fvec[0], fvec[3], fvec[6]),
        Vec3F64::new(fvec[1], fvec[4], fvec[7]),
        Vec3F64::new(fvec[2], fvec[5], fvec[8]),
    );

    let f_rank2 = enforce_rank2(&f)?;

    // Denormalize: F = T2^T * F * T1
    let f_denorm = t2.transpose() * f_rank2 * t1;
    Ok(f_denorm)
}

/// Return the 9-vector spanning the null space of the 8-point design matrix A
/// built from the normalized correspondences `x1n`, `x2n` (length n ≥ 8).
///
/// Builds Aᵀ in column-major layout (9 rows × n cols, each column 9 contiguous
/// f64), runs 8 Householder reflections, and extracts the null vector as
/// `Q·e₈` via reverse-order application of the reflectors.
#[inline]
fn null_vector_householder(x1n: &[Vec2F64], x2n: &[Vec2F64]) -> [f64; 9] {
    let n = x1n.len();

    // Stack-only up to STACK_N correspondences; larger n spills to heap.
    // STACK_N = 32 covers the RANSAC minimal set (8) plus typical inlier
    // refits. 32 × 9 × 8B = 2304 B — comfortably within A78AE stack budgets.
    const STACK_N: usize = 32;
    let mut stack_buf = [0.0f64; 9 * STACK_N];
    let mut heap_buf;
    let at: &mut [f64] = if n <= STACK_N {
        &mut stack_buf[..9 * n]
    } else {
        heap_buf = vec![0.0f64; 9 * n];
        heap_buf.as_mut_slice()
    };

    for i in 0..n {
        let (x, y) = (x1n[i].x, x1n[i].y);
        let (xp, yp) = (x2n[i].x, x2n[i].y);
        let base = i * 9;
        at[base] = xp * x;
        at[base + 1] = xp * y;
        at[base + 2] = xp;
        at[base + 3] = yp * x;
        at[base + 4] = yp * y;
        at[base + 5] = yp;
        at[base + 6] = x;
        at[base + 7] = y;
        at[base + 8] = 1.0;
    }

    let mut u = [[0.0f64; 9]; 8];

    for k in 0..8 {
        let m = 9 - k;
        let col_k = &at[k * 9..k * 9 + 9];
        // Build u_full[k..k+m] in place: raw column → sign-flipped Householder
        // vector → normalize, all in one buffer.
        let mut u_full = [0.0f64; 9];
        let mut norm_sq = 0.0;
        for i in 0..m {
            let v = col_k[k + i];
            u_full[k + i] = v;
            norm_sq += v * v;
        }
        let norm = norm_sq.sqrt();
        if norm < 1e-14 {
            continue;
        }
        let x0 = u_full[k];
        let alpha = if x0 >= 0.0 { -norm } else { norm };
        u_full[k] = x0 - alpha;
        // ||u||² = 2·norm·(norm ± x0) analytically — avoids a second pass.
        let unorm_sq = 2.0 * norm * (norm - alpha.signum() * x0);
        if unorm_sq < 1e-28 {
            continue;
        }
        let inv_unorm = 1.0 / unorm_sq.sqrt();
        for i in 0..m {
            u_full[k + i] *= inv_unorm;
        }
        u[k] = u_full;

        for j in k..n {
            let col = &mut at[j * 9..j * 9 + 9];
            apply_reflector_col(col, &u_full, k);
        }
    }

    // Null vector = Q·e₈ = H₀·H₁·…·H₇·e₈. Apply right-to-left.
    let mut v = [0.0f64; 9];
    v[8] = 1.0;
    for k in (0..8).rev() {
        apply_reflector_col(&mut v, &u[k], k);
    }
    v
}

/// Apply H = I - 2·u·uᵀ to a 9-element slice `col` in place; `u` is non-zero
/// only in indices `k..9`, so we skip `col[0..k]`.
///
/// aarch64 path: 2-lane f64 `vfmaq_f64` for both the dot and AXPY passes, one
/// scalar tail at the end. 9-k ∈ {1..9} gives up to 4 vector iters + 1 scalar.
#[inline(always)]
fn apply_reflector_col(col: &mut [f64], u: &[f64; 9], k: usize) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::{
            vaddvq_f64, vdupq_n_f64, vfmaq_f64, vfmsq_f64, vld1q_f64, vst1q_f64,
        };
        let u_ptr = u.as_ptr();
        let c_ptr = col.as_mut_ptr();
        let mut i = k;
        let mut acc = vdupq_n_f64(0.0);
        while i + 2 <= 9 {
            let uv = vld1q_f64(u_ptr.add(i));
            let cv = vld1q_f64(c_ptr.add(i));
            acc = vfmaq_f64(acc, uv, cv);
            i += 2;
        }
        let mut dot = vaddvq_f64(acc);
        if i < 9 {
            dot += u[i] * col[i];
        }
        let two_dot = 2.0 * dot;
        let tdv = vdupq_n_f64(two_dot);
        let mut i = k;
        while i + 2 <= 9 {
            let uv = vld1q_f64(u_ptr.add(i));
            let cv = vld1q_f64(c_ptr.add(i));
            let new = vfmsq_f64(cv, tdv, uv);
            vst1q_f64(c_ptr.add(i), new);
            i += 2;
        }
        if i < 9 {
            col[i] -= two_dot * u[i];
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if kornia_imgproc::simd::cpu_features().has_avx2 {
            unsafe { apply_reflector_col_avx2(col, u, k) };
            return;
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut dot = 0.0;
        for i in k..9 {
            dot += u[i] * col[i];
        }
        let two_dot = 2.0 * dot;
        for i in k..9 {
            col[i] -= two_dot * u[i];
        }
    }
}

/// AVX2 mirror of the `apply_reflector_col` NEON path. 4-lane f64 (`__m256d`)
/// FMA covers the dot accumulator and the AXPY pass; 9-element vector means
/// at most 2 vector iters + a 1-3 element scalar tail. Horizontal reduce
/// uses the standard `extractf128 + addhi` chain (AVX2 has no
/// `vaddvq_f64`-equivalent reduction primitive).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn apply_reflector_col_avx2(col: &mut [f64], u: &[f64; 9], k: usize) {
    use std::arch::x86_64::*;
    let u_ptr = u.as_ptr();
    let c_ptr = col.as_mut_ptr();
    let mut i = k;
    let mut acc = _mm256_setzero_pd();
    while i + 4 <= 9 {
        let uv = _mm256_loadu_pd(u_ptr.add(i));
        let cv = _mm256_loadu_pd(c_ptr.add(i));
        acc = _mm256_fmadd_pd(uv, cv, acc);
        i += 4;
    }
    let lo128 = _mm256_castpd256_pd128(acc);
    let hi128 = _mm256_extractf128_pd::<1>(acc);
    let sum2 = _mm_add_pd(lo128, hi128);
    let sum2_hi = _mm_unpackhi_pd(sum2, sum2);
    let mut dot = _mm_cvtsd_f64(_mm_add_sd(sum2, sum2_hi));
    while i < 9 {
        dot += u[i] * col[i];
        i += 1;
    }
    let two_dot = 2.0 * dot;
    let tdv = _mm256_set1_pd(two_dot);
    let mut i = k;
    while i + 4 <= 9 {
        let uv = _mm256_loadu_pd(u_ptr.add(i));
        let cv = _mm256_loadu_pd(c_ptr.add(i));
        let new = _mm256_fnmadd_pd(tdv, uv, cv);
        _mm256_storeu_pd(c_ptr.add(i), new);
        i += 4;
    }
    while i < 9 {
        col[i] -= two_dot * u[i];
        i += 1;
    }
}

/// Fallback null-vector path for large n: accumulate M = AᵀA (9×9 symmetric)
/// via rank-1 outer products, then take the eigenvector of the smallest
/// eigenvalue. The fixed O(9³) eigen cost amortizes better than Householder's
/// O(8n) column updates once n exceeds ~70.
#[inline]
fn null_vector_mtm(x1n: &[Vec2F64], x2n: &[Vec2F64]) -> [f64; 9] {
    let mut m = [[0.0f64; 9]; 9];
    for i in 0..x1n.len() {
        let (x, y) = (x1n[i].x, x1n[i].y);
        let (xp, yp) = (x2n[i].x, x2n[i].y);
        let row = [xp * x, xp * y, xp, yp * x, yp * y, yp, x, y, 1.0];
        for a in 0..9 {
            let ra = row[a];
            for b in a..9 {
                m[a][b] += ra * row[b];
            }
        }
    }
    for a in 0..9 {
        for b in 0..a {
            m[a][b] = m[b][a];
        }
    }

    smallest_eigenvector_9x9_sym(&m)
}

/// Smallest-eigenvector of a 9×9 symmetric PSD matrix via inverse iteration on
/// an LDLᵀ factorization of `M + εI`.
///
/// For the 8-point problem M is rank-deficient by construction (λ_min ≈ 0), so
/// the λ_min/λ_2nd convergence ratio is ~0 — 3 iterations converge to double
/// precision. Total ≈ 420 f64 ops vs ~5000 for a generic eigendecomposition.
/// The ε-shift keeps LDLᵀ numerically PD without biasing the recovered vector.
#[inline]
fn smallest_eigenvector_9x9_sym(m: &[[f64; 9]; 9]) -> [f64; 9] {
    let mut trace = 0.0;
    for i in 0..9 {
        trace += m[i][i];
    }
    let eps = 1e-14 * trace.max(1e-300);

    let mut l = *m;
    for i in 0..9 {
        l[i][i] += eps;
    }
    for j in 0..9 {
        let mut djj = l[j][j];
        for k in 0..j {
            djj -= l[j][k] * l[j][k] * l[k][k];
        }
        l[j][j] = djj;
        // Safety guard — never triggers on well-conditioned 8-point data, but
        // keeps division safe if ε underflows against a degenerate input.
        let inv_djj = if djj.abs() > 1e-300 { 1.0 / djj } else { 0.0 };
        for i in (j + 1)..9 {
            let mut sum = l[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k] * l[k][k];
            }
            l[i][j] = sum * inv_djj;
        }
    }

    // Constant seed (1/3, …, 1/3) is safe for the 8-point problem: the null
    // basis has no systematic sign pattern that would orthogonalize it.
    let mut v = [1.0f64 / 3.0; 9];
    for _ in 0..3 {
        let mut y = v;
        for i in 1..9 {
            let mut s = y[i];
            for k in 0..i {
                s -= l[i][k] * y[k];
            }
            y[i] = s;
        }
        for i in 0..9 {
            let dii = l[i][i];
            y[i] = if dii.abs() > 1e-300 { y[i] / dii } else { 0.0 };
        }
        for i in (0..9).rev() {
            let mut s = y[i];
            for k in (i + 1)..9 {
                s -= l[k][i] * y[k];
            }
            y[i] = s;
        }
        let mut norm_sq = 0.0;
        for i in 0..9 {
            norm_sq += y[i] * y[i];
        }
        let inv_norm = if norm_sq > 0.0 {
            1.0 / norm_sq.sqrt()
        } else {
            1.0
        };
        for i in 0..9 {
            v[i] = y[i] * inv_norm;
        }
    }
    v
}

/// Compute the Sampson distance for a correspondence.
pub fn sampson_distance(f: &Mat3F64, x1: &Vec2F64, x2: &Vec2F64) -> f64 {
    let x1h = Vec3F64::new(x1.x, x1.y, 1.0);
    let x2h = Vec3F64::new(x2.x, x2.y, 1.0);

    let fx1 = *f * x1h;
    let ftx2 = f.transpose() * x2h;

    let err = x2h.dot(fx1);
    let denom = fx1.x * fx1.x + fx1.y * fx1.y + ftx2.x * ftx2.x + ftx2.y * ftx2.y;
    if denom <= 1e-12 {
        return err * err;
    }
    (err * err) / denom
}

fn normalize_points_2d(x: &[Vec2F64]) -> (Vec<Vec2F64>, Mat3F64) {
    let n = x.len() as f64;
    let mut mx = 0.0;
    let mut my = 0.0;
    for p in x {
        mx += p.x;
        my += p.y;
    }
    mx /= n;
    my /= n;

    let mut mean_dist = 0.0;
    for p in x {
        let dx = p.x - mx;
        let dy = p.y - my;
        mean_dist += (dx * dx + dy * dy).sqrt();
    }
    mean_dist /= n;

    let scale = if mean_dist > 0.0 {
        std::f64::consts::SQRT_2 / mean_dist
    } else {
        1.0
    };

    let mut xn = Vec::with_capacity(x.len());
    for p in x {
        xn.push(Vec2F64::new((p.x - mx) * scale, (p.y - my) * scale));
    }

    let t = Mat3F64::from_cols(
        Vec3F64::new(scale, 0.0, 0.0),
        Vec3F64::new(0.0, scale, 0.0),
        Vec3F64::new(-scale * mx, -scale * my, 1.0),
    );

    (xn, t)
}

fn enforce_rank2(f: &Mat3F64) -> Result<Mat3F64, FundamentalError> {
    let svd = svd3_f64(f);
    let mut s = *svd.s();
    s.z_axis.z = 0.0;
    let f_rank2 = *svd.u() * s * svd.v().transpose();
    Ok(f_rank2)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Shared test setup: well-conditioned correspondences from a known two-camera geometry.
    /// Returns (x1, x2, F_true) where x2^T * F_true * x1 = 0 for all pairs.
    fn make_test_correspondences() -> (Vec<Vec2F64>, Vec<Vec2F64>, Mat3F64) {
        let k = Mat3F64::from_cols(
            Vec3F64::new(500.0, 0.0, 0.0),
            Vec3F64::new(0.0, 500.0, 0.0),
            Vec3F64::new(320.0, 240.0, 1.0),
        );
        let k_inv = k.inverse();

        // Rotation ~5.7 deg around Y axis
        let angle = 0.1_f64;
        let r = Mat3F64::from_cols(
            Vec3F64::new(angle.cos(), 0.0, -angle.sin()),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(angle.sin(), 0.0, angle.cos()),
        );
        let t = Vec3F64::new(1.0, 0.0, 0.2);
        let t_len = t.length();
        let t_unit = Vec3F64::new(t.x / t_len, t.y / t_len, t.z / t_len);

        // E = [t]_x * R, F = K^{-T} E K^{-1}
        let tx = Mat3F64::from_cols(
            Vec3F64::new(0.0, t_unit.z, -t_unit.y),
            Vec3F64::new(-t_unit.z, 0.0, t_unit.x),
            Vec3F64::new(t_unit.y, -t_unit.x, 0.0),
        );
        let e = tx * r;
        let f_true = k_inv.transpose() * e * k_inv;

        // 3D points well-spread in front of both cameras
        let pts = [
            Vec3F64::new(-0.5, -0.3, 4.0),
            Vec3F64::new(0.4, -0.2, 3.5),
            Vec3F64::new(-0.3, 0.5, 5.0),
            Vec3F64::new(0.6, 0.4, 4.5),
            Vec3F64::new(-0.1, -0.6, 3.0),
            Vec3F64::new(0.2, 0.3, 6.0),
            Vec3F64::new(-0.4, 0.1, 3.8),
            Vec3F64::new(0.5, -0.5, 4.2),
            Vec3F64::new(0.0, 0.0, 5.5),
            Vec3F64::new(-0.2, 0.4, 4.8),
            Vec3F64::new(0.3, -0.1, 3.2),
            Vec3F64::new(-0.6, 0.2, 5.8),
        ];

        let mut x1 = Vec::new();
        let mut x2 = Vec::new();
        for p in &pts {
            let p1 = k * *p;
            x1.push(Vec2F64::new(p1.x / p1.z, p1.y / p1.z));
            let p2_cam = r * *p + t;
            let p2 = k * p2_cam;
            x2.push(Vec2F64::new(p2.x / p2.z, p2.y / p2.z));
        }

        (x1, x2, f_true)
    }

    #[test]
    fn test_fundamental_8point_epipolar_constraint() {
        let (x1, x2, _) = make_test_correspondences();

        let f_est = fundamental_8point(&x1, &x2).unwrap();

        // Check rank-2 property
        let svd = svd3_f64(&f_est);
        let s = svd.s();
        assert!(
            s.z_axis.z.abs() < 1e-6,
            "smallest singular value should be near zero (rank-2 constraint), got: {}",
            s.z_axis.z
        );

        // Check epipolar constraint: x2^T * F_est * x1 ≈ 0 for all correspondences.
        // This would FAIL with large errors if F were transposed.
        for i in 0..x1.len() {
            let x1h = Vec3F64::new(x1[i].x, x1[i].y, 1.0);
            let x2h = Vec3F64::new(x2[i].x, x2[i].y, 1.0);
            let err = x2h.dot(f_est * x1h);
            assert!(
                err.abs() < 1e-8,
                "epipolar constraint violated for point {i}: err = {err}"
            );
        }
    }

    #[test]
    fn test_fundamental_8point_sampson_distance_near_zero() {
        let (x1, x2, _) = make_test_correspondences();

        let f_est = fundamental_8point(&x1, &x2).unwrap();

        // Sampson distance (first-order geometric error) should be near zero
        // for all correspondences when F is correctly oriented.
        for i in 0..x1.len() {
            let d = sampson_distance(&f_est, &x1[i], &x2[i]);
            assert!(
                d < 1e-10,
                "sampson distance too large for point {i}: d = {d}"
            );
        }
    }

    #[test]
    fn test_fundamental_8point_proportional_to_true() {
        let (x1, x2, f_true) = make_test_correspondences();

        let f_est = fundamental_8point(&x1, &x2).unwrap();

        // Normalize both matrices by Frobenius norm, then check element-wise proportionality.
        // This catches a transpose: F^T normalized differs from F normalized.
        let f_true_arr: [f64; 9] = f_true.into();
        let f_est_arr: [f64; 9] = f_est.into();

        let norm_true: f64 = f_true_arr.iter().map(|v| v * v).sum::<f64>().sqrt();
        let norm_est: f64 = f_est_arr.iter().map(|v| v * v).sum::<f64>().sqrt();

        // Find scale factor from a large element
        let mut scale = 0.0;
        for i in 0..9 {
            let a = f_true_arr[i] / norm_true;
            if a.abs() > 0.05 {
                scale = (f_est_arr[i] / norm_est) / a;
                break;
            }
        }
        assert!(scale.abs() > 1e-6, "could not determine scale factor");

        for i in 0..9 {
            let a = f_true_arr[i] / norm_true;
            let b = f_est_arr[i] / norm_est;
            let diff = (b - scale * a).abs();
            assert!(
                diff < 1e-4,
                "F_est not proportional to F_true at element {i}: expected {}, got {}, diff = {diff}",
                scale * a, b
            );
        }
    }

    #[test]
    fn test_sampson_distance_zero_on_epipolar_line() {
        let (_, _, f_true) = make_test_correspondences();

        // Pick an arbitrary point and find its match on the epipolar line
        let x1 = Vec2F64::new(350.0, 200.0);
        let x1h = Vec3F64::new(x1.x, x1.y, 1.0);
        let l = f_true * x1h;
        // Find x2 on line l: l.x * u + l.y * v + l.z = 0
        // Choose u = 300, solve for v
        let u = 300.0;
        let v = -(l.x * u + l.z) / l.y;
        let x2 = Vec2F64::new(u, v);

        let d = sampson_distance(&f_true, &x1, &x2);
        assert!(d < 1e-10, "sampson distance = {d}");
    }

    // ---------------------------------------------------------------------
    // Solver-internal tests: these exercise the null-space kernels directly,
    // so a regression in Householder/LDLᵀ/NEON gets caught without being
    // masked by the rank-2 enforcement and denormalization that follow.
    // ---------------------------------------------------------------------

    /// Frobenius norm of a 9-vector treated as a flattened 3×3.
    fn vec9_norm(v: &[f64; 9]) -> f64 {
        v.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// |A·v|₂ for A implicit in the normalized correspondences.
    fn residual_norm(x1n: &[Vec2F64], x2n: &[Vec2F64], v: &[f64; 9]) -> f64 {
        let mut rr = 0.0;
        for i in 0..x1n.len() {
            let (x, y) = (x1n[i].x, x1n[i].y);
            let (xp, yp) = (x2n[i].x, x2n[i].y);
            let r = xp * x * v[0]
                + xp * y * v[1]
                + xp * v[2]
                + yp * x * v[3]
                + yp * y * v[4]
                + yp * v[5]
                + x * v[6]
                + y * v[7]
                + v[8];
            rr += r * r;
        }
        rr.sqrt()
    }

    #[test]
    fn test_null_vector_householder_kills_design_matrix() {
        // Build normalized correspondences with a known planar rank (rank 8 ⇒
        // exact 1-D null space exists). Use fundamental_8point's Hartley
        // normalizer so the design matrix is well-conditioned.
        let (x1, x2, _) = make_test_correspondences();
        let (x1n, _) = normalize_points_2d(&x1[..8]);
        let (x2n, _) = normalize_points_2d(&x2[..8]);

        let v = null_vector_householder(&x1n, &x2n);

        let norm = vec9_norm(&v);
        assert!((norm - 1.0).abs() < 1e-10, "null vector not unit: {norm}");

        let res = residual_norm(&x1n, &x2n, &v);
        assert!(res < 1e-10, "A·v residual too large: {res}");
    }

    #[test]
    fn test_null_vector_mtm_kills_design_matrix() {
        // Use n>64 to land in the M-build + LDLᵀ + inverse-iteration path.
        // Synthetic planar homography data has rank-8 design matrix → perfect
        // null-space recovery.
        let h_true = Mat3F64::from_cols(
            Vec3F64::new(1.2, 0.0, 0.001),
            Vec3F64::new(0.1, 0.9, 0.002),
            Vec3F64::new(5.0, -3.0, 1.0),
        );
        let mut x1 = Vec::new();
        let mut x2 = Vec::new();
        for i in 0..80 {
            let xi = (i as f64 % 10.0) * 2.0 - 10.0;
            let yi = (i as f64 / 10.0) * 1.5 - 6.0;
            let p = Vec3F64::new(xi, yi, 1.0);
            let hp = h_true * p;
            x1.push(Vec2F64::new(xi, yi));
            x2.push(Vec2F64::new(hp.x / hp.z, hp.y / hp.z));
        }
        let (x1n, _) = normalize_points_2d(&x1);
        let (x2n, _) = normalize_points_2d(&x2);

        let v = null_vector_mtm(&x1n, &x2n);
        let norm = vec9_norm(&v);
        assert!((norm - 1.0).abs() < 1e-8, "null vector not unit: {norm}");

        let res = residual_norm(&x1n, &x2n, &v);
        assert!(res < 1e-6, "A·v residual too large: {res}");
    }

    #[test]
    fn test_null_vector_householder_and_mtm_agree_on_same_input() {
        // Both kernels must recover the same null vector (up to sign) on
        // identical input. Exercises the crossover boundary.
        let (x1, x2, _) = make_test_correspondences();
        let (x1n, _) = normalize_points_2d(&x1);
        let (x2n, _) = normalize_points_2d(&x2);

        let vh = null_vector_householder(&x1n, &x2n);
        let vm = null_vector_mtm(&x1n, &x2n);

        // Align signs by the largest-magnitude entry so comparison is direction-agnostic.
        let mut k = 0;
        for i in 1..9 {
            if vh[i].abs() > vh[k].abs() {
                k = i;
            }
        }
        let sign = if vh[k] * vm[k] < 0.0 { -1.0 } else { 1.0 };
        for i in 0..9 {
            let d = (vh[i] - sign * vm[i]).abs();
            assert!(d < 1e-6, "paths disagree at {i}: h={} m={}", vh[i], vm[i]);
        }
    }

    #[test]
    fn test_smallest_eigenvector_9x9_sym_known_spectrum() {
        // Build M = Σ λ_i · e_i · e_iᵀ with eigenvalues [10, 9, 8, …, 2, 0]
        // (smallest = 0, eigenvector = e₈). Verify recovery.
        let mut m = [[0.0f64; 9]; 9];
        let evals = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 2.0, 0.0];
        for (i, &lam) in evals.iter().enumerate() {
            m[i][i] = lam;
        }
        let v = smallest_eigenvector_9x9_sym(&m);
        // Smallest eigenvalue is at index 8, so v should be ±e₈.
        assert!(v[8].abs() > 0.99, "expected |v[8]|≈1, got {}", v[8]);
        for (i, &x) in v.iter().enumerate().take(8) {
            assert!(x.abs() < 1e-6, "expected v[{i}]≈0, got {x}");
        }
    }

    #[test]
    fn test_smallest_eigenvector_9x9_sym_rotated_spectrum() {
        // Non-diagonal: M = Qᵀ · diag(λ) · Q where Q is a random-ish orthogonal
        // rotation. Verify we recover Qᵀ · e₈ (column 8 of Qᵀ).
        // Use a simple block rotation in (0,1), (2,3), (4,5), (6,7) planes.
        let theta = [0.7_f64, 1.1, -0.4, 2.1];
        // Q = block-diag of 2×2 rotations + 1 on diagonal 8.
        let mut q = [[0.0f64; 9]; 9];
        for b in 0..4 {
            let c = theta[b].cos();
            let s = theta[b].sin();
            q[2 * b][2 * b] = c;
            q[2 * b][2 * b + 1] = -s;
            q[2 * b + 1][2 * b] = s;
            q[2 * b + 1][2 * b + 1] = c;
        }
        q[8][8] = 1.0;
        let diag = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 1e-10];
        // M = Qᵀ · D · Q  (symmetric by construction)
        let mut m = [[0.0f64; 9]; 9];
        for i in 0..9 {
            for j in 0..9 {
                let mut s = 0.0;
                for k in 0..9 {
                    s += q[k][i] * diag[k] * q[k][j];
                }
                m[i][j] = s;
            }
        }
        let v = smallest_eigenvector_9x9_sym(&m);
        let norm = vec9_norm(&v);
        assert!((norm - 1.0).abs() < 1e-10, "not unit: {norm}");
        // Verify M·v ≈ 0 (up to the 1e-10 shift).
        let mut mv = [0.0f64; 9];
        for i in 0..9 {
            for j in 0..9 {
                mv[i] += m[i][j] * v[j];
            }
        }
        let res = vec9_norm(&mv);
        assert!(res < 1e-7, "M·v not near zero: {res}");
    }

    #[test]
    fn test_apply_reflector_col_is_involution() {
        // H = I - 2uuᵀ for unit u is its own inverse — applying twice returns
        // the original vector (to within floating-point roundoff).
        let mut u = [0.0f64; 9];
        u[2] = 0.6;
        u[3] = 0.8;
        // Normalized in indices [2..4].
        let original = [0.1_f64, 0.3, -0.5, 1.2, 2.0, -1.1, 0.7, -0.2, 0.05];
        let mut col = original;
        apply_reflector_col(&mut col, &u, 2);
        apply_reflector_col(&mut col, &u, 2);
        for i in 0..9 {
            assert!(
                (col[i] - original[i]).abs() < 1e-12,
                "H²≠I at {i}: got {} want {}",
                col[i],
                original[i]
            );
        }
    }

    #[test]
    fn test_apply_reflector_col_preserves_norm() {
        // Orthogonal transformation → ‖H·v‖₂ = ‖v‖₂.
        let mut u = [0.0f64; 9];
        // u supported on indices 1..9 (k=1). Normalize.
        let raw = [0.0, 0.3, -0.2, 0.5, 0.1, -0.4, 0.2, 0.6, -0.1];
        let mut nn = 0.0;
        for (i, &r) in raw.iter().enumerate() {
            u[i] = r;
            nn += r * r;
        }
        let inv = 1.0 / nn.sqrt();
        for i in 1..9 {
            u[i] *= inv;
        }
        let original = [1.0_f64, 2.0, 3.0, -1.0, 0.5, -2.0, 0.1, -0.7, 1.4];
        let before: f64 = original.iter().map(|x| x * x).sum::<f64>().sqrt();
        let mut col = original;
        apply_reflector_col(&mut col, &u, 1);
        let after: f64 = col.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            (before - after).abs() < 1e-12,
            "norm not preserved: {before} vs {after}"
        );
    }

    #[test]
    fn test_apply_reflector_col_reflects_u_to_minus_u() {
        // Classic property: H·u = -u when u is the reflector direction.
        let mut u = [0.0f64; 9];
        // Supported in 3..9, unit-norm.
        let raw = [0.0, 0.0, 0.0, 0.5, -0.3, 0.6, 0.2, -0.4, 0.1];
        let mut nn = 0.0;
        for (i, &r) in raw.iter().enumerate() {
            u[i] = r;
            nn += r * r;
        }
        let inv = 1.0 / nn.sqrt();
        for i in 3..9 {
            u[i] *= inv;
        }
        let mut col = [0.0f64; 9];
        col[3..9].copy_from_slice(&u[3..9]);
        apply_reflector_col(&mut col, &u, 3);
        for i in 3..9 {
            assert!(
                (col[i] + u[i]).abs() < 1e-12,
                "H·u ≠ -u at {i}: got {} want {}",
                col[i],
                -u[i]
            );
        }
    }

    #[test]
    fn test_fundamental_8point_large_n_hits_mtm_path() {
        // n=100 ≥ 65 → crossover dispatch sends this through null_vector_mtm.
        // Verify the full pipeline (including LDLᵀ inverse iteration) still
        // produces a valid F.
        let h_true = Mat3F64::from_cols(
            Vec3F64::new(1.2, 0.0, 0.001),
            Vec3F64::new(0.1, 0.9, 0.002),
            Vec3F64::new(5.0, -3.0, 1.0),
        );
        let mut x1 = Vec::new();
        let mut x2 = Vec::new();
        for i in 0..100 {
            let xi = (i as f64 % 10.0) * 2.0 - 10.0;
            let yi = (i as f64 / 10.0) * 1.5 - 6.0;
            let p = Vec3F64::new(xi, yi, 1.0);
            let hp = h_true * p;
            x1.push(Vec2F64::new(xi, yi));
            x2.push(Vec2F64::new(hp.x / hp.z, hp.y / hp.z));
        }
        let f = fundamental_8point(&x1, &x2).expect("should succeed on clean data");
        // Rank-2 constraint enforced by the solver.
        let svd = svd3_f64(&f);
        assert!(
            svd.s().z_axis.z.abs() < 1e-6,
            "rank-2 violated: σ₃={}",
            svd.s().z_axis.z
        );
        // Epipolar constraint residual on every correspondence.
        for i in 0..x1.len() {
            let x1h = Vec3F64::new(x1[i].x, x1[i].y, 1.0);
            let x2h = Vec3F64::new(x2[i].x, x2[i].y, 1.0);
            let err = x2h.dot(f * x1h);
            assert!(err.abs() < 1e-6, "point {i}: x2ᵀFx1 = {err}");
        }
    }

    #[test]
    fn test_fundamental_8point_noisy_data_is_stable() {
        // Small Gaussian-style perturbation (±0.3 px) on projected points
        // should not blow up the solver. A poorly conditioned null-space
        // extraction shows up here as NaN / huge residuals.
        let (mut x1, mut x2, _) = make_test_correspondences();
        // Deterministic pseudo-noise from a linear congruential sequence.
        let mut seed = 12345u64;
        let mut next_noise = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f64 / (1u64 << 31) as f64 - 1.0) * 0.3
        };
        for p in x1.iter_mut().chain(x2.iter_mut()) {
            p.x += next_noise();
            p.y += next_noise();
        }
        let f = fundamental_8point(&x1, &x2).expect("solver should not fail on noisy data");
        // All entries finite.
        let farr: [f64; 9] = f.into();
        for (i, &v) in farr.iter().enumerate() {
            assert!(v.is_finite(), "F[{i}] not finite: {v}");
        }
        // Sampson distance is small but not zero under noise; sanity-check
        // it stays bounded rather than NaN-or-huge.
        let median_d = {
            let mut ds: Vec<f64> = (0..x1.len())
                .map(|i| sampson_distance(&f, &x1[i], &x2[i]))
                .collect();
            ds.sort_by(|a, b| a.partial_cmp(b).unwrap());
            ds[ds.len() / 2]
        };
        assert!(
            median_d < 1.0,
            "median Sampson distance under light noise exploded: {median_d}"
        );
    }
}
