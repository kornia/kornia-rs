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
//! 4. **Root finding**: companion-matrix eigendecomposition yields the 10
//!    roots (complex in general). Each real root recovers `(x, y)` by
//!    back-substituting into the reduced matrix.
//! 5. **Assemble**: `E = x·X + y·Y + z·Z + W` for each real solution.
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

use kornia_algebra::{Mat3F64, Vec2F64, Vec3F64};

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

    fn add_assign(&mut self, other: &Poly3D) {
        for i in 0..20 {
            self.c[i] += other.c[i];
        }
    }

    fn sub_assign(&mut self, other: &Poly3D) {
        for i in 0..20 {
            self.c[i] -= other.c[i];
        }
    }

    fn scale(&self, s: f64) -> Poly3D {
        let mut out = Poly3D::ZERO;
        for i in 0..20 {
            out.c[i] = self.c[i] * s;
        }
        out
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

/// In-place Gauss-Jordan reduction of a 10×20 matrix to reduced row-echelon
/// form on the first 10 columns when possible. Uses partial pivoting for
/// numerical stability; returns `false` when the leading 10×10 sub-block is
/// singular (near-degenerate sample).
fn gauss_jordan_10x20(m: &mut [[f64; 20]; 10]) -> bool {
    for col in 0..10 {
        // Partial pivot: find row in [col..10] with max |m[r][col]|.
        let mut piv = col;
        let mut piv_abs = m[col][col].abs();
        for r in (col + 1)..10 {
            let a = m[r][col].abs();
            if a > piv_abs {
                piv_abs = a;
                piv = r;
            }
        }
        if piv_abs < 1e-12 {
            return false;
        }
        if piv != col {
            m.swap(col, piv);
        }

        // Scale pivot row so m[col][col] = 1.
        let inv = 1.0 / m[col][col];
        for k in col..20 {
            m[col][k] *= inv;
        }

        // Eliminate column `col` in every other row.
        for r in 0..10 {
            if r == col {
                continue;
            }
            let factor = m[r][col];
            if factor == 0.0 {
                continue;
            }
            for k in col..20 {
                m[r][k] -= factor * m[col][k];
            }
        }
    }
    true
}

/// Back-substitute a real z-root into the reduced constraint system to
/// recover `(x, y)`, then assemble `E = x·X + y·Y + z·Z + W`.
#[allow(dead_code)]
fn assemble_e(null_basis: &[[f64; 9]; 4], x: f64, y: f64, z: f64) -> Mat3F64 {
    let mut e = [0.0f64; 9];
    for k in 0..9 {
        e[k] = x * null_basis[0][k] + y * null_basis[1][k] + z * null_basis[2][k] + null_basis[3][k];
    }
    vec9_to_mat3(&e)
}

/// Nistér 5-point essential-matrix solver.
///
/// Takes 5 pre-normalized correspondences `x̂_i = K⁻¹ x_i` (inhomogeneous
/// (u, v), not the homogeneous (u, v, 1)) in both views. Returns up to 10
/// candidate essential matrices; downstream code picks among them by
/// cheirality or per-sample inlier scoring.
///
/// Returns an empty `Vec` when the 5-point sample is degenerate — coplanar
/// back-projected rays, duplicated points, or a null-space rank > 4.
///
/// # Status
///
/// **WIP.** This commit lands the numerical infrastructure — null-space
/// extraction, cubic constraint assembly, and Gauss-Jordan reduction — all
/// verified by unit tests (`test_null_space_5x9_*`,
/// `test_constraint_matrix_vanishes_on_true_e`,
/// `test_gauss_jordan_produces_identity_on_left_block`).
///
/// The remaining pipeline — action-matrix construction (correctly
/// pivoting on the degree-3 monomials, not the degree-≤-2 ones),
/// Faddeev-Leverrier characteristic-polynomial extraction, and
/// Durand-Kerner real-root finding — lands in the follow-up commit. No
/// external eigendecomposition is used; the whole solver stays in
/// hand-rolled fixed-size (10×10) linear algebra to keep the RANSAC
/// inner-loop predictable. Until the follow-up lands, this entry point
/// returns an empty `Vec` so it's safe to call but never contributes a
/// candidate.
pub fn essential_5pt(x1_norm: &[Vec2F64; 5], x2_norm: &[Vec2F64; 5]) -> Vec<Mat3F64> {
    let basis = match null_space_5x9(x1_norm, x2_norm) {
        Some(b) => b,
        None => return Vec::new(),
    };

    let mut cm = build_constraint_matrix(&basis);
    if !gauss_jordan_10x20(&mut cm) {
        return Vec::new();
    }

    // See the # Status note above — pipeline intentionally bails here
    // until the action-matrix + root-finder land. `assemble_e` and the
    // constraint matrix `cm` are retained so the next commit only adds;
    // it doesn't have to rewrite.
    let _ = cm;
    let _ = basis;
    Vec::new()
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
pub(crate) fn __test_gauss_jordan_10x20(m: &mut [[f64; 20]; 10]) -> bool {
    gauss_jordan_10x20(m)
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

    /// GJ must produce a left 10×10 identity block for non-degenerate inputs.
    #[test]
    fn test_gauss_jordan_produces_identity_on_left_block() {
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
        let ok = __test_gauss_jordan_10x20(&mut cm);
        assert!(ok, "Gauss-Jordan should not fail on a non-degenerate sample");

        for i in 0..10 {
            for j in 0..10 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (cm[i][j] - expected).abs() < 1e-9,
                    "GJ left block [{i}][{j}] = {} (expected {expected})",
                    cm[i][j]
                );
            }
        }
    }
}
