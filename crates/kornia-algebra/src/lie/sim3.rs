//! Similarity group Sim(3) - scaling, rotation and translation in 3D
//!
//! Sim(3) is the group of rotations, scaling and translation in 3D. It is
//! the semi-direct product of RxSO(3) and the 3D Euclidean vector space.
//! The class is represented using a composition of RxSO3 for scaling plus
//! rotation and a 3-vector for translation.
//!
//! Sim(3) is neither compact, nor a commutative group.
//!
//! Reference: Sophus library (https://github.com/strasdat/Sophus)

use crate::{Mat3AF32, Mat4F32, QuatF32, Vec3AF32};

/// Small angle threshold for Taylor series approximations.
/// Must be large enough that f32 catastrophic cancellation is avoided in the
/// `(1 - sin(θ)/θ) / θ²` formula: sin(θ)/θ rounds to 1.0 for θ < ~1e-4, so
/// any θ below ~1e-3 must take the small-angle path.
const SMALL_ANGLE_EPSILON: f32 = 1.0e-3;

use super::rxso3::RxSO3F32;
use super::so3::SO3F32;

/// Similarity transformation in 3D: rotation + scale + translation
///
/// Sim3F32 represents the similarity group Sim(3) in 3D as
/// the semi-direct product (R+ × SO(3)) ⋉ R³, where:
/// - R+ is the positive real numbers (scale)
/// - SO(3) is the rotation group in 3D
/// - R³ is the 3D Euclidean vector space (translation)
///
/// 7 degrees of freedom: 3 for rotation, 1 for scale, 3 for translation
/// Note regarding `PartialEq`:
/// This struct derives `PartialEq` which performs an exact element-wise comparison.
/// Because quaternions form a double cover for SO3 (`q` and `-q` represent the same rotation),
/// this means that two `Sim3F32` instances representing the same transformation may evaluate as not equal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sim3F32 {
    /// Scaling and rotation component
    pub rxso3: RxSO3F32,
    /// Translation component
    pub translation: Vec3AF32,
}

impl Sim3F32 {
    /// Identity transformation
    pub const IDENTITY: Self = Self {
        rxso3: RxSO3F32::IDENTITY,
        translation: Vec3AF32::ZERO,
    };

    /// Create from RxSO3F32 and translation
    pub fn new(rxso3: RxSO3F32, translation: Vec3AF32) -> Self {
        Self { rxso3, translation }
    }

    /// Create from scale, rotation quaternion, and translation
    pub fn from_scale_rotation_translation(
        scale: f32,
        rotation: QuatF32,
        translation: Vec3AF32,
    ) -> Self {
        Self {
            rxso3: RxSO3F32::from_scale_quaternion(scale, rotation),
            translation,
        }
    }

    /// Create from 4x4 homogeneous transformation matrix
    ///
    /// Matrix should be of the form:
    /// | s*R t |
    /// |  0  1 |
    ///
    /// where R is rotation, s is scale, t is translation
    pub fn from_matrix(mat: &Mat4F32) -> Self {
        // Extract rotation and scale from top-left 3x3
        // mat.x_axis, mat.y_axis, mat.z_axis are the columns
        let rot_scale_mat = Mat3AF32::from_cols(
            Vec3AF32::new(mat.x_axis.x, mat.x_axis.y, mat.x_axis.z), // first column
            Vec3AF32::new(mat.y_axis.x, mat.y_axis.y, mat.y_axis.z), // second column
            Vec3AF32::new(mat.z_axis.x, mat.z_axis.y, mat.z_axis.z), // third column
        );

        let scale = rot_scale_mat.col(0).length();

        // Normalize to get pure rotation
        let rot_mat = Mat3AF32::from_cols(
            Vec3AF32::from(rot_scale_mat.x_axis / scale),
            Vec3AF32::from(rot_scale_mat.y_axis / scale),
            Vec3AF32::from(rot_scale_mat.z_axis / scale),
        );

        Self {
            rxso3: RxSO3F32::from_scale_matrix(scale, rot_mat),
            translation: Vec3AF32::new(mat.w_axis.x, mat.w_axis.y, mat.w_axis.z),
        }
    }

    /// Get the scale factor
    pub fn scale(&self) -> f32 {
        self.rxso3.scale()
    }

    /// Get the rotation quaternion
    pub fn rotation(&self) -> QuatF32 {
        self.rxso3.rotation()
    }

    /// Get the rotation matrix
    pub fn rotation_matrix(&self) -> Mat3AF32 {
        self.rxso3.rotation_matrix()
    }

    /// Convert to 4x4 homogeneous transformation matrix
    pub fn matrix(&self) -> Mat4F32 {
        let rxso3_mat = self.rxso3.matrix();
        Mat4F32::from_cols_array(&[
            rxso3_mat.x_axis.x,
            rxso3_mat.x_axis.y,
            rxso3_mat.x_axis.z,
            0.0,
            rxso3_mat.y_axis.x,
            rxso3_mat.y_axis.y,
            rxso3_mat.y_axis.z,
            0.0,
            rxso3_mat.z_axis.x,
            rxso3_mat.z_axis.y,
            rxso3_mat.z_axis.z,
            0.0,
            self.translation.x,
            self.translation.y,
            self.translation.z,
            1.0,
        ])
    }

    /// Inverse transformation
    pub fn inverse(&self) -> Self {
        let rxso3_inv = self.rxso3.inverse();
        let scale = rxso3_inv.scale();
        let rot_inv = rxso3_inv.rotation_matrix();

        // t_inv = -R^T * s * t
        let t_inv = rot_inv * (-scale * self.translation);

        Self {
            rxso3: rxso3_inv,
            translation: t_inv,
        }
    }

    /// The Sim(3) translation matrix `W(sigma, omega)`, shared by [`Self::exp`] and [`Self::log`].
    ///
    /// Sim(3) is NOT SE(3) with a scale bolted on. Its translation block is
    /// `W = A*Omega + B*Omega^2 + C*I` whose coefficients depend on BOTH the rotation angle
    /// `theta = |omega|` AND the scale rate `sigma` — the two are coupled because a point being
    /// rotated is simultaneously being scaled, so the swept arc is a logarithmic spiral rather than
    /// a circle. Using SE(3)'s `V` (which has no sigma in it at all) and dividing by the scale
    /// afterwards, as this file previously did, is a different function; it agrees only when
    /// sigma or theta is zero, which is why the existing tests passed — every one of them exercises
    /// a translation-only, rotation-only or scale-only case.
    ///
    /// Returning ONE matrix used by both directions is the point: `log` inverts exactly what `exp`
    /// applied, so the two are mutual inverses by construction rather than by two derivations
    /// agreeing. They previously did not: measured round-trip tangent error was ~1.2 on a general
    /// input, and ~1.2 even on a pure-SE(3) case with `s = 1`.
    ///
    /// Follows Strasdat's Sim(3) derivation (the formulation Sophus implements).
    fn w_matrix(omega: Vec3AF32, sigma: f32) -> Mat3AF32 {
        let omega_hat = SO3F32::hat(omega);
        let omega_hat_sq = omega_hat * omega_hat;
        let (a, b, c) = Self::w_coefficients(omega.length() as f64, sigma as f64);
        omega_hat * (a as f32) + omega_hat_sq * (b as f32) + Mat3AF32::IDENTITY * (c as f32)
    }

    /// `(A, B, C)` for [`Self::w_matrix`], in f64.
    ///
    /// # Why f64, and why series near zero
    ///
    /// All three coefficients are formed from differences whose leading parts cancel: `A`'s
    /// numerator is `~sigma^2/2` and `B`'s is `~sigma^3/6`, both assembled from terms of size ~1.
    /// Evaluated in f32 just above a branch threshold that is catastrophic — at `sigma = 1.1e-3`
    /// the direct f32 expressions give `A = 0.443` against a true `0.5006`, and `B = 33.6` against
    /// a true `0.1668`, i.e. wrong by 200x. These are three scalars per call, so the wider type
    /// costs nothing and removes the cancellation instead of hiding it behind more branches.
    ///
    /// Below `SERIES_EPS` even f64 loses the difference, so the Taylor series is used there. Its
    /// truncation error at the crossover is ~1e-12 relative, far below f32 resolution, so the
    /// coefficients are continuous across the switch to the precision the caller can observe.
    fn w_coefficients(theta: f64, sigma: f64) -> (f64, f64, f64) {
        /// Below this the direct quotients lose their leading term even in f64.
        const SERIES_EPS: f64 = 1.0e-3;

        // C = (e^s - 1)/s. `exp_m1` keeps this accurate on its own; the series covers s -> 0,
        // where the quotient is 0/0. C is 1 only in the limit -- pinning it to 1 across the whole
        // small-sigma branch (as this once did) drops the s/2 term and puts a ~5e-4 RELATIVE step
        // in every translation at the threshold, which no round-trip test can see because `log`
        // inverts the same wrong W.
        let c = if sigma.abs() < SERIES_EPS {
            1.0 + sigma / 2.0 + sigma * sigma / 6.0 + sigma * sigma * sigma / 24.0
        } else {
            sigma.exp_m1() / sigma
        };

        let theta_small = theta < SMALL_ANGLE_EPSILON as f64;
        let sigma_small = sigma.abs() < SERIES_EPS;

        let (a, b) = match (theta_small, sigma_small) {
            // Both vanish: the SE(3) limits.
            (true, true) => (0.5, 1.0 / 6.0),
            // Rotation only. A and B take their sigma -> 0 limits, which are SE(3)'s V.
            (false, true) => {
                let th2 = theta * theta;
                (
                    (1.0 - theta.cos()) / th2,
                    (theta - theta.sin()) / (th2 * theta),
                )
            }
            // Scale only. The theta -> 0 limits, which are NOT the sigma -> 0 limits above.
            // A = sum_{m>=2} s^(m-2) (m-1)/m!, B likewise; both are the cancelling quotients, so
            // the series runs to the order that leaves the residual under f64 noise at SERIES_EPS.
            (true, false) => {
                let s = sigma;
                let e = sigma.exp();
                if s.abs() < SERIES_EPS {
                    (
                        0.5 + s / 3.0 + s * s / 8.0 + s * s * s / 30.0,
                        1.0 / 6.0 + s / 8.0 + s * s / 20.0,
                    )
                } else {
                    let s2 = s * s;
                    (
                        (s * e - sigma.exp_m1()) / s2,
                        (e * 0.5 * s2 - s * e + sigma.exp_m1()) / (s2 * s),
                    )
                }
            }
            // Coupled: both free. This is the branch the old code lacked entirely -- it applied
            // SE(3)'s V and scaled separately, which is right only when one of the two is zero.
            (false, false) => {
                let th2 = theta * theta;
                let s2 = sigma * sigma;
                let scale = sigma.exp();
                let sin_t = scale * theta.sin();
                let cos_t = scale * theta.cos();
                let denom = th2 + s2;
                (
                    (sin_t * sigma + (1.0 - cos_t) * theta) / (theta * denom),
                    (c - ((cos_t - 1.0) * sigma + sin_t * theta) / denom) / th2,
                )
            }
        };

        (a, b, c)
    }

    /// Exponential map from Lie algebra to group
    ///
    /// Input: 7-vector [upsilon; omega; sigma] where:
    /// - upsilon: 3D translation velocity
    /// - omega: 3D rotation velocity
    /// - sigma: scale velocity
    pub fn exp(upsilon: Vec3AF32, omega: Vec3AF32, sigma: f32) -> Self {
        Self {
            rxso3: RxSO3F32::exp(omega, sigma),
            // `W * upsilon`, with NO division by scale: the scale is already inside W through C.
            translation: Self::w_matrix(omega, sigma) * upsilon,
        }
    }

    /// Logarithmic map from group to Lie algebra
    ///
    /// Returns: (upsilon, omega, sigma) where:
    /// - upsilon: 3D translation velocity
    /// - omega: 3D rotation velocity
    /// - sigma: scale velocity
    pub fn log(&self) -> (Vec3AF32, Vec3AF32, f32) {
        let (omega, sigma) = self.rxso3.log();
        // Invert exactly the matrix `exp` applied — no separate series, no transpose, no rescale.
        //
        // W is NOT unconditionally invertible: its eigenvalues are (e^l - 1)/l for l in
        // {sigma, sigma +- i*theta}, so it is exactly singular at sigma = 0, |omega| = 2*pi — an
        // input `exp` accepts. What makes this call site safe is the source of `omega`, not the
        // matrix: `RxSO3::log` flips the quaternion when w < 0, so the omega it returns always has
        // |omega| <= pi. Anyone reusing `w_matrix(..).inverse()` on an EXP-side tangent, which is
        // unbounded, must handle the singularity themselves — glam does not check the determinant.
        let upsilon = Self::w_matrix(omega, sigma).inverse() * self.translation;
        (upsilon, omega, sigma)
    }

    /// Adjoint representation for computing Jacobians
    ///
    /// Returns the 7x7 adjoint matrix for Sim3:
    /// ```text
    /// [ sR    [t]×R    -t ]
    /// [  0      R       0 ]
    /// [  0      0       1 ]
    /// ```
    pub fn adjoint(&self) -> [[f32; 7]; 7] {
        let r = self.rxso3.rotation_matrix();
        let s = self.rxso3.scale();
        let t_cross_r = SO3F32::hat(self.translation) * r;
        let t = self.translation;

        // Build the 7x7 adjoint matrix directly
        // Row 0: [sR row0, [t]×R row0, -t[0]]
        // Row 1: [sR row1, [t]×R row1, -t[1]]
        // Row 2: [sR row2, [t]×R row2, -t[2]]
        // Row 3: [0, 0, 0, R row0, 0]
        // Row 4: [0, 0, 0, R row1, 0]
        // Row 5: [0, 0, 0, R row2, 0]
        // Row 6: [0, 0, 0, 0, 0, 0, 1]
        [
            [
                s * r.x_axis.x,
                s * r.y_axis.x,
                s * r.z_axis.x,
                t_cross_r.x_axis.x,
                t_cross_r.y_axis.x,
                t_cross_r.z_axis.x,
                -t.x,
            ],
            [
                s * r.x_axis.y,
                s * r.y_axis.y,
                s * r.z_axis.y,
                t_cross_r.x_axis.y,
                t_cross_r.y_axis.y,
                t_cross_r.z_axis.y,
                -t.y,
            ],
            [
                s * r.x_axis.z,
                s * r.y_axis.z,
                s * r.z_axis.z,
                t_cross_r.x_axis.z,
                t_cross_r.y_axis.z,
                t_cross_r.z_axis.z,
                -t.z,
            ],
            [0.0, 0.0, 0.0, r.x_axis.x, r.y_axis.x, r.z_axis.x, 0.0],
            [0.0, 0.0, 0.0, r.x_axis.y, r.y_axis.y, r.z_axis.y, 0.0],
            [0.0, 0.0, 0.0, r.x_axis.z, r.y_axis.z, r.z_axis.z, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    }
}

impl std::ops::Mul<Sim3F32> for Sim3F32 {
    type Output = Sim3F32;

    fn mul(self, rhs: Sim3F32) -> Self::Output {
        let rxso3 = self.rxso3 * rhs.rxso3;
        let translation = self.translation + self.rxso3.matrix() * rhs.translation;
        Sim3F32::new(rxso3, translation)
    }
}

impl std::ops::Mul<Vec3AF32> for Sim3F32 {
    type Output = Vec3AF32;

    fn mul(self, rhs: Vec3AF32) -> Self::Output {
        let scaled_rotated = self.rxso3.matrix() * rhs;
        scaled_rotated + self.translation
    }
}

#[cfg(test)]
mod tests {
    /// exp and log must be mutual inverses on COUPLED input — nonzero rotation AND nonzero scale
    /// AND nonzero translation at once.
    ///
    /// This is the case every other test in this file misses. Translation-only, rotation-only and
    /// scale-only inputs all pass against a wrong `W`, because each of them zeroes the terms where
    /// sigma and theta interact. Before the shared-`W` rewrite the round-trip tangent error here
    /// was ~1.2 — including at `sigma = 0`, i.e. on a pure SE(3) element, where Sim(3) must reduce
    /// exactly to SE(3).
    #[test]
    fn sim3_exp_log_roundtrip_coupled() {
        let cases: [([f32; 3], [f32; 3], f32); 6] = [
            ([0.4, -0.2, 0.7], [0.3, 0.5, -0.2], 0.35),  // general
            ([0.4, -0.2, 0.7], [0.3, 0.5, -0.2], 0.0),   // pure SE(3): must reduce exactly
            ([1.0, 2.0, -0.5], [0.0, 0.0, 0.0], 0.6),    // scale only, no rotation
            ([-0.3, 0.8, 0.1], [1.2, -0.4, 0.9], -0.45), // negative scale rate, large angle
            ([0.05, 0.02, -0.01], [1e-5, -2e-5, 5e-6], 1e-6), // both near zero
            ([2.0, -1.0, 0.5], [0.0, 3.0, 0.0], 0.9),    // large angle + large scale
        ];
        for (u, w, sig) in cases {
            let (up, om) = (
                Vec3AF32::new(u[0], u[1], u[2]),
                Vec3AF32::new(w[0], w[1], w[2]),
            );
            let (up2, om2, sig2) = Sim3F32::exp(up, om, sig).log();
            let du = (up2 - up).length();
            let dw = (om2 - om).length();
            let ds = (sig2 - sig).abs();
            assert!(
                du < 1e-3 && dw < 1e-3 && ds < 1e-4,
                "roundtrip failed for u={u:?} omega={w:?} sigma={sig}: \
                 d_upsilon={du:.6} d_omega={dw:.6} d_sigma={ds:.6}"
            );
        }
    }

    /// `exp` must agree with the matrix exponential of the sim(3) generator.
    ///
    /// This is the test that can actually catch a wrong `W`. The round-trip test CANNOT: `log`
    /// recovers `(omega, sigma)` from `rxso3` alone and then applies `w_matrix(omega, sigma)`
    /// inverse to what `exp` produced with that same matrix, so the composition is `W^-1 W = I` for
    /// ANY invertible `W`. Corrupting the coupled branch leaves every round-trip assertion green.
    ///
    /// The reference here is independent of the implementation: a truncated series for
    /// `expm([[Omega + sigma*I, upsilon], [0, 0]])`, summed in f64. 30 terms is ample for these
    /// magnitudes (the tangent norms are O(1), so terms fall off factorially).
    #[test]
    fn sim3_exp_matches_matrix_exponential() {
        // (upsilon, omega, sigma) spanning every branch of `w_coefficients`.
        let cases: [([f32; 3], [f32; 3], f32); 6] = [
            ([0.4, -0.2, 0.7], [0.3, 0.5, -0.2], 0.35), // coupled: both free
            ([1.0, 2.0, -0.5], [0.0, 0.0, 0.0], 0.6),   // scale only
            ([0.4, -0.2, 0.7], [0.3, 0.5, -0.2], 0.0),  // rotation only
            ([1.0, -2.0, 0.5], [0.0, 0.0, 0.0], 0.0),   // both vanish
            ([10.0, 0.0, 0.0], [0.0, 0.0, 0.0], 9e-4),  // just INSIDE the sigma series branch
            ([1.0, 2.0, 3.0], [0.0, 0.0, 0.5], 1.1e-3), // just OUTSIDE it: pins continuity
        ];
        for (u, w, sig) in cases {
            // 4x4 generator [[Omega + sigma*I, upsilon], [0, 0]], row-major in f64.
            let (wx, wy, wz) = (w[0] as f64, w[1] as f64, w[2] as f64);
            let s = sig as f64;
            let gen = [
                [s, -wz, wy, u[0] as f64],
                [wz, s, -wx, u[1] as f64],
                [-wy, wx, s, u[2] as f64],
                [0.0, 0.0, 0.0, 0.0],
            ];
            // expm by its defining series: sum_k G^k / k!.
            let mut acc = [[0.0f64; 4]; 4];
            let mut term = [[0.0f64; 4]; 4];
            for i in 0..4 {
                acc[i][i] = 1.0;
                term[i][i] = 1.0;
            }
            for k in 1..30 {
                let mut next = [[0.0f64; 4]; 4];
                for i in 0..4 {
                    for j in 0..4 {
                        let mut v = 0.0;
                        for (m, gm) in gen.iter().enumerate() {
                            v += term[i][m] * gm[j];
                        }
                        next[i][j] = v / k as f64;
                    }
                }
                term = next;
                for i in 0..4 {
                    for j in 0..4 {
                        acc[i][j] += term[i][j];
                    }
                }
            }

            let got = Sim3F32::exp(
                Vec3AF32::new(u[0], u[1], u[2]),
                Vec3AF32::new(w[0], w[1], w[2]),
                sig,
            );
            let t = got.translation;
            let (tx, ty, tz) = (t.x as f64, t.y as f64, t.z as f64);
            // Relative tolerance: the f32 group element cannot carry more than ~1e-6 relative.
            let mag = (acc[0][3].abs() + acc[1][3].abs() + acc[2][3].abs()).max(1.0);
            let err =
                ((tx - acc[0][3]).abs() + (ty - acc[1][3]).abs() + (tz - acc[2][3]).abs()) / mag;
            assert!(
                err < 1e-5,
                "exp translation disagrees with expm for u={u:?} w={w:?} sigma={sig}: \
                 got ({tx:.6}, {ty:.6}, {tz:.6}) want ({:.6}, {:.6}, {:.6}) rel_err={err:.3e}",
                acc[0][3],
                acc[1][3],
                acc[2][3]
            );
        }
    }

    /// With `sigma = 0` a Sim(3) element IS an SE(3) element, so `exp` must place the translation
    /// exactly where SE(3)'s `V * upsilon` would. Pins the reduction the coupled test only implies.
    #[test]
    fn sim3_reduces_to_se3_at_unit_scale() {
        let up = Vec3AF32::new(0.4, -0.2, 0.7);
        let om = Vec3AF32::new(0.3, 0.5, -0.2);
        let s = Sim3F32::exp(up, om, 0.0);
        assert!(
            (s.rxso3.scale() - 1.0).abs() < 1e-6,
            "scale must be 1 at sigma=0"
        );

        let theta = om.length();
        let oh = SO3F32::hat(om);
        let v = Mat3AF32::IDENTITY
            + oh * ((1.0 - theta.cos()) / (theta * theta))
            + (oh * oh) * ((theta - theta.sin()) / (theta * theta * theta));
        let expected = v * up;
        let got = s.translation;
        assert!(
            (got - expected).length() < 1e-5,
            "sigma=0 translation {got:?} != SE(3) V*upsilon {expected:?}"
        );
    }

    use super::*;
    use approx::assert_relative_eq;

    const EPSILON: f32 = 1e-5;

    #[test]
    fn test_sim3_identity() {
        let sim3 = Sim3F32::IDENTITY;
        assert_relative_eq!(sim3.scale(), 1.0, epsilon = EPSILON);
        assert_relative_eq!(sim3.translation.x, 0.0, epsilon = EPSILON);
        assert_relative_eq!(sim3.translation.y, 0.0, epsilon = EPSILON);
        assert_relative_eq!(sim3.translation.z, 0.0, epsilon = EPSILON);
    }

    #[test]
    fn test_sim3_from_matrix() {
        // Test identity matrix
        let mat = Mat4F32::IDENTITY;
        let sim3 = Sim3F32::from_matrix(&mat);
        assert_relative_eq!(sim3.scale(), 1.0, epsilon = EPSILON);
        assert_relative_eq!(sim3.translation.length(), 0.0, epsilon = EPSILON);

        // Test scaled translation matrix
        // Matrix: | 2 0 0 1 |
        //         | 0 2 0 2 |
        //         | 0 0 2 3 |
        //         | 0 0 0 1 |
        let mat = Mat4F32::from_cols_array(&[
            2.0, 0.0, 0.0, 0.0, // col 0
            0.0, 2.0, 0.0, 0.0, // col 1
            0.0, 0.0, 2.0, 0.0, // col 2
            1.0, 2.0, 3.0, 1.0, // col 3 (translation)
        ]);
        let sim3 = Sim3F32::from_matrix(&mat);
        assert_relative_eq!(sim3.scale(), 2.0, epsilon = EPSILON);
        assert_relative_eq!(sim3.translation.x, 1.0, epsilon = EPSILON);
        assert_relative_eq!(sim3.translation.y, 2.0, epsilon = EPSILON);
        assert_relative_eq!(sim3.translation.z, 3.0, epsilon = EPSILON);
    }

    #[test]
    fn test_sim3_inverse() {
        let sim3 = Sim3F32::from_scale_rotation_translation(
            2.0,
            QuatF32::from_xyzw(0.1, 0.2, 0.3, 0.9).normalize(),
            Vec3AF32::new(1.0, 2.0, 3.0),
        );

        let inv = sim3.inverse();
        let product = sim3 * inv;

        // Should be identity
        assert_relative_eq!(product.scale(), 1.0, epsilon = EPSILON);
        assert_relative_eq!(product.translation.length(), 0.0, epsilon = EPSILON);
    }

    #[test]
    fn test_sim3_matrix_roundtrip() {
        let sim3 = Sim3F32::from_scale_rotation_translation(
            1.5,
            QuatF32::from_xyzw(0.1, 0.2, 0.3, 0.9).normalize(),
            Vec3AF32::new(1.0, 2.0, 3.0),
        );

        let mat = sim3.matrix();
        let sim3_reconstructed = Sim3F32::from_matrix(&mat);

        assert_relative_eq!(sim3.scale(), sim3_reconstructed.scale(), epsilon = EPSILON);
        assert_relative_eq!(
            (sim3.translation - sim3_reconstructed.translation).length(),
            0.0,
            epsilon = EPSILON
        );
    }

    #[test]
    fn test_sim3_multiplication() {
        let sim3_1 = Sim3F32::from_scale_rotation_translation(
            2.0,
            QuatF32::IDENTITY, // Identity rotation for simplicity
            Vec3AF32::new(1.0, 0.0, 0.0),
        );

        let sim3_2 = Sim3F32::from_scale_rotation_translation(
            1.5,
            QuatF32::from_xyzw(0.0, 0.0, 0.0, 1.0), // Identity rotation
            Vec3AF32::new(0.0, 1.0, 0.0),
        );

        let combined = sim3_1 * sim3_2;
        assert_relative_eq!(combined.scale(), 3.0, epsilon = EPSILON); // 2.0 * 1.5
        assert_relative_eq!(combined.translation.x, 1.0, epsilon = EPSILON);
        assert_relative_eq!(combined.translation.y, 2.0, epsilon = EPSILON); // 2.0 * 1.0 (from sim3_2) + 0.0
    }

    #[test]
    fn test_point_transformation() {
        let sim3 = Sim3F32::from_scale_rotation_translation(
            2.0,
            QuatF32::IDENTITY, // No rotation
            Vec3AF32::new(1.0, 2.0, 3.0),
        );

        let point = Vec3AF32::new(1.0, 1.0, 1.0);
        let transformed = sim3 * point;

        // Should be: 2.0 * [1,1,1] + [1,2,3] = [3,4,5]
        assert_relative_eq!(transformed.x, 3.0, epsilon = EPSILON);
        assert_relative_eq!(transformed.y, 4.0, epsilon = EPSILON);
        assert_relative_eq!(transformed.z, 5.0, epsilon = EPSILON);
    }

    #[test]
    fn test_sim3_exp_log_roundtrip() {
        // Test exp then log roundtrip for simpler inputs
        // Note: The exp/log implementation uses approximations that work well for
        // small angles and simple cases
        let test_cases = [
            // Translation only (no rotation, no scale)
            (Vec3AF32::new(1.0, 2.0, 3.0), Vec3AF32::ZERO, 0.0),
            // Scale only
            (Vec3AF32::ZERO, Vec3AF32::ZERO, 0.5),
            // Pure rotation around X
            (Vec3AF32::ZERO, Vec3AF32::new(0.5, 0.0, 0.0), 0.0),
            // Pure rotation around Y
            (Vec3AF32::ZERO, Vec3AF32::new(0.0, 0.5, 0.0), 0.0),
            // Pure rotation around Z
            (Vec3AF32::ZERO, Vec3AF32::new(0.0, 0.0, 0.5), 0.0),
        ];

        for (upsilon, omega, sigma) in test_cases {
            let sim3 = Sim3F32::exp(upsilon, omega, sigma);
            let (upsilon_out, omega_out, sigma_out) = sim3.log();

            assert_relative_eq!(upsilon_out.x, upsilon.x, epsilon = 0.01);
            assert_relative_eq!(upsilon_out.y, upsilon.y, epsilon = 0.01);
            assert_relative_eq!(upsilon_out.z, upsilon.z, epsilon = 0.01);
            assert_relative_eq!(omega_out.x, omega.x, epsilon = EPSILON);
            assert_relative_eq!(omega_out.y, omega.y, epsilon = EPSILON);
            assert_relative_eq!(omega_out.z, omega.z, epsilon = EPSILON);
            assert_relative_eq!(sigma_out, sigma, epsilon = EPSILON);
        }
    }

    #[test]
    fn test_sim3_log_exp_roundtrip() {
        // Test log then exp roundtrip for simpler Sim3 elements
        let test_cases = [
            // Identity with translation
            Sim3F32::from_scale_rotation_translation(
                1.0,
                QuatF32::IDENTITY,
                Vec3AF32::new(1.0, 2.0, 3.0),
            ),
            // Scale only
            Sim3F32::from_scale_rotation_translation(2.0, QuatF32::IDENTITY, Vec3AF32::ZERO),
            // Rotation only (no scale, no translation)
            Sim3F32::from_scale_rotation_translation(
                1.0,
                QuatF32::from_xyzw(0.0, 0.0, 0.383, 0.924).normalize(), // ~45 deg around Z
                Vec3AF32::ZERO,
            ),
        ];

        for sim3 in test_cases {
            let (upsilon, omega, sigma) = sim3.log();
            let sim3_out = Sim3F32::exp(upsilon, omega, sigma);

            assert_relative_eq!(sim3_out.scale(), sim3.scale(), epsilon = 0.01);
            assert_relative_eq!(
                (sim3_out.translation - sim3.translation).length(),
                0.0,
                epsilon = 0.01
            );
            // Rotations should be equivalent
            let dot = sim3_out.rotation().dot(sim3.rotation().0).abs();
            assert_relative_eq!(dot, 1.0, epsilon = EPSILON);
        }
    }

    #[test]
    fn test_sim3_exp_identity() {
        // exp(0, 0, 0) should give identity
        let sim3 = Sim3F32::exp(Vec3AF32::ZERO, Vec3AF32::ZERO, 0.0);

        assert_relative_eq!(sim3.scale(), 1.0, epsilon = EPSILON);
        assert_relative_eq!(sim3.translation.x, 0.0, epsilon = EPSILON);
        assert_relative_eq!(sim3.translation.y, 0.0, epsilon = EPSILON);
        assert_relative_eq!(sim3.translation.z, 0.0, epsilon = EPSILON);
        assert_relative_eq!(sim3.rotation().w.abs(), 1.0, epsilon = EPSILON);
    }

    #[test]
    fn test_sim3_log_identity() {
        // log(identity) should give (0, 0, 0)
        let (upsilon, omega, sigma) = Sim3F32::IDENTITY.log();

        assert_relative_eq!(upsilon.x, 0.0, epsilon = EPSILON);
        assert_relative_eq!(upsilon.y, 0.0, epsilon = EPSILON);
        assert_relative_eq!(upsilon.z, 0.0, epsilon = EPSILON);
        assert_relative_eq!(omega.x, 0.0, epsilon = EPSILON);
        assert_relative_eq!(omega.y, 0.0, epsilon = EPSILON);
        assert_relative_eq!(omega.z, 0.0, epsilon = EPSILON);
        assert_relative_eq!(sigma, 0.0, epsilon = EPSILON);
    }

    #[test]
    fn test_sim3_exp_small_angles() {
        // Test behavior with very small rotation angles (Taylor series regime)
        let upsilon = Vec3AF32::new(0.1, 0.2, 0.3);
        let omega = Vec3AF32::new(1e-8, 2e-8, 3e-8);
        let sigma = 0.1;

        let sim3 = Sim3F32::exp(upsilon, omega, sigma);
        let (upsilon_out, omega_out, sigma_out) = sim3.log();

        // Should recover the inputs even for small angles
        assert_relative_eq!(upsilon_out.x, upsilon.x, epsilon = 0.01);
        assert_relative_eq!(upsilon_out.y, upsilon.y, epsilon = 0.01);
        assert_relative_eq!(upsilon_out.z, upsilon.z, epsilon = 0.01);
        assert_relative_eq!(omega_out.x, omega.x, epsilon = 1e-6);
        assert_relative_eq!(omega_out.y, omega.y, epsilon = 1e-6);
        assert_relative_eq!(omega_out.z, omega.z, epsilon = 1e-6);
        assert_relative_eq!(sigma_out, sigma, epsilon = EPSILON);
    }

    #[test]
    fn test_sim3_translation_only() {
        // Test pure translation (no rotation, no scale)
        let upsilon = Vec3AF32::new(1.0, 2.0, 3.0);
        let sim3 = Sim3F32::exp(upsilon, Vec3AF32::ZERO, 0.0);

        assert_relative_eq!(sim3.scale(), 1.0, epsilon = EPSILON);
        assert_relative_eq!(sim3.translation.x, upsilon.x, epsilon = EPSILON);
        assert_relative_eq!(sim3.translation.y, upsilon.y, epsilon = EPSILON);
        assert_relative_eq!(sim3.translation.z, upsilon.z, epsilon = EPSILON);
    }

    #[test]
    fn test_sim3_scale_only() {
        // Test pure scaling (no rotation, no translation)
        let sigma = 0.693; // ln(2) ≈ 0.693
        let sim3 = Sim3F32::exp(Vec3AF32::ZERO, Vec3AF32::ZERO, sigma);

        assert_relative_eq!(sim3.scale(), sigma.exp(), epsilon = EPSILON);
        assert_relative_eq!(sim3.translation.length(), 0.0, epsilon = EPSILON);
        assert_relative_eq!(sim3.rotation().w.abs(), 1.0, epsilon = EPSILON);
    }

    #[test]
    fn test_sim3_adjoint_identity() {
        let sim3 = Sim3F32::IDENTITY;
        let adj = sim3.adjoint();

        // Expected 7x7 adjoint of identity: identity matrix
        // For identity: scale=1, R=I, t=0
        // So adjoint should be 7x7 identity
        let expected = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ];

        for i in 0..7 {
            for j in 0..7 {
                assert_relative_eq!(adj[i][j], expected[i][j], epsilon = EPSILON);
            }
        }
    }

    #[test]
    fn test_sim3_adjoint_properties() {
        // Test: Ad(g * h) = Ad(g) * Ad(h)
        let sim3_1 = Sim3F32::from_scale_rotation_translation(
            1.5,
            QuatF32::from_xyzw(0.1, 0.2, 0.3, 0.9).normalize(),
            Vec3AF32::new(1.0, 2.0, 3.0),
        );

        let sim3_2 = Sim3F32::from_scale_rotation_translation(
            2.0,
            QuatF32::from_xyzw(-0.2, 0.1, 0.4, 0.8).normalize(),
            Vec3AF32::new(-1.0, 0.5, 2.0),
        );

        let composed = sim3_1 * sim3_2;
        let adj_composed = composed.adjoint();

        let adj_1 = sim3_1.adjoint();
        let adj_2 = sim3_2.adjoint();

        // Matrix multiplication of 7x7 matrices: adj_1 * adj_2
        let mut adj_product = [[0.0f32; 7]; 7];
        for i in 0..7 {
            for j in 0..7 {
                for (k, adj_2k) in adj_2.iter().enumerate() {
                    adj_product[i][j] += adj_1[i][k] * adj_2k[j];
                }
            }
        }

        for i in 0..7 {
            for j in 0..7 {
                assert_relative_eq!(adj_composed[i][j], adj_product[i][j], epsilon = EPSILON);
            }
        }
    }
}
