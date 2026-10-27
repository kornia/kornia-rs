use super::so3::{SO3F32, SO3F64};
use crate::{
    param::{Param, ParamError},
    Mat3AF32, Mat3F64, Mat4F32, Mat4F64, QuatF32, QuatF64, Vec3AF32, Vec3F64,
};
use rand::RngExt;

const SMALL_ANGLE_EPSILON: f32 = 1.0e-8;

/// θ² below which [`SE3F32::left_jacobian`] uses the Taylor series of the
/// `Q` coefficients: their closed forms cancel catastrophically in f32 below
/// ~0.1 rad.
const Q_SERIES_THETA2: f32 = 1.0e-2;

/// θ² below which [`SE3F64`] uses the Taylor series of the `V⁻¹` and `Q`
/// coefficients, whose closed forms lose digits to cancellation there.
const SERIES_THETA2_F64: f64 = 1.0e-4;

/// Note regarding `PartialEq`:
/// This struct derives `PartialEq` which performs an exact element-wise comparison.
/// Because quaternions form a double cover for SO3 (`q` and `-q` represent the same rotation),
/// this means that two `SE3F32` instances representing the same transformation may evaluate as not equal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SE3F32 {
    pub r: SO3F32,
    pub t: Vec3AF32,
}

impl SE3F32 {
    pub const IDENTITY: Self = Self {
        r: SO3F32::IDENTITY,
        t: Vec3AF32::ZERO,
    };

    pub fn new(rotation: SO3F32, translation: Vec3AF32) -> Self {
        Self {
            r: rotation,
            t: translation,
        }
    }

    pub fn from_array(arr: [f32; 7]) -> Self {
        Self {
            r: SO3F32::from_array([arr[0], arr[1], arr[2], arr[3]]),
            t: Vec3AF32::from_array([arr[4], arr[5], arr[6]]),
        }
    }

    pub fn to_array(&self) -> [f32; 7] {
        let r = self.r.to_array();
        let t = self.t.to_array();
        [r[0], r[1], r[2], r[3], t[0], t[1], t[2]]
    }

    pub fn from_matrix(mat: &Mat4F32) -> Self {
        Self {
            r: SO3F32::from_matrix4(mat),
            t: Vec3AF32::new(mat.w_axis.x, mat.w_axis.y, mat.w_axis.z),
        }
    }

    pub fn from_random() -> Self {
        let mut rng = rand::rng();

        let r1: f32 = rng.random();
        let r2: f32 = rng.random();
        let r3: f32 = rng.random();

        Self {
            r: SO3F32::from_random(),
            t: Vec3AF32::new(r1, r2, r3),
        }
    }

    pub fn from_qxyz(quat: QuatF32, xyz: Vec3AF32) -> Self {
        Self {
            r: SO3F32::from_quaternion(quat),
            t: xyz,
        }
    }

    pub fn from_params(params: &[f32]) -> Self {
        debug_assert!(
            params.len() >= 7,
            "SE3F32::from_params requires at least 7 elements"
        );
        // Input is [qw, qx, qy, qz, tx, ty, tz]
        Self {
            r: SO3F32::from_array([params[1], params[2], params[3], params[0]]),
            t: Vec3AF32::new(params[4], params[5], params[6]),
        }
    }

    pub fn to_params(&self) -> [f32; 7] {
        let r = self.r.to_array();
        let t = self.t.to_array();
        // Return [qw, qx, qy, qz, tx, ty, tz] for optimizer compatibility
        [r[3], r[0], r[1], r[2], t[0], t[1], t[2]]
    }

    pub fn retract(&self, delta: &[f32]) -> Self {
        debug_assert!(
            delta.len() >= 6,
            "SE3F32::retract requires at least 6 elements"
        );
        let upsilon = Vec3AF32::new(delta[0], delta[1], delta[2]);
        let omega = Vec3AF32::new(delta[3], delta[4], delta[5]);
        self.rplus(upsilon, omega)
    }

    #[inline]
    pub fn rplus(&self, upsilon: Vec3AF32, omega: Vec3AF32) -> Self {
        *self * SE3F32::exp(upsilon, omega)
    }

    #[inline]
    pub fn rminus(&self, other: &Self) -> (Vec3AF32, Vec3AF32) {
        (self.inverse() * *other).log()
    }

    #[inline]
    pub fn lplus(upsilon: Vec3AF32, omega: Vec3AF32, x: &Self) -> Self {
        SE3F32::exp(upsilon, omega) * *x
    }

    #[inline]
    pub fn lminus(y: &Self, x: &Self) -> (Vec3AF32, Vec3AF32) {
        (*y * x.inverse()).log()
    }

    pub fn inverse(&self) -> Self {
        let r_inv = self.r.inverse();
        Self {
            r: r_inv,
            t: r_inv * (-self.t),
        }
    }

    pub fn matrix(&self) -> Mat4F32 {
        let r = self.r.matrix();
        Mat4F32::from_cols_array(&[
            r.x_axis.x, r.x_axis.y, r.x_axis.z, 0.0, // col0
            r.y_axis.x, r.y_axis.y, r.y_axis.z, 0.0, // col1
            r.z_axis.x, r.z_axis.y, r.z_axis.z, 0.0, // col2
            self.t.x, self.t.y, self.t.z, 1.0, // col3
        ])
    }

    /// Adjoint for the `[υ; ω]` tangent order of [`Self::exp`]:
    /// `Ad(T) = [[R, t×R], [0, R]]`, so that `T · Exp(ξ) · T⁻¹ = Exp(Ad(T) · ξ)`.
    pub fn adjoint(&self) -> [[f32; 6]; 6] {
        let r = self.r.matrix();
        let t = SO3F32::hat(self.t) * r;

        [
            [
                r.x_axis.x, r.y_axis.x, r.z_axis.x, t.x_axis.x, t.y_axis.x, t.z_axis.x,
            ],
            [
                r.x_axis.y, r.y_axis.y, r.z_axis.y, t.x_axis.y, t.y_axis.y, t.z_axis.y,
            ],
            [
                r.x_axis.z, r.y_axis.z, r.z_axis.z, t.x_axis.z, t.y_axis.z, t.z_axis.z,
            ],
            [0.0, 0.0, 0.0, r.x_axis.x, r.y_axis.x, r.z_axis.x],
            [0.0, 0.0, 0.0, r.x_axis.y, r.y_axis.y, r.z_axis.y],
            [0.0, 0.0, 0.0, r.x_axis.z, r.y_axis.z, r.z_axis.z],
        ]
    }

    pub fn exp(upsilon: Vec3AF32, omega: Vec3AF32) -> Self {
        let theta2 = omega.dot(omega);
        let theta = theta2.sqrt();

        Self {
            r: SO3F32::exp(omega),
            t: if theta > SMALL_ANGLE_EPSILON {
                let omega_hat = SO3F32::hat(omega);
                let omega_hat_sq = omega_hat * omega_hat;

                let a = (1.0 - theta.cos()) / theta2;
                let b = (theta - theta.sin()) / (theta2 * theta);

                let mat_v = Mat3AF32::IDENTITY + a * omega_hat + b * omega_hat_sq;

                mat_v * upsilon
            } else {
                // Small-angle approximation: t = I * upsilon
                upsilon
            },
        }
    }

    /// returns translation, rotation
    pub fn log(&self) -> (Vec3AF32, Vec3AF32) {
        let omega = self.r.log();
        let theta2 = omega.dot(omega);
        let theta = theta2.sqrt();
        (
            if theta > SMALL_ANGLE_EPSILON {
                let omega_hat = SO3F32::hat(omega);
                let omega_hat_sq = omega_hat * omega_hat;
                let mat_v_inv = Mat3AF32::IDENTITY - 0.5 * omega_hat
                    + ((1.0 - theta * (theta / 2.0).cos() / (2.0 * (theta / 2.0).sin())) / theta2)
                        * omega_hat_sq;

                mat_v_inv * self.t
            } else {
                self.t
            },
            omega,
        )
    }

    pub fn hat(upsilon: Vec3AF32, omega: Vec3AF32) -> Mat4F32 {
        let h = SO3F32::hat(omega);

        Mat4F32::from_cols_array(&[
            h.x_axis.x, h.x_axis.y, h.x_axis.z, 0.0, h.y_axis.x, h.y_axis.y, h.y_axis.z, 0.0,
            h.z_axis.x, h.z_axis.y, h.z_axis.z, 0.0, upsilon.x, upsilon.y, upsilon.z, 0.0,
        ])
    }

    pub fn vee(omega: Mat4F32) -> (Vec3AF32, Vec3AF32) {
        (
            Vec3AF32::new(omega.w_axis.x, omega.w_axis.y, omega.w_axis.z),
            SO3F32::vee4(omega),
        )
    }

    /// Jₗ(ρ,θ) =
    /// ⎡ Jₗ(θ)      Q(ρ,θ) ⎤
    /// ⎢                   ⎥
    /// ⎣  0         Jₗ(θ)  ⎦
    ///
    /// Jₗ(θ)is the left Jacobian of SO(3)
    ///
    /// ```text
    ///              1            θ - sinθ
    /// Q(ρ,θ) = ───── ρ× + ───────────────── (θ×ρ× + ρ×θ× + θ×ρ×θ×)
    ///              2              θ³
    ///                       1 - θ²/2 - cosθ
    ///          - ────────────────────────── (θ²×ρ× + ρ×θ²× - 3 θ×ρ×θ×)
    ///                               θ⁴
    ///                       ⎛ 1 - θ²/2 - cosθ      θ - sinθ - θ³/6  ⎞
    ///          - ½ ⎜───────────────────────── - 3 ──────────────────⎟
    ///                       ⎝         θ⁴                    θ⁵      ⎠
    ///            · (θ×ρ×θ²× + θ²×ρ×θ×).
    /// ```
    /// ref: https://arxiv.org/pdf/1812.01537 (eq 180)
    pub fn left_jacobian(rho: Vec3AF32, omega: Vec3AF32) -> [[f32; 6]; 6] {
        let theta2 = omega.dot(omega);

        let jl_so3 = SO3F32::left_jacobian(omega);

        let rho_hat = SO3F32::hat(rho);
        let omega_hat = SO3F32::hat(omega);
        let wp = omega_hat * rho_hat;
        let pw = rho_hat * omega_hat;
        let wpw = wp * omega_hat;

        // The three coefficients of Q above, signs folded in (Barfoot, State
        // Estimation for Robotics, eq. 7.86).
        let (c1, c2, c3) = if theta2 < Q_SERIES_THETA2 {
            (
                1.0 / 6.0 - theta2 / 120.0,
                1.0 / 24.0 - theta2 / 720.0,
                1.0 / 120.0 - theta2 / 2520.0,
            )
        } else {
            let theta = theta2.sqrt();
            let (s, c) = theta.sin_cos();
            (
                (theta - s) / (theta2 * theta),
                (theta2 + 2.0 * c - 2.0) / (2.0 * theta2 * theta2),
                (2.0 * theta - 3.0 * s + theta * c) / (2.0 * theta2 * theta2 * theta),
            )
        };
        let q_mat = 0.5 * rho_hat
            + c1 * (wp + pw + wpw)
            + c2 * (omega_hat * wp + pw * omega_hat - 3.0 * wpw)
            + c3 * (wpw * omega_hat + omega_hat * wpw);

        let mut jl = [[0.0; 6]; 6];

        // top-left 3×3 : Jₗ_SO3
        for (i, col) in [jl_so3.x_axis, jl_so3.y_axis, jl_so3.z_axis]
            .iter()
            .enumerate()
        {
            jl[0][i] = col.x;
            jl[1][i] = col.y;
            jl[2][i] = col.z;
        }

        // top-right 3×3 : Q
        for (i, col) in [q_mat.x_axis, q_mat.y_axis, q_mat.z_axis]
            .iter()
            .enumerate()
        {
            jl[0][i + 3] = col.x;
            jl[1][i + 3] = col.y;
            jl[2][i + 3] = col.z;
        }

        // bottom-right 3×3 : Jₗ_SO3
        for (i, col) in [jl_so3.x_axis, jl_so3.y_axis, jl_so3.z_axis]
            .iter()
            .enumerate()
        {
            jl[3][i + 3] = col.x;
            jl[4][i + 3] = col.y;
            jl[5][i + 3] = col.z;
        }

        jl
    }

    /// Right Jacobian  Jᵣ(ρ, θ)  ∈ ℝ⁶ˣ⁶.
    /// Using Jᵣ(ρ,θ) = Jₗ(−ρ, −θ).
    pub fn right_jacobian(rho: Vec3AF32, omega: Vec3AF32) -> [[f32; 6]; 6] {
        Self::left_jacobian(-rho, -omega)
    }
}

impl Param for SE3F32 {
    const GLOBAL_SIZE: usize = 7;
    const LOCAL_SIZE: usize = 6;

    #[inline]
    fn plus(x: &[f32], delta: &[f32], out: &mut [f32]) -> Result<(), ParamError> {
        if x.len() < Self::GLOBAL_SIZE {
            return Err(ParamError::WrongGlobalSize {
                expected: Self::GLOBAL_SIZE,
                got: x.len(),
            });
        }
        if delta.len() < Self::LOCAL_SIZE {
            return Err(ParamError::WrongLocalSize {
                expected: Self::LOCAL_SIZE,
                got: delta.len(),
            });
        }
        if out.len() < Self::GLOBAL_SIZE {
            return Err(ParamError::WrongOutSize {
                expected: Self::GLOBAL_SIZE,
                got: out.len(),
            });
        }

        let se3 = SE3F32::from_params(x);
        let se3_plus = se3.retract(delta);
        out[..Self::GLOBAL_SIZE].copy_from_slice(&se3_plus.to_params());
        Ok(())
    }
}

impl std::ops::Mul<SE3F32> for SE3F32 {
    type Output = SE3F32;

    fn mul(self, rhs: SE3F32) -> Self::Output {
        let r = self.r * rhs.r;
        let t = self.t + self.r * rhs.t;
        Self { r, t }
    }
}

impl std::ops::MulAssign<SE3F32> for SE3F32 {
    #[inline]
    fn mul_assign(&mut self, rhs: SE3F32) {
        *self = *self * rhs;
    }
}

impl std::ops::Mul<Vec3AF32> for SE3F32 {
    type Output = Vec3AF32;

    fn mul(self, rhs: Vec3AF32) -> Self::Output {
        self.r * rhs + self.t
    }
}

#[cfg(feature = "approx")]
impl approx::AbsDiffEq for SE3F32 {
    type Epsilon = <SO3F32 as approx::AbsDiffEq>::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        <SO3F32 as approx::AbsDiffEq>::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.r.abs_diff_eq(&other.r, epsilon) && self.t.abs_diff_eq(&other.t, epsilon)
    }
}

#[cfg(feature = "approx")]
impl approx::RelativeEq for SE3F32 {
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        <SO3F32 as approx::RelativeEq>::default_max_relative()
    }

    #[inline]
    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.r.relative_eq(&other.r, epsilon, max_relative)
            && self.t.relative_eq(&other.t, epsilon, max_relative)
    }
}

/// Double-precision rigid-body transform with the [`SE3F32`] conventions:
/// `[υ; ω]` tangent order, `exp`/`log`, adjoint and Jacobians, over
/// [`SO3F64`] and [`Vec3F64`].
///
/// Like [`SO3F64`], it does not implement [`Param`], which is `f32`-only.
/// `PartialEq` compares the quaternions exactly, so `q` and `-q` differ.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SE3F64 {
    pub r: SO3F64,
    pub t: Vec3F64,
}

impl SE3F64 {
    pub const IDENTITY: Self = Self {
        r: SO3F64::IDENTITY,
        t: Vec3F64::ZERO,
    };

    pub fn new(rotation: SO3F64, translation: Vec3F64) -> Self {
        Self {
            r: rotation,
            t: translation,
        }
    }

    /// From `[qx, qy, qz, qw, tx, ty, tz]`.
    pub fn from_array(arr: [f64; 7]) -> Self {
        Self {
            r: SO3F64::from_array([arr[0], arr[1], arr[2], arr[3]]),
            t: Vec3F64::from_array([arr[4], arr[5], arr[6]]),
        }
    }

    /// To `[qx, qy, qz, qw, tx, ty, tz]`.
    pub fn to_array(&self) -> [f64; 7] {
        let r = self.r.to_array();
        let t = self.t.to_array();
        [r[0], r[1], r[2], r[3], t[0], t[1], t[2]]
    }

    pub fn from_matrix(mat: &Mat4F64) -> Self {
        Self {
            r: SO3F64::from_matrix4(mat),
            t: Vec3F64::new(mat.w_axis.x, mat.w_axis.y, mat.w_axis.z),
        }
    }

    /// Uniformly random rotation and a translation in `[0, 1)³`.
    pub fn from_random() -> Self {
        let mut rng = rand::rng();
        Self {
            r: SO3F64::from_random(),
            t: Vec3F64::new(rng.random(), rng.random(), rng.random()),
        }
    }

    pub fn from_qxyz(quat: QuatF64, xyz: Vec3F64) -> Self {
        Self {
            r: SO3F64::from_quaternion(quat),
            t: xyz,
        }
    }

    #[inline]
    pub fn rplus(&self, upsilon: Vec3F64, omega: Vec3F64) -> Self {
        *self * SE3F64::exp(upsilon, omega)
    }

    #[inline]
    pub fn rminus(&self, other: &Self) -> (Vec3F64, Vec3F64) {
        (self.inverse() * *other).log()
    }

    #[inline]
    pub fn lplus(upsilon: Vec3F64, omega: Vec3F64, x: &Self) -> Self {
        SE3F64::exp(upsilon, omega) * *x
    }

    #[inline]
    pub fn lminus(y: &Self, x: &Self) -> (Vec3F64, Vec3F64) {
        (*y * x.inverse()).log()
    }

    pub fn inverse(&self) -> Self {
        let r_inv = self.r.inverse();
        Self {
            r: r_inv,
            t: r_inv * (-self.t),
        }
    }

    pub fn matrix(&self) -> Mat4F64 {
        let r = self.r.matrix().to_cols_array();
        Mat4F64::from_cols_array(&[
            r[0], r[1], r[2], 0.0, // col0
            r[3], r[4], r[5], 0.0, // col1
            r[6], r[7], r[8], 0.0, // col2
            self.t.x, self.t.y, self.t.z, 1.0, // col3
        ])
    }

    /// Adjoint for the `[υ; ω]` tangent order of [`Self::exp`]:
    /// `Ad(T) = [[R, t×R], [0, R]]`, so that `T · Exp(ξ) · T⁻¹ = Exp(Ad(T) · ξ)`.
    pub fn adjoint(&self) -> [[f64; 6]; 6] {
        let r = self.r.matrix();
        block_upper_triangular(&r, &(SO3F64::hat(self.t) * r))
    }

    /// `Exp(υ, ω)`, with translation `V(ω) · υ` and `V` the SO(3) left Jacobian.
    pub fn exp(upsilon: Vec3F64, omega: Vec3F64) -> Self {
        Self {
            r: SO3F64::exp(omega),
            t: SO3F64::left_jacobian(omega) * upsilon,
        }
    }

    /// Returns `(υ, ω)` with `Exp(υ, ω) = self`.
    pub fn log(&self) -> (Vec3F64, Vec3F64) {
        let omega = self.r.log();
        let theta2 = omega.dot(omega);
        // V⁻¹ = I − ½ω× + c·ω×², c = (1 − (θ/2)·cot(θ/2)) / θ².
        let c = if theta2 < SERIES_THETA2_F64 {
            1.0 / 12.0 + theta2 / 720.0
        } else {
            let half = 0.5 * theta2.sqrt();
            (1.0 - half * half.cos() / half.sin()) / theta2
        };
        let w = SO3F64::hat(omega);
        let v_inv = Mat3F64::IDENTITY - 0.5 * w + c * (w * w);
        (v_inv * self.t, omega)
    }

    pub fn hat(upsilon: Vec3F64, omega: Vec3F64) -> Mat4F64 {
        let h = SO3F64::hat(omega).to_cols_array();
        Mat4F64::from_cols_array(&[
            h[0], h[1], h[2], 0.0, h[3], h[4], h[5], 0.0, h[6], h[7], h[8], 0.0, upsilon.x,
            upsilon.y, upsilon.z, 0.0,
        ])
    }

    pub fn vee(omega: Mat4F64) -> (Vec3F64, Vec3F64) {
        (
            Vec3F64::new(omega.w_axis.x, omega.w_axis.y, omega.w_axis.z),
            SO3F64::vee4(omega),
        )
    }

    /// `Jₗ(ρ, ω) = [[Jₗ(ω), Q(ρ, ω)], [0, Jₗ(ω)]]`, with `Jₗ(ω)` the SO(3) left
    /// Jacobian and `Q` as in [`SE3F32::left_jacobian`].
    pub fn left_jacobian(rho: Vec3F64, omega: Vec3F64) -> [[f64; 6]; 6] {
        block_upper_triangular(&SO3F64::left_jacobian(omega), &q_block(rho, omega))
    }

    /// `Jᵣ(ρ, ω) = Jₗ(−ρ, −ω)`.
    pub fn right_jacobian(rho: Vec3F64, omega: Vec3F64) -> [[f64; 6]; 6] {
        Self::left_jacobian(-rho, -omega)
    }

    /// `Jₗ⁻¹(ρ, ω) = [[Jₗ⁻¹, −Jₗ⁻¹ Q Jₗ⁻¹], [0, Jₗ⁻¹]]` (Barfoot, State
    /// Estimation for Robotics, eq. 7.95).
    pub fn left_jacobian_inverse(rho: Vec3F64, omega: Vec3F64) -> [[f64; 6]; 6] {
        let j_inv = SO3F64::left_jacobian(omega).inverse();
        let upper = j_inv * q_block(rho, omega) * j_inv;
        block_upper_triangular(&j_inv, &(-1.0 * upper))
    }

    /// `Jᵣ⁻¹(ρ, ω) = Jₗ⁻¹(−ρ, −ω)`.
    pub fn right_jacobian_inverse(rho: Vec3F64, omega: Vec3F64) -> [[f64; 6]; 6] {
        Self::left_jacobian_inverse(-rho, -omega)
    }
}

/// The `Q(ρ, ω)` block of the SE(3) left Jacobian (Barfoot, State Estimation
/// for Robotics, eq. 7.86).
fn q_block(rho: Vec3F64, omega: Vec3F64) -> Mat3F64 {
    let theta2 = omega.dot(omega);
    let (c1, c2, c3) = if theta2 < SERIES_THETA2_F64 {
        (
            1.0 / 6.0 - theta2 / 120.0,
            1.0 / 24.0 - theta2 / 720.0,
            1.0 / 120.0 - theta2 / 2520.0,
        )
    } else {
        let theta = theta2.sqrt();
        let (s, c) = theta.sin_cos();
        (
            (theta - s) / (theta2 * theta),
            (theta2 + 2.0 * c - 2.0) / (2.0 * theta2 * theta2),
            (2.0 * theta - 3.0 * s + theta * c) / (2.0 * theta2 * theta2 * theta),
        )
    };
    let p = SO3F64::hat(rho);
    let w = SO3F64::hat(omega);
    let wp = w * p;
    let pw = p * w;
    let wpw = wp * w;
    0.5 * p + c1 * (wp + pw + wpw) + c2 * (w * wp + pw * w - 3.0 * wpw) + c3 * (wpw * w + w * wpw)
}

/// `[[a, b], [0, a]]` as a row-major 6×6 array.
fn block_upper_triangular(a: &Mat3F64, b: &Mat3F64) -> [[f64; 6]; 6] {
    let (a, b) = (a.to_cols_array(), b.to_cols_array());
    let mut m = [[0.0; 6]; 6];
    for c in 0..3 {
        for r in 0..3 {
            m[r][c] = a[c * 3 + r];
            m[r][c + 3] = b[c * 3 + r];
            m[r + 3][c + 3] = a[c * 3 + r];
        }
    }
    m
}

impl std::ops::Mul<SE3F64> for SE3F64 {
    type Output = SE3F64;

    fn mul(self, rhs: SE3F64) -> Self::Output {
        Self {
            r: self.r * rhs.r,
            t: self.t + self.r * rhs.t,
        }
    }
}

impl std::ops::MulAssign<SE3F64> for SE3F64 {
    #[inline]
    fn mul_assign(&mut self, rhs: SE3F64) {
        *self = *self * rhs;
    }
}

impl std::ops::Mul<Vec3F64> for SE3F64 {
    type Output = Vec3F64;

    fn mul(self, rhs: Vec3F64) -> Self::Output {
        self.r * rhs + self.t
    }
}

#[cfg(feature = "approx")]
impl approx::AbsDiffEq for SE3F64 {
    type Epsilon = <SO3F64 as approx::AbsDiffEq>::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        <SO3F64 as approx::AbsDiffEq>::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.r.abs_diff_eq(&other.r, epsilon) && self.t.abs_diff_eq(&other.t, epsilon)
    }
}

#[cfg(feature = "approx")]
impl approx::RelativeEq for SE3F64 {
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        <SO3F64 as approx::RelativeEq>::default_max_relative()
    }

    #[inline]
    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.r.relative_eq(&other.r, epsilon, max_relative)
            && self.t.relative_eq(&other.t, epsilon, max_relative)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    const EPSILON: f32 = 1e-5;

    fn make_random_se3() -> SE3F32 {
        SE3F32::from_random()
    }

    fn make_random_vec3() -> Vec3AF32 {
        let mut rng = rand::rng();
        Vec3AF32::new(rng.random(), rng.random(), rng.random())
    }

    #[test]
    fn test_identity() {
        let se3 = SE3F32::IDENTITY;
        assert_relative_eq!(se3.r.q.x, 0.0);
        assert_relative_eq!(se3.r.q.y, 0.0);
        assert_relative_eq!(se3.r.q.z, 0.0);
        assert_relative_eq!(se3.r.q.w, 1.0);
        assert_relative_eq!(se3.t.x, 0.0);
        assert_relative_eq!(se3.t.y, 0.0);
        assert_relative_eq!(se3.t.z, 0.0);
    }

    #[test]
    fn test_new() {
        let rotation = SO3F32::from_quaternion(QuatF32::IDENTITY);
        let translation = Vec3AF32::new(1.0, 2.0, 3.0);
        let se3 = SE3F32::new(rotation, translation);
        assert_relative_eq!(se3.t.x, translation.x);
        assert_relative_eq!(se3.t.y, translation.y);
        assert_relative_eq!(se3.t.z, translation.z);
        assert_relative_eq!(se3.r.q.x, rotation.q.x);
        assert_relative_eq!(se3.r.q.y, rotation.q.y);
        assert_relative_eq!(se3.r.q.z, rotation.q.z);
        assert_relative_eq!(se3.r.q.w, rotation.q.w);
    }

    #[test]
    fn test_from_matrix() {
        // Test with identity matrix
        let mat = Mat4F32::IDENTITY;
        let se3 = SE3F32::from_matrix(&mat);
        assert_relative_eq!(se3.t.x, 0.0);
        assert_relative_eq!(se3.t.y, 0.0);
        assert_relative_eq!(se3.t.z, 0.0);

        let expected_rotation = SO3F32::IDENTITY.matrix();
        let actual_rotation = se3.r.matrix();
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(actual_rotation.col(i)[j], expected_rotation.col(i)[j],);
            }
        }

        // Test with a specific transformation matrix
        let mat = Mat4F32::from_cols_array(&[
            1.0, 0.0, 0.0, 0.0, // col0
            0.0, 1.0, 0.0, 0.0, // col1
            0.0, 0.0, 1.0, 0.0, // col2
            1.0, 2.0, 3.0, 1.0, // col3
        ]);
        let se3 = SE3F32::from_matrix(&mat);
        assert_relative_eq!(se3.t.x, 1.0);
        assert_relative_eq!(se3.t.y, 2.0);
        assert_relative_eq!(se3.t.z, 3.0);
    }

    #[test]
    fn test_from_qxyz() {
        let quat = QuatF32::from_xyzw(0.1, 0.2, 0.3, 0.9).normalize();
        let xyz = Vec3AF32::new(1.0, 2.0, 3.0);
        let se3 = SE3F32::from_qxyz(quat, xyz);

        assert_relative_eq!(se3.r.q.x, quat.x);
        assert_relative_eq!(se3.t.x, xyz.x);
        assert_relative_eq!(se3.t.y, xyz.y);
        assert_relative_eq!(se3.t.z, xyz.z);
    }

    #[test]
    fn test_se3_rplus_rminus_roundtrip() {
        let x = make_random_se3();
        let tau_v = Vec3AF32::new(0.3, -0.4, 0.2); // translational part
        let tau_w = Vec3AF32::new(0.1, 0.5, -0.2); // rotational part

        let y = x.rplus(tau_v, tau_w); // X ⊕ τ  →  Y
        let (dv, dw) = x.rminus(&y);

        const L_EPS: f32 = 1e-5;
        assert_relative_eq!(dv.x, tau_v.x, epsilon = L_EPS);
        assert_relative_eq!(dv.y, tau_v.y, epsilon = L_EPS);
        assert_relative_eq!(dv.z, tau_v.z, epsilon = L_EPS);
        assert_relative_eq!(dw.x, tau_w.x, epsilon = L_EPS);
        assert_relative_eq!(dw.y, tau_w.y, epsilon = L_EPS);
        assert_relative_eq!(dw.z, tau_w.z, epsilon = L_EPS);
    }

    #[test]
    fn test_se3_lplus_lminus_consistency() {
        let x = make_random_se3();
        let tau_v = Vec3AF32::new(-1.0, 0.7, 0.3);
        let tau_w = Vec3AF32::new(0.2, 0.1, -0.6);

        let y = SE3F32::lplus(tau_v, tau_w, &x); // τ ⊕ X → Y
        let (dv, dw) = SE3F32::lminus(&y, &x); // Y ⊖ X → τ
        assert_relative_eq!(dv.x, tau_v.x, epsilon = EPSILON);
        assert_relative_eq!(dv.y, tau_v.y, epsilon = EPSILON);
        assert_relative_eq!(dv.z, tau_v.z, epsilon = EPSILON);
        assert_relative_eq!(dw.x, tau_w.x, epsilon = EPSILON);
        assert_relative_eq!(dw.y, tau_w.y, epsilon = EPSILON);
        assert_relative_eq!(dw.z, tau_w.z, epsilon = EPSILON);
    }

    #[test]
    fn test_inverse() {
        // Test with identity
        let se3 = SE3F32::IDENTITY;
        let inv = se3.inverse();
        assert_relative_eq!(inv.r.q.x, se3.r.q.x);
        assert_relative_eq!(inv.r.q.y, se3.r.q.y);
        assert_relative_eq!(inv.r.q.z, se3.r.q.z);
        assert_relative_eq!(inv.r.q.w, se3.r.q.w);
        assert_relative_eq!(inv.t.x, se3.t.x);
        assert_relative_eq!(inv.t.y, se3.t.y);
        assert_relative_eq!(inv.t.z, se3.t.z);

        // Test with translation only
        let se3 = SE3F32::new(SO3F32::IDENTITY, Vec3AF32::new(1.0, 2.0, 3.0));
        let inv = se3.inverse();
        assert_relative_eq!(inv.t.x, -1.0);
        assert_relative_eq!(inv.t.y, -2.0);
        assert_relative_eq!(inv.t.z, -3.0);

        // Test inverse property: se3 * se3.inverse() = identity
        let se3 = make_random_se3();
        let inv = se3.inverse();
        let result = se3 * inv;
        assert_relative_eq!(result.r.q.x, SE3F32::IDENTITY.r.q.x, epsilon = EPSILON);
        assert_relative_eq!(result.r.q.y, SE3F32::IDENTITY.r.q.y, epsilon = EPSILON);
        assert_relative_eq!(result.r.q.z, SE3F32::IDENTITY.r.q.z, epsilon = EPSILON);
        assert_relative_eq!(result.r.q.w, SE3F32::IDENTITY.r.q.w, epsilon = EPSILON);
        assert_relative_eq!(result.t.x, SE3F32::IDENTITY.t.x, epsilon = EPSILON);
        assert_relative_eq!(result.t.y, SE3F32::IDENTITY.t.y, epsilon = EPSILON);
        assert_relative_eq!(result.t.z, SE3F32::IDENTITY.t.z, epsilon = EPSILON);
    }

    #[test]
    fn test_matrix() {
        // Test identity
        let se3 = SE3F32::IDENTITY;
        let mat = se3.matrix();
        let expected = Mat4F32::IDENTITY;
        for i in 0..4 {
            for j in 0..4 {
                assert_relative_eq!(mat.col(i)[j], expected.col(i)[j]);
            }
        }

        // Test with translation
        let se3 = SE3F32::new(SO3F32::IDENTITY, Vec3AF32::new(1.0, 2.0, 3.0));
        let mat = se3.matrix();
        let expected = Mat4F32::from_cols_array(&[
            1.0, 0.0, 0.0, 0.0, // col0
            0.0, 1.0, 0.0, 0.0, // col1
            0.0, 0.0, 1.0, 0.0, // col2
            1.0, 2.0, 3.0, 1.0, // col3
        ]);
        for i in 0..4 {
            for j in 0..4 {
                assert_relative_eq!(mat.col(i)[j], expected.col(i)[j]);
            }
        }
    }

    #[test]
    fn test_multiplication_identity() {
        let se3 = make_random_se3();
        let identity = SE3F32::IDENTITY;

        // Identity * se3 = se3
        let result1 = identity * se3;
        assert_relative_eq!(result1.r.q.x, se3.r.q.x);
        assert_relative_eq!(result1.r.q.y, se3.r.q.y);
        assert_relative_eq!(result1.r.q.z, se3.r.q.z);
        assert_relative_eq!(result1.r.q.w, se3.r.q.w);
        assert_relative_eq!(result1.t.x, se3.t.x);
        assert_relative_eq!(result1.t.y, se3.t.y);
        assert_relative_eq!(result1.t.z, se3.t.z);

        // se3 * identity = se3
        let result2 = se3 * identity;
        assert_relative_eq!(result2.r.q.x, se3.r.q.x);
        assert_relative_eq!(result2.r.q.y, se3.r.q.y);
        assert_relative_eq!(result2.r.q.z, se3.r.q.z);
        assert_relative_eq!(result2.r.q.w, se3.r.q.w);
        assert_relative_eq!(result2.t.x, se3.t.x);
        assert_relative_eq!(result2.t.y, se3.t.y);
        assert_relative_eq!(result2.t.z, se3.t.z);
    }

    #[test]
    fn test_mul_assign() {
        let mut s = SE3F32::IDENTITY;
        let s2 = make_random_se3();
        s *= s2;
        assert_relative_eq!(s.r.q.x, s2.r.q.x, epsilon = EPSILON);
        assert_relative_eq!(s.r.q.y, s2.r.q.y, epsilon = EPSILON);
        assert_relative_eq!(s.r.q.z, s2.r.q.z, epsilon = EPSILON);
        assert_relative_eq!(s.r.q.w, s2.r.q.w, epsilon = EPSILON);
        assert_relative_eq!(s.t.x, s2.t.x, epsilon = EPSILON);
        assert_relative_eq!(s.t.y, s2.t.y, epsilon = EPSILON);
        assert_relative_eq!(s.t.z, s2.t.z, epsilon = EPSILON);

        let mut s3 = make_random_se3();
        let original_s3 = s3;
        let s4 = make_random_se3();
        s3 *= s4;
        let expected = original_s3 * s4;
        assert_relative_eq!(s3.r.q.x, expected.r.q.x, epsilon = EPSILON);
        assert_relative_eq!(s3.r.q.y, expected.r.q.y, epsilon = EPSILON);
        assert_relative_eq!(s3.r.q.z, expected.r.q.z, epsilon = EPSILON);
        assert_relative_eq!(s3.r.q.w, expected.r.q.w, epsilon = EPSILON);
        assert_relative_eq!(s3.t.x, expected.t.x, epsilon = EPSILON);
        assert_relative_eq!(s3.t.y, expected.t.y, epsilon = EPSILON);
        assert_relative_eq!(s3.t.z, expected.t.z, epsilon = EPSILON);
    }

    #[test]
    fn test_multiplication_point() {
        let se3_1 = make_random_se3();
        let se3_2 = make_random_se3();
        let point = make_random_vec3();

        // Test transformation consistency
        let relative_transform = se3_1.inverse() * se3_2;
        let point_in_1 = se3_1.inverse() * point;
        let point_in_2 = se3_2.inverse() * point;
        let point_1_to_2 = relative_transform.inverse() * point_in_1;
        let point_2_to_1 = relative_transform * point_in_2;

        assert_relative_eq!(point_in_1.x, point_2_to_1.x, epsilon = EPSILON);
        assert_relative_eq!(point_in_1.y, point_2_to_1.y, epsilon = EPSILON);
        assert_relative_eq!(point_in_1.z, point_2_to_1.z, epsilon = EPSILON);
        assert_relative_eq!(point_in_2.x, point_1_to_2.x, epsilon = EPSILON);
        assert_relative_eq!(point_in_2.y, point_1_to_2.y, epsilon = EPSILON);
        assert_relative_eq!(point_in_2.z, point_1_to_2.z, epsilon = EPSILON);
    }

    #[test]
    fn test_exp_identity() {
        // exp(0) = identity
        let upsilon = Vec3AF32::ZERO;
        let omega = Vec3AF32::ZERO;
        let se3 = SE3F32::exp(upsilon, omega);

        assert_relative_eq!(se3.r.q.x, SE3F32::IDENTITY.r.q.x);
        assert_relative_eq!(se3.r.q.y, SE3F32::IDENTITY.r.q.y);
        assert_relative_eq!(se3.r.q.z, SE3F32::IDENTITY.r.q.z);
        assert_relative_eq!(se3.r.q.w, SE3F32::IDENTITY.r.q.w);
        assert_relative_eq!(se3.t.x, 0.0);
        assert_relative_eq!(se3.t.y, 0.0);
        assert_relative_eq!(se3.t.z, 0.0);
    }

    #[test]
    fn test_exp_translation_only() {
        // Pure translation (omega = 0)
        let upsilon = Vec3AF32::new(1.0, 2.0, 3.0);
        let omega = Vec3AF32::ZERO;
        let se3 = SE3F32::exp(upsilon, omega);

        assert_relative_eq!(se3.r.q.x, SE3F32::IDENTITY.r.q.x);
        assert_relative_eq!(se3.r.q.y, SE3F32::IDENTITY.r.q.y);
        assert_relative_eq!(se3.r.q.z, SE3F32::IDENTITY.r.q.z);
        assert_relative_eq!(se3.r.q.w, SE3F32::IDENTITY.r.q.w);
        assert_relative_eq!(se3.t.x, upsilon.x);
        assert_relative_eq!(se3.t.y, upsilon.y);
        assert_relative_eq!(se3.t.z, upsilon.z);
    }

    #[test]
    fn test_log_identity() {
        let se3 = SE3F32::IDENTITY;
        let (upsilon, omega) = se3.log();

        assert_relative_eq!(upsilon.x, 0.0);
        assert_relative_eq!(upsilon.y, 0.0);
        assert_relative_eq!(upsilon.z, 0.0);
        assert_relative_eq!(omega.x, 0.0);
        assert_relative_eq!(omega.y, 0.0);
        assert_relative_eq!(omega.z, 0.0);
    }

    #[test]
    fn test_log_translation_only() {
        let translation = Vec3AF32::new(1.0, 2.0, 3.0);
        let se3 = SE3F32::new(SO3F32::IDENTITY, translation);
        let (upsilon, omega) = se3.log();

        assert_relative_eq!(upsilon.x, translation.x);
        assert_relative_eq!(upsilon.y, translation.y);
        assert_relative_eq!(upsilon.z, translation.z);
        assert_relative_eq!(omega.x, 0.0);
        assert_relative_eq!(omega.y, 0.0);
        assert_relative_eq!(omega.z, 0.0);
    }

    #[test]
    fn test_exp_log_roundtrip() {
        let upsilon = Vec3AF32::new(0.5, -0.5, 1.0);
        let omega = Vec3AF32::new(0.1, 0.2, -0.3);

        let se3 = SE3F32::exp(upsilon, omega);
        let (log_upsilon, log_omega) = se3.log();

        assert_relative_eq!(log_upsilon.x, upsilon.x);
        assert_relative_eq!(log_upsilon.y, upsilon.y);
        assert_relative_eq!(log_upsilon.z, upsilon.z);
        assert_relative_eq!(log_omega.x, omega.x);
        assert_relative_eq!(log_omega.y, omega.y);
        assert_relative_eq!(log_omega.z, omega.z);
    }

    #[test]
    fn test_hat_vee_roundtrip() {
        let upsilon = Vec3AF32::new(1.0, 2.0, 3.0);
        let omega = Vec3AF32::new(0.1, -0.2, 0.3);

        let hat_matrix = SE3F32::hat(upsilon, omega);
        let (vee_upsilon, vee_omega) = SE3F32::vee(hat_matrix);

        assert_relative_eq!(vee_upsilon.x, upsilon.x);
        assert_relative_eq!(vee_upsilon.y, upsilon.y);
        assert_relative_eq!(vee_upsilon.z, upsilon.z);
        assert_relative_eq!(vee_omega.x, omega.x);
        assert_relative_eq!(vee_omega.y, omega.y);
    }

    #[test]
    fn test_hat_structure() {
        let upsilon = Vec3AF32::new(1.0, 2.0, 3.0);
        let omega = Vec3AF32::new(0.1, -0.2, 0.3);
        let hat_matrix = SE3F32::hat(upsilon, omega);

        // Check that the bottom row is [0, 0, 0, 0]
        assert_relative_eq!(hat_matrix.x_axis.w, 0.0);
        assert_relative_eq!(hat_matrix.y_axis.w, 0.0);
        assert_relative_eq!(hat_matrix.z_axis.w, 0.0);
        assert_relative_eq!(hat_matrix.w_axis.w, 0.0);

        // Check that the translation part is correct
        assert_relative_eq!(hat_matrix.w_axis.x, upsilon.x);
        assert_relative_eq!(hat_matrix.w_axis.y, upsilon.y);
        assert_relative_eq!(hat_matrix.w_axis.z, upsilon.z);
    }

    #[test]
    fn test_adjoint_identity() {
        let se3 = SE3F32::IDENTITY;
        let adj = se3.adjoint();

        // Expected 6x6 adjoint of identity: block diagonal with two 3x3 identity matrices
        let expected = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ];

        for i in 0..6 {
            for j in 0..6 {
                assert_relative_eq!(adj[i][j], expected[i][j]);
            }
        }
    }

    #[test]
    fn test_adjoint_properties() {
        let se3_1 = make_random_se3();
        let se3_2 = make_random_se3();

        // Test: (se3_1 * se3_2).adjoint() = se3_1.adjoint() * se3_2.adjoint()
        let composed = se3_1 * se3_2;
        let adj_composed = composed.adjoint();

        let adj_1 = se3_1.adjoint();
        let adj_2 = se3_2.adjoint();

        // Matrix multiplication of 6x6 matrices
        let mut adj_product = [[0.0f32; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                for (k, adj_2k) in adj_2.iter().enumerate() {
                    adj_product[i][j] += adj_1[i][k] * adj_2k[j];
                }
            }
        }

        for i in 0..6 {
            for j in 0..6 {
                assert_relative_eq!(adj_composed[i][j], adj_product[i][j], epsilon = EPSILON);
            }
        }
    }

    #[test]
    fn test_from_random() {
        let se3 = SE3F32::from_random();

        // Test that the quaternion is normalized
        let q_norm = (se3.r.q.x * se3.r.q.x
            + se3.r.q.y * se3.r.q.y
            + se3.r.q.z * se3.r.q.z
            + se3.r.q.w * se3.r.q.w)
            .sqrt();
        assert_relative_eq!(q_norm, 1.0);

        // Test that inverse property holds
        let inv = se3.inverse();
        let result = se3 * inv;
        assert_relative_eq!(result.r.q.x, SE3F32::IDENTITY.r.q.x, epsilon = EPSILON);
        assert_relative_eq!(result.r.q.y, SE3F32::IDENTITY.r.q.y, epsilon = EPSILON);
        assert_relative_eq!(result.r.q.z, SE3F32::IDENTITY.r.q.z, epsilon = EPSILON);
        assert_relative_eq!(result.r.q.w, SE3F32::IDENTITY.r.q.w, epsilon = EPSILON);
        assert_relative_eq!(result.t.x, SE3F32::IDENTITY.t.x, epsilon = EPSILON);
        assert_relative_eq!(result.t.y, SE3F32::IDENTITY.t.y, epsilon = EPSILON);
        assert_relative_eq!(result.t.z, SE3F32::IDENTITY.t.z, epsilon = EPSILON);
    }

    #[test]
    fn test_matrix_from_matrix_roundtrip() {
        let se3 = make_random_se3();
        let matrix = se3.matrix();
        let se3_reconstructed = SE3F32::from_matrix(&matrix);

        // Check rotation (quaternions might have opposite signs but represent same rotation)
        let q1 = se3.r.q;
        let q2 = se3_reconstructed.r.q;
        let same_rotation = (q1.x - q2.x).abs() < 1e-5
            && (q1.y - q2.y).abs() < 1e-5
            && (q1.z - q2.z).abs() < 1e-5
            && (q1.w - q2.w).abs() < 1e-5;
        let opposite_rotation = (q1.x + q2.x).abs() < 1e-5
            && (q1.y + q2.y).abs() < 1e-5
            && (q1.z + q2.z).abs() < 1e-5
            && (q1.w + q2.w).abs() < 1e-5;
        assert!(same_rotation || opposite_rotation);

        // Check translation
        assert_relative_eq!(se3.t.x, se3_reconstructed.t.x);
        assert_relative_eq!(se3.t.y, se3_reconstructed.t.y);
        assert_relative_eq!(se3.t.z, se3_reconstructed.t.z);
    }

    #[test]
    fn test_local_param_plus_matches_retract() {
        let se3 = make_random_se3();
        let params = se3.to_params();

        let delta = Vec3AF32::new(0.01, -0.02, 0.03);
        let omega = Vec3AF32::new(0.001, 0.002, -0.003);
        let tangent = [delta.x, delta.y, delta.z, omega.x, omega.y, omega.z];

        let se3_retract = se3.retract(&tangent);
        let expected = se3_retract.to_params();

        let mut out = [0.0_f32; SE3F32::GLOBAL_SIZE];
        SE3F32::plus(&params, &tangent, &mut out).unwrap();

        for i in 0..SE3F32::GLOBAL_SIZE {
            assert_relative_eq!(out[i], expected[i], epsilon = EPSILON);
        }
    }

    #[test]
    fn test_composition_associativity() {
        let se3_1 = make_random_se3();
        let se3_2 = make_random_se3();
        let se3_3 = make_random_se3();

        // Test (se3_1 * se3_2) * se3_3 = se3_1 * (se3_2 * se3_3)
        let left_assoc = (se3_1 * se3_2) * se3_3;
        let right_assoc = se3_1 * (se3_2 * se3_3);

        assert_relative_eq!(left_assoc.r.q.x, right_assoc.r.q.x, epsilon = EPSILON);
        assert_relative_eq!(left_assoc.r.q.y, right_assoc.r.q.y, epsilon = EPSILON);
        assert_relative_eq!(left_assoc.r.q.z, right_assoc.r.q.z, epsilon = EPSILON);
        assert_relative_eq!(left_assoc.r.q.w, right_assoc.r.q.w, epsilon = EPSILON);
        assert_relative_eq!(left_assoc.t.x, right_assoc.t.x, epsilon = EPSILON);
        assert_relative_eq!(left_assoc.t.y, right_assoc.t.y, epsilon = EPSILON);
        assert_relative_eq!(left_assoc.t.z, right_assoc.t.z, epsilon = EPSILON);
    }

    /// Multiply a 6×6 matrix stored as [[f32;6];6] with a length-6 vector.
    fn mat6_mul_vec6(mat: &[[f32; 6]; 6], v: &[f32; 6]) -> [f32; 6] {
        let mut out = [0.0_f32; 6];
        for (i, row) in mat.iter().enumerate() {
            for (j, &vj) in v.iter().enumerate().take(6) {
                out[i] += row[j] * vj;
            }
        }
        out
    }

    /// True if all entries of the 6×6 matrix are finite.
    fn mat6_is_finite(mat: &[[f32; 6]; 6]) -> bool {
        mat.iter().all(|row| row.iter().all(|x| x.is_finite()))
    }

    /// Transpose of a 6×6 matrix.
    fn mat6_transpose(mat: &[[f32; 6]; 6]) -> [[f32; 6]; 6] {
        let mut m = [[0.0_f32; 6]; 6];
        for (i, row) in m.iter_mut().enumerate() {
            for (j, item) in row.iter_mut().enumerate() {
                *item = mat[j][i];
            }
        }
        m
    }

    /// Diverse translation/rotation test pairs (ρ, θ).
    fn test_pairs() -> [(Vec3AF32, Vec3AF32); 5] {
        [
            (
                Vec3AF32::new(0.1, 0.2, 0.3),
                Vec3AF32::new(0.02, 0.03, 0.04),
            ),
            (Vec3AF32::new(-0.3, 0.5, 0.1), Vec3AF32::new(0.7, 0.0, 0.0)),
            (
                Vec3AF32::new(0.0, 0.0, 0.0),
                Vec3AF32::new(0.001, -0.002, 0.003),
            ), // tiny θ
            (Vec3AF32::new(0.5, -0.4, 0.2), Vec3AF32::ZERO), // pure translation
            (
                Vec3AF32::new(0.01, 0.02, 0.03),
                Vec3AF32::new(-0.01, 0.02, -0.03),
            ), // small mixed
        ]
    }

    #[test]
    fn test_left_jacobian_se3() {
        for (rho, omega) in test_pairs() {
            let jl = SE3F32::left_jacobian(rho, omega);

            // All entries must be finite
            assert!(mat6_is_finite(&jl));

            // Identity property:  Jₗ * τ ≈ τ   where τ = [ρ;θ]
            let tau = [rho.x, rho.y, rho.z, omega.x, omega.y, omega.z];
            let result = mat6_mul_vec6(&jl, &tau);
            for k in 0..6 {
                assert_relative_eq!(result[k], tau[k], epsilon = 1e-4);
            }
        }
    }

    #[test]
    fn test_right_jacobian_se3() {
        for (rho, omega) in test_pairs() {
            let jr = SE3F32::right_jacobian(rho, omega);

            // All entries must be finite
            assert!(mat6_is_finite(&jr));

            // Identity property:  Jᵣ * τ ≈ τ
            let tau = [rho.x, rho.y, rho.z, omega.x, omega.y, omega.z];
            let result = mat6_mul_vec6(&jr, &tau);
            for k in 0..6 {
                assert_relative_eq!(result[k], tau[k], epsilon = 1e-4);
            }
        }
    }

    #[test]
    fn test_left_right_relationship() {
        for (rho, omega) in test_pairs() {
            let jl_minus = SE3F32::left_jacobian(-rho, -omega);
            let jr = SE3F32::right_jacobian(rho, omega);

            for i in 0..6 {
                for j in 0..6 {
                    assert_relative_eq!(jr[i][j], jl_minus[i][j], epsilon = 1e-6);
                }
            }
        }
    }

    // Approximate symmetry  (small-angle heuristic)  Jₗ ≈ Jᵣᵀ
    #[test]
    fn test_left_right_transpose_small_angles() {
        // Use only the tiny-angle pair (index 2 in `test_pairs`)
        let (rho, omega) = test_pairs()[2];
        let jl = SE3F32::left_jacobian(rho, omega);
        let jr_t = mat6_transpose(&SE3F32::right_jacobian(rho, omega));

        for i in 0..6 {
            for j in 0..6 {
                assert_relative_eq!(jl[i][j], jr_t[i][j], epsilon = 1e-3);
            }
        }
    }

    /// `T · Exp(ξ) · T⁻¹ = Exp(Ad(T) · ξ)` for the `[υ; ω]` order of `exp`.
    #[test]
    fn test_adjoint_conjugates_exp() {
        let t = SE3F32::exp(Vec3AF32::new(0.4, -1.2, 0.7), Vec3AF32::new(0.3, -0.5, 0.9));
        let (rho, omega) = (
            Vec3AF32::new(0.2, 0.1, -0.3),
            Vec3AF32::new(-0.1, 0.25, 0.05),
        );
        let xi = [rho.x, rho.y, rho.z, omega.x, omega.y, omega.z];
        let m = mat6_mul_vec6(&t.adjoint(), &xi);
        let mapped = SE3F32::exp(
            Vec3AF32::new(m[0], m[1], m[2]),
            Vec3AF32::new(m[3], m[4], m[5]),
        );
        let conjugated = t * SE3F32::exp(rho, omega) * t.inverse();
        let (a, b) = (
            mapped.matrix().to_cols_array(),
            conjugated.matrix().to_cols_array(),
        );
        for k in 0..16 {
            assert_relative_eq!(a[k], b[k], epsilon = 1e-5);
        }
    }

    /// Twists for the Jacobian checks: large and medium rotations, one on each
    /// side of the `Q` series threshold, a tiny one and a pure translation.
    fn jacobian_test_twists() -> [(Vec3AF32, Vec3AF32); 6] {
        [
            (Vec3AF32::new(0.4, -0.3, 0.8), Vec3AF32::new(0.9, -0.6, 0.5)),
            (Vec3AF32::new(-0.3, 0.5, 0.1), Vec3AF32::new(0.7, 0.0, 0.0)),
            (
                Vec3AF32::new(0.6, 0.2, -0.4),
                Vec3AF32::new(0.08, -0.05, 0.03),
            ),
            (
                Vec3AF32::new(0.6, 0.2, -0.4),
                Vec3AF32::new(0.06, -0.08, 0.02),
            ),
            (
                Vec3AF32::new(0.5, -0.4, 0.2),
                Vec3AF32::new(1e-3, -2e-3, 1.5e-3),
            ),
            (Vec3AF32::new(0.5, -0.4, 0.2), Vec3AF32::ZERO),
        ]
    }

    /// Column `k` of `∂ Log(f(Exp(ξ + δ))) / ∂δ` at `δ = 0`, by central differences.
    fn central_difference(
        rho: Vec3AF32,
        omega: Vec3AF32,
        k: usize,
        f: impl Fn(SE3F32) -> SE3F32,
    ) -> [f32; 6] {
        const H: f32 = 1e-2;
        let at = |h: f32| {
            let mut xi = [rho.x, rho.y, rho.z, omega.x, omega.y, omega.z];
            xi[k] += h;
            let (u, w) = f(SE3F32::exp(
                Vec3AF32::new(xi[0], xi[1], xi[2]),
                Vec3AF32::new(xi[3], xi[4], xi[5]),
            ))
            .log();
            [u.x, u.y, u.z, w.x, w.y, w.z]
        };
        let (up, down) = (at(H), at(-H));
        std::array::from_fn(|i| (up[i] - down[i]) / (2.0 * H))
    }

    /// `Exp(ξ + δ) ≈ Exp(Jₗ(ξ) · δ) · Exp(ξ)`.
    #[test]
    fn test_left_jacobian_matches_central_differences() {
        for (rho, omega) in jacobian_test_twists() {
            let jl = SE3F32::left_jacobian(rho, omega);
            let x_inv = SE3F32::exp(rho, omega).inverse();
            let columns: [[f32; 6]; 6] =
                std::array::from_fn(|k| central_difference(rho, omega, k, |y| y * x_inv));
            for (i, row) in jl.iter().enumerate() {
                for (k, &analytic) in row.iter().enumerate() {
                    let numeric = columns[k][i];
                    assert!(
                        (analytic - numeric).abs() < 2e-3,
                        "ξ = ({rho:?}, {omega:?}), J[{i}][{k}] = {analytic} vs {numeric}"
                    );
                }
            }
        }
    }

    /// `Exp(ξ + δ) ≈ Exp(ξ) · Exp(Jᵣ(ξ) · δ)`.
    #[test]
    fn test_right_jacobian_matches_central_differences() {
        for (rho, omega) in jacobian_test_twists() {
            let jr = SE3F32::right_jacobian(rho, omega);
            let x_inv = SE3F32::exp(rho, omega).inverse();
            let columns: [[f32; 6]; 6] =
                std::array::from_fn(|k| central_difference(rho, omega, k, |y| x_inv * y));
            for (i, row) in jr.iter().enumerate() {
                for (k, &analytic) in row.iter().enumerate() {
                    let numeric = columns[k][i];
                    assert!(
                        (analytic - numeric).abs() < 2e-3,
                        "ξ = ({rho:?}, {omega:?}), J[{i}][{k}] = {analytic} vs {numeric}"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests_f64 {
    use super::*;

    fn twist(xi: [f64; 6]) -> (Vec3F64, Vec3F64) {
        (
            Vec3F64::new(xi[0], xi[1], xi[2]),
            Vec3F64::new(xi[3], xi[4], xi[5]),
        )
    }

    fn flat((u, w): (Vec3F64, Vec3F64)) -> [f64; 6] {
        [u.x, u.y, u.z, w.x, w.y, w.z]
    }

    fn mat6_mul(a: &[[f64; 6]; 6], b: &[[f64; 6]; 6]) -> [[f64; 6]; 6] {
        std::array::from_fn(|i| std::array::from_fn(|j| (0..6).map(|k| a[i][k] * b[k][j]).sum()))
    }

    fn mat6_mul_vec6(a: &[[f64; 6]; 6], v: &[f64; 6]) -> [f64; 6] {
        std::array::from_fn(|i| (0..6).map(|k| a[i][k] * v[k]).sum())
    }

    fn assert_close(a: &[f64], b: &[f64], tol: f64) {
        for (k, (x, y)) in a.iter().zip(b).enumerate() {
            assert!((x - y).abs() <= tol, "entry {k}: {x} vs {y}");
        }
    }

    /// Large and medium rotations, one on each side of the series threshold,
    /// a tiny one and a pure translation.
    const TWISTS: [[f64; 6]; 6] = [
        [0.4, -0.3, 0.8, 0.9, -0.6, 0.5],
        [-1.2, 0.7, 0.3, 0.5, 2.0, -1.3],
        [0.6, 0.2, -0.4, 8e-3, -5e-3, 3e-3],
        [0.6, 0.2, -0.4, 6e-3, -9e-3, 2e-3],
        [0.5, -0.4, 0.2, 1e-6, -2e-6, 1.5e-6],
        [0.5, -0.4, 0.2, 0.0, 0.0, 0.0],
    ];

    /// `Exp(ξ)` against scipy 1.15.2 `expm` of the 4×4 twist matrix for
    /// `ξ = [0.4, -0.3, 0.8, 0.9, -0.6, 0.5]`.
    #[test]
    fn exp_matches_matrix_exponential() {
        let expected_rows = [
            [
                0.7294259701953586,
                -0.629314358515437,
                -0.26814397657016986,
                0.2967766087600957,
            ],
            [
                0.15026525656623585,
                0.5298221777165246,
                -0.8346908485593949,
                -0.5520744073079797,
            ],
            [
                0.6673515615278374,
                0.5685524585876164,
                0.48103013955503193,
                0.683312815462252,
            ],
        ];
        let (u, w) = twist(TWISTS[0]);
        let m = SE3F64::exp(u, w).matrix().to_cols_array();
        for (row, expected) in expected_rows.iter().enumerate() {
            let got: [f64; 4] = std::array::from_fn(|col| m[col * 4 + row]);
            assert_close(&got, expected, 1e-14);
        }
    }

    /// `Log` of the pose scipy 1.15.2 `expm` gives for
    /// `ξ = [-1.2, 0.7, 0.3, 0.5, 2.0, -1.3]` (a 2.44 rad rotation) returns `ξ`.
    #[test]
    fn log_inverts_matrix_exponential() {
        let rows = [
            [
                -0.6878536798787738,
                0.642042568755561,
                0.3385833058244113,
                0.15442339118762224,
            ],
            [
                -0.04877238250643101,
                0.42452791933834433,
                -0.9041002712127129,
                0.7056124035610446,
            ],
            [
                -0.7242089268863455,
                -0.6384022130350239,
                -0.2606991457794001,
                0.8295665405506929,
            ],
        ];
        let cols: [f64; 16] = std::array::from_fn(|k| {
            let (col, row) = (k / 4, k % 4);
            if row < 3 {
                rows[row][col]
            } else if col == 3 {
                1.0
            } else {
                0.0
            }
        });
        let pose = SE3F64::from_matrix(&Mat4F64::from_cols_array(&cols));
        assert_close(&flat(pose.log()), &TWISTS[1], 1e-13);
    }

    #[test]
    fn exp_log_roundtrip() {
        for xi in TWISTS {
            let (u, w) = twist(xi);
            assert_close(&flat(SE3F64::exp(u, w).log()), &xi, 1e-13);
        }
    }

    #[test]
    fn hat_vee_roundtrip() {
        let (u, w) = twist(TWISTS[0]);
        assert_close(&flat(SE3F64::vee(SE3F64::hat(u, w))), &TWISTS[0], 0.0);
    }

    #[test]
    fn inverse_composition_and_points() {
        let (a, b) = (twist(TWISTS[0]), twist(TWISTS[1]));
        let (x, y) = (SE3F64::exp(a.0, a.1), SE3F64::exp(b.0, b.1));
        let id = (x * x.inverse()).matrix().to_cols_array();
        assert_close(&id, &Mat4F64::IDENTITY.to_cols_array(), 1e-15);
        let p = Vec3F64::new(0.3, -1.1, 2.4);
        assert_close(&((x * y) * p).to_array(), &(x * (y * p)).to_array(), 1e-14);
        assert_close(&flat(x.rminus(&x.rplus(b.0, b.1))), &TWISTS[1], 1e-13);
        let round = SE3F64::from_array(x.to_array()).matrix().to_cols_array();
        assert_close(&round, &x.matrix().to_cols_array(), 0.0);
    }

    /// `T · Exp(ξ) · T⁻¹ = Exp(Ad(T) · ξ)`.
    #[test]
    fn adjoint_conjugates_exp() {
        let (a, b) = (twist(TWISTS[1]), twist(TWISTS[0]));
        let t = SE3F64::exp(a.0, a.1);
        let mapped = twist(mat6_mul_vec6(&t.adjoint(), &TWISTS[0]));
        let lhs = SE3F64::exp(mapped.0, mapped.1).matrix().to_cols_array();
        let rhs = (t * SE3F64::exp(b.0, b.1) * t.inverse())
            .matrix()
            .to_cols_array();
        assert_close(&lhs, &rhs, 1e-13);
    }

    /// Column `k` of `∂ Log(f(Exp(ξ + δ))) / ∂δ` at `δ = 0`, by central differences.
    fn central_difference(xi: [f64; 6], k: usize, f: impl Fn(SE3F64) -> SE3F64) -> [f64; 6] {
        const H: f64 = 1e-6;
        let at = |h: f64| {
            let mut x = xi;
            x[k] += h;
            let (u, w) = twist(x);
            flat(f(SE3F64::exp(u, w)).log())
        };
        let (up, down) = (at(H), at(-H));
        std::array::from_fn(|i| (up[i] - down[i]) / (2.0 * H))
    }

    fn assert_jacobian(analytic: &[[f64; 6]; 6], xi: [f64; 6], f: impl Fn(SE3F64) -> SE3F64) {
        let columns: [[f64; 6]; 6] = std::array::from_fn(|k| central_difference(xi, k, &f));
        for (i, row) in analytic.iter().enumerate() {
            for (k, &value) in row.iter().enumerate() {
                let numeric = columns[k][i];
                assert!(
                    (value - numeric).abs() < 1e-8,
                    "ξ = {xi:?}, J[{i}][{k}] = {value} vs {numeric}"
                );
            }
        }
    }

    /// `Exp(ξ + δ) ≈ Exp(Jₗ(ξ) · δ) · Exp(ξ)` and `≈ Exp(ξ) · Exp(Jᵣ(ξ) · δ)`.
    #[test]
    fn jacobians_match_central_differences() {
        for xi in TWISTS {
            let (u, w) = twist(xi);
            let x_inv = SE3F64::exp(u, w).inverse();
            assert_jacobian(&SE3F64::left_jacobian(u, w), xi, |y| y * x_inv);
            assert_jacobian(&SE3F64::right_jacobian(u, w), xi, |y| x_inv * y);
        }
    }

    #[test]
    fn jacobian_inverses_invert() {
        let identity: [[f64; 6]; 6] =
            std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 1.0 } else { 0.0 }));
        for xi in TWISTS {
            let (u, w) = twist(xi);
            for (j, j_inv) in [
                (
                    SE3F64::left_jacobian(u, w),
                    SE3F64::left_jacobian_inverse(u, w),
                ),
                (
                    SE3F64::right_jacobian(u, w),
                    SE3F64::right_jacobian_inverse(u, w),
                ),
            ] {
                let product = mat6_mul(&j, &j_inv);
                for (row, expected) in product.iter().zip(&identity) {
                    assert_close(row, expected, 1e-12);
                }
            }
        }
    }

    /// The f64 and f32 types agree to f32 precision.
    #[test]
    fn matches_se3f32() {
        for xi in TWISTS {
            let (u, w) = twist(xi);
            let jl64 = SE3F64::left_jacobian(u, w);
            let to32 = |v: Vec3F64| Vec3AF32::new(v.x as f32, v.y as f32, v.z as f32);
            let jl32 = SE3F32::left_jacobian(to32(u), to32(w));
            for (r64, r32) in jl64.iter().zip(&jl32) {
                let r32: [f64; 6] = std::array::from_fn(|k| r32[k] as f64);
                assert_close(r64, &r32, 1e-5);
            }
            let t = SE3F64::exp(u, w).matrix().to_cols_array();
            let t32 = SE3F32::exp(to32(u), to32(w)).matrix().to_cols_array();
            let t32: [f64; 16] = std::array::from_fn(|k| t32[k] as f64);
            assert_close(&t, &t32, 1e-5);
        }
    }
}
