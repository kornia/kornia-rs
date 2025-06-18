use crate::so3::SO3;
use glam::{Mat3A, Mat4, Quat, Vec3A};
use rand::Rng;

#[derive(Debug, Clone, Copy)]
pub struct SE3 {
    pub r: SO3,
    pub t: Vec3A,
}

impl SE3 {
    pub const IDENTITY: Self = Self {
        r: SO3::IDENTITY,
        t: Vec3A::new(0.0, 0.0, 0.0),
    };

    pub fn new(rotation: SO3, translation: Vec3A) -> Self {
        Self {
            r: rotation,
            t: translation,
        }
    }

    pub fn from_matrix(mat: &Mat4) -> Self {
        Self {
            r: SO3::from_matrix4(mat),
            t: Vec3A::new(mat.w_axis.x, mat.w_axis.y, mat.w_axis.z),
        }
    }

    pub fn from_random() -> Self {
        let mut rng = rand::rng();

        let r1: f32 = rng.random();
        let r2: f32 = rng.random();
        let r3: f32 = rng.random();

        Self {
            r: SO3::from_random(),
            t: Vec3A::new(r1, r2, r3),
        }
    }

    pub fn from_qxyz(quat: Quat, xyz: Vec3A) -> Self {
        Self {
            r: SO3::from_quaternion(quat),
            t: xyz,
        }
    }

    pub fn inverse(&self) -> Self {
        let r_inv = self.r.inverse();
        Self {
            r: r_inv,
            t: r_inv * (-self.t),
        }
    }

    pub fn matrix(&self) -> Mat4 {
        let r = self.r.matrix();
        Mat4::from_cols_array(&[
            r.x_axis.x, r.x_axis.y, r.x_axis.z, 0.0, // col0
            r.y_axis.x, r.y_axis.y, r.y_axis.z, 0.0, // col1
            r.z_axis.x, r.z_axis.y, r.z_axis.z, 0.0, // col2
            self.t.x, self.t.y, self.t.z, 1.0, // col3
        ])
    }

    pub fn adjoint(&self) -> [[f32; 6]; 6] {
        let r = self.r.matrix();
        let t = SO3::hat(self.t) * r;

        [
            [r.x_axis.x, r.y_axis.x, r.z_axis.x, 0.0, 0.0, 0.0],
            [r.x_axis.y, r.y_axis.y, r.z_axis.y, 0.0, 0.0, 0.0],
            [r.x_axis.z, r.y_axis.z, r.z_axis.z, 0.0, 0.0, 0.0],
            [
                t.x_axis.x, t.y_axis.x, t.z_axis.x, r.x_axis.x, r.y_axis.x, r.z_axis.x,
            ],
            [
                t.x_axis.y, t.y_axis.y, t.z_axis.y, r.x_axis.y, r.y_axis.y, r.z_axis.y,
            ],
            [
                t.x_axis.z, t.y_axis.z, t.z_axis.z, r.x_axis.z, r.y_axis.z, r.z_axis.z,
            ],
        ]
    }

    pub fn exp(upsilon: Vec3A, omega: Vec3A) -> Self {
        let theta = omega.dot(omega).sqrt();

        Self {
            r: SO3::exp(omega),
            t: if theta != 0.0 {
                let omega_hat = SO3::hat(omega);
                let omega_hat_sq = omega_hat * omega_hat;

                let mat_v = Mat3A::IDENTITY
                    + ((1.0 - theta.cos()) / (theta * theta)) * omega_hat
                    + ((theta - theta.sin()) / (theta.powi(3))) * omega_hat_sq;

                mat_v.mul_vec3a(upsilon) // TODO: a bit sus (should it be the other way around?)
            } else {
                upsilon
            },
        }
    }

    /// returns translation, rotation
    pub fn log(&self) -> (Vec3A, Vec3A) {
        let omega = self.r.log();
        let theta = omega.dot(omega).sqrt();

        (
            if theta != 0.0 {
                let omega_hat = SO3::hat(omega);
                let omega_hat_sq = omega_hat * omega_hat;

                let mat_v_inv = Mat3A::IDENTITY - 0.5 * omega_hat
                    + ((1.0 - theta * (theta / 2.0).cos() / (2.0 * (theta / 2.0).sin()))
                        / theta.powi(2))
                        * omega_hat_sq;

                mat_v_inv.mul_vec3a(self.t) // TODO:
            } else {
                self.t
            },
            omega,
        )
    }

    pub fn hat(upsilon: Vec3A, omega: Vec3A) -> Mat4 {
        let h = SO3::hat(omega);

        Mat4::from_cols_array(&[
            h.x_axis.x, h.x_axis.y, h.x_axis.z, 0.0, h.y_axis.x, h.y_axis.y, h.y_axis.z, 0.0,
            h.z_axis.x, h.z_axis.y, h.z_axis.z, 0.0, upsilon.x, upsilon.y, upsilon.z, 0.0,
        ])
    }

    pub fn vee(omega: Mat4) -> (Vec3A, Vec3A) {
        (
            Vec3A::new(omega.w_axis.x, omega.w_axis.y, omega.w_axis.z),
            SO3::vee4(omega),
        )
    }

    /// Jₗ(ρ,θ) =
    /// ⎡ Jₗ(θ)      Q(ρ,θ) ⎤
    /// ⎢                   ⎥
    /// ⎣  0         Jₗ(θ)  ⎦
    ///
    /// Jₗ(θ)is the left Jacobian of SO(3)
    ///
    ///              1            θ − sinθ
    /// Q(ρ,θ) = ───── ρ× + ───────────────── (θ×ρ× + ρ×θ× + θ×ρ×θ×)
    ///              2              θ³
    ///                       1 − θ²/2 − cosθ
    ///          − ────────────────────────── (θ²×ρ× + ρ×θ²× − 3 θ×ρ×θ×)
    ///                               θ⁴
    ///                       ⎛ 1 − θ²/2 − cosθ      θ − sinθ − θ³/6  ⎞
    ///          − ½ ⎜───────────────────────── − 3 ──────────────────⎟
    ///                       ⎝         θ⁴                    θ⁵      ⎠
    ///            · (θ×ρ×θ²× + θ²×ρ×θ×).
    /// ref: https://arxiv.org/pdf/1812.01537 (eq 180)
    pub fn left_jacobian(rho: Vec3A, omega: Vec3A) -> [[f32; 6]; 6] {
        let theta2 = omega.dot(omega);
        let theta = theta2.sqrt();

        let jl_so3 = SO3::left_jacobian(omega);

        let rho_hat = SO3::hat(rho);
        let omega_hat = SO3::hat(omega);
        let omega_hat2 = omega_hat * omega_hat;

        // Small-angle safeguard
        const EPS: f32 = 1e-6;
        let (q_mat, jl_so3) = if theta < EPS {
            // Series:  Q ≈ ½ρ× + 1/12(ω×ρ× + ρ×ω×) − 1/24 ω×ρ×ω×
            //          Jₗ_SO3 ≈ I + ½ω× + 1/6 ω×²
            let q = 0.5 * rho_hat + (1.0 / 12.0) * (omega_hat * rho_hat + rho_hat * omega_hat)
                - (1.0 / 24.0) * (omega_hat * rho_hat * omega_hat);
            let j = Mat3A::IDENTITY + 0.5 * omega_hat + (1.0 / 6.0) * omega_hat2;
            (q, j)
        } else {
            let theta3 = theta2 * theta;
            let theta4 = theta2 * theta2;

            let a = 0.5;
            let b = (theta - theta.sin()) / theta3;
            let c = (1.0 - 0.5 * theta2 - theta.cos()) / theta4;
            let d = 0.5 * (c - 3.0 * b / theta);

            let term1 = a * rho_hat;
            let term2 =
                b * (omega_hat * rho_hat + rho_hat * omega_hat + omega_hat * rho_hat * omega_hat);
            let term3 = c
                * (omega_hat2 * rho_hat + rho_hat * omega_hat2
                    - 3.0 * omega_hat * rho_hat * omega_hat);
            let term4 = d * (omega_hat * rho_hat * omega_hat2 + omega_hat2 * rho_hat * omega_hat);

            (term1 + term2 - term3 - term4, jl_so3)
        };

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
    pub fn right_jacobian(rho: Vec3A, omega: Vec3A) -> [[f32; 6]; 6] {
        Self::left_jacobian(-rho, -omega)
    }
}

impl std::ops::Mul<SE3> for SE3 {
    type Output = SE3;

    fn mul(self, rhs: SE3) -> Self::Output {
        let r = self.r * rhs.r;
        let t = self.t + self.r * rhs.t;
        Self { r, t }
    }
}

impl std::ops::Mul<Vec3A> for SE3 {
    type Output = Vec3A;

    fn mul(self, rhs: Vec3A) -> Self::Output {
        self.r * rhs + self.t
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    const EPSILON: f32 = 1e-6;

    fn make_random_se3() -> SE3 {
        SE3::from_random()
    }

    fn make_random_vec3() -> Vec3A {
        let mut rng = rand::rng();
        Vec3A::new(rng.random(), rng.random(), rng.random())
    }

    #[test]
    fn test_identity() {
        let se3 = SE3::IDENTITY;
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
        let rotation = SO3::from_quaternion(Quat::IDENTITY);
        let translation = Vec3A::new(1.0, 2.0, 3.0);
        let se3 = SE3::new(rotation, translation);
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
        let mat = Mat4::IDENTITY;
        let se3 = SE3::from_matrix(&mat);
        assert_relative_eq!(se3.t.x, 0.0);
        assert_relative_eq!(se3.t.y, 0.0);
        assert_relative_eq!(se3.t.z, 0.0);

        let expected_rotation = SO3::IDENTITY.matrix();
        let actual_rotation = se3.r.matrix();
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(actual_rotation.col(i)[j], expected_rotation.col(i)[j],);
            }
        }

        // Test with a specific transformation matrix
        let mat = Mat4::from_cols_array(&[
            1.0, 0.0, 0.0, 0.0, // col0
            0.0, 1.0, 0.0, 0.0, // col1
            0.0, 0.0, 1.0, 0.0, // col2
            1.0, 2.0, 3.0, 1.0, // col3
        ]);
        let se3 = SE3::from_matrix(&mat);
        assert_relative_eq!(se3.t.x, 1.0);
        assert_relative_eq!(se3.t.y, 2.0);
        assert_relative_eq!(se3.t.z, 3.0);
    }

    #[test]
    fn test_from_qxyz() {
        let quat = Quat::from_xyzw(0.1, 0.2, 0.3, 0.9).normalize();
        let xyz = Vec3A::new(1.0, 2.0, 3.0);
        let se3 = SE3::from_qxyz(quat, xyz);

        assert_relative_eq!(se3.r.q.x, quat.x);
        assert_relative_eq!(se3.t.x, xyz.x);
        assert_relative_eq!(se3.t.y, xyz.y);
        assert_relative_eq!(se3.t.z, xyz.z);
    }

    #[test]
    fn test_inverse() {
        // Test with identity
        let se3 = SE3::IDENTITY;
        let inv = se3.inverse();
        assert_relative_eq!(inv.r.q.x, se3.r.q.x);
        assert_relative_eq!(inv.r.q.y, se3.r.q.y);
        assert_relative_eq!(inv.r.q.z, se3.r.q.z);
        assert_relative_eq!(inv.r.q.w, se3.r.q.w);
        assert_relative_eq!(inv.t.x, se3.t.x);
        assert_relative_eq!(inv.t.y, se3.t.y);
        assert_relative_eq!(inv.t.z, se3.t.z);

        // Test with translation only
        let se3 = SE3::new(SO3::IDENTITY, Vec3A::new(1.0, 2.0, 3.0));
        let inv = se3.inverse();
        assert_relative_eq!(inv.t.x, -1.0);
        assert_relative_eq!(inv.t.y, -2.0);
        assert_relative_eq!(inv.t.z, -3.0);

        // Test inverse property: se3 * se3.inverse() = identity
        let se3 = make_random_se3();
        let inv = se3.inverse();
        let result = se3 * inv;
        assert_relative_eq!(result.r.q.x, SE3::IDENTITY.r.q.x, epsilon = EPSILON);
        assert_relative_eq!(result.r.q.y, SE3::IDENTITY.r.q.y, epsilon = EPSILON);
        assert_relative_eq!(result.r.q.z, SE3::IDENTITY.r.q.z, epsilon = EPSILON);
        assert_relative_eq!(result.r.q.w, SE3::IDENTITY.r.q.w, epsilon = EPSILON);
        assert_relative_eq!(result.t.x, SE3::IDENTITY.t.x, epsilon = EPSILON);
        assert_relative_eq!(result.t.y, SE3::IDENTITY.t.y, epsilon = EPSILON);
        assert_relative_eq!(result.t.z, SE3::IDENTITY.t.z, epsilon = EPSILON);
    }

    #[test]
    fn test_matrix() {
        // Test identity
        let se3 = SE3::IDENTITY;
        let mat = se3.matrix();
        let expected = Mat4::IDENTITY;
        for i in 0..4 {
            for j in 0..4 {
                assert_relative_eq!(mat.col(i)[j], expected.col(i)[j]);
            }
        }

        // Test with translation
        let se3 = SE3::new(SO3::IDENTITY, Vec3A::new(1.0, 2.0, 3.0));
        let mat = se3.matrix();
        let expected = Mat4::from_cols_array(&[
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
        let identity = SE3::IDENTITY;

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
        let upsilon = Vec3A::ZERO;
        let omega = Vec3A::ZERO;
        let se3 = SE3::exp(upsilon, omega);

        assert_relative_eq!(se3.r.q.x, SE3::IDENTITY.r.q.x);
        assert_relative_eq!(se3.r.q.y, SE3::IDENTITY.r.q.y);
        assert_relative_eq!(se3.r.q.z, SE3::IDENTITY.r.q.z);
        assert_relative_eq!(se3.r.q.w, SE3::IDENTITY.r.q.w);
        assert_relative_eq!(se3.t.x, 0.0);
        assert_relative_eq!(se3.t.y, 0.0);
        assert_relative_eq!(se3.t.z, 0.0);
    }

    #[test]
    fn test_exp_translation_only() {
        // Pure translation (omega = 0)
        let upsilon = Vec3A::new(1.0, 2.0, 3.0);
        let omega = Vec3A::ZERO;
        let se3 = SE3::exp(upsilon, omega);

        assert_relative_eq!(se3.r.q.x, SE3::IDENTITY.r.q.x);
        assert_relative_eq!(se3.r.q.y, SE3::IDENTITY.r.q.y);
        assert_relative_eq!(se3.r.q.z, SE3::IDENTITY.r.q.z);
        assert_relative_eq!(se3.r.q.w, SE3::IDENTITY.r.q.w);
        assert_relative_eq!(se3.t.x, upsilon.x);
        assert_relative_eq!(se3.t.y, upsilon.y);
        assert_relative_eq!(se3.t.z, upsilon.z);
    }

    #[test]
    fn test_log_identity() {
        let se3 = SE3::IDENTITY;
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
        let translation = Vec3A::new(1.0, 2.0, 3.0);
        let se3 = SE3::new(SO3::IDENTITY, translation);
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
        let upsilon = Vec3A::new(0.5, -0.5, 1.0);
        let omega = Vec3A::new(0.1, 0.2, -0.3);

        let se3 = SE3::exp(upsilon, omega);
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
        let upsilon = Vec3A::new(1.0, 2.0, 3.0);
        let omega = Vec3A::new(0.1, -0.2, 0.3);

        let hat_matrix = SE3::hat(upsilon, omega);
        let (vee_upsilon, vee_omega) = SE3::vee(hat_matrix);

        assert_relative_eq!(vee_upsilon.x, upsilon.x);
        assert_relative_eq!(vee_upsilon.y, upsilon.y);
        assert_relative_eq!(vee_upsilon.z, upsilon.z);
        assert_relative_eq!(vee_omega.x, omega.x);
        assert_relative_eq!(vee_omega.y, omega.y);
    }

    #[test]
    fn test_hat_structure() {
        let upsilon = Vec3A::new(1.0, 2.0, 3.0);
        let omega = Vec3A::new(0.1, -0.2, 0.3);
        let hat_matrix = SE3::hat(upsilon, omega);

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
        let se3 = SE3::IDENTITY;
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
        let se3 = SE3::from_random();

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
        assert_relative_eq!(result.r.q.x, SE3::IDENTITY.r.q.x, epsilon = EPSILON);
        assert_relative_eq!(result.r.q.y, SE3::IDENTITY.r.q.y, epsilon = EPSILON);
        assert_relative_eq!(result.r.q.z, SE3::IDENTITY.r.q.z, epsilon = EPSILON);
        assert_relative_eq!(result.r.q.w, SE3::IDENTITY.r.q.w, epsilon = EPSILON);
        assert_relative_eq!(result.t.x, SE3::IDENTITY.t.x, epsilon = EPSILON);
        assert_relative_eq!(result.t.y, SE3::IDENTITY.t.y, epsilon = EPSILON);
        assert_relative_eq!(result.t.z, SE3::IDENTITY.t.z, epsilon = EPSILON);
    }

    #[test]
    fn test_matrix_from_matrix_roundtrip() {
        let se3 = make_random_se3();
        let matrix = se3.matrix();
        let se3_reconstructed = SE3::from_matrix(&matrix);

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
        for i in 0..6 {
            for j in 0..6 {
                out[i] += mat[i][j] * v[j];
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
        for i in 0..6 {
            for j in 0..6 {
                m[i][j] = mat[j][i];
            }
        }
        m
    }

    /// Diverse translation/rotation test pairs (ρ, θ).
    fn test_pairs() -> [(Vec3A, Vec3A); 5] {
        [
            (Vec3A::new(0.1, 0.2, 0.3), Vec3A::new(0.02, 0.03, 0.04)),
            (Vec3A::new(-0.3, 0.5, 0.1), Vec3A::new(0.7, 0.0, 0.0)),
            (Vec3A::new(0.0, 0.0, 0.0), Vec3A::new(0.001, -0.002, 0.003)), // tiny θ
            (Vec3A::new(0.5, -0.4, 0.2), Vec3A::ZERO),                     // pure translation
            (Vec3A::new(0.01, 0.02, 0.03), Vec3A::new(-0.01, 0.02, -0.03)), // small mixed
        ]
    }

    #[test]
    fn test_left_jacobian_se3() {
        for (rho, omega) in test_pairs() {
            let jl = SE3::left_jacobian(rho, omega);

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
            let jr = SE3::right_jacobian(rho, omega);

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
            let jl_minus = SE3::left_jacobian(-rho, -omega);
            let jr = SE3::right_jacobian(rho, omega);

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
        let jl = SE3::left_jacobian(rho, omega);
        let jr_t = mat6_transpose(&SE3::right_jacobian(rho, omega));

        for i in 0..6 {
            for j in 0..6 {
                assert_relative_eq!(jl[i][j], jr_t[i][j], epsilon = 1e-3);
            }
        }
    }
}
