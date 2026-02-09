//! # kornia-algebra
//!
//! **kornia-algebra** is a core library for geometric algebra in the [kornia-rs](https://github.com/kornia/kornia-rs) ecosystem. It provides a robust set of types for linear algebra and Lie theory, designed specifically for computer vision and robotics.
//!
//! It is built on top of the excellent [glam](https://github.com/bitshifter/glam-rs) crate, providing strict newtypes and extending functionality for the algebra types.
//!
//! Most operations are SIMD accelerated thanks to `glam`.
//!
//! ## Features
//!
//! - **Vectors**: `Vec2F32`, `Vec3F32`, `Vec3AF32`, `Vec4F32` (and `F64` variants) with arithmetic operations, dot products, and length calculations.
//! - **Matrices**: `Mat2F32`, `Mat3F32`, `Mat3AF32`, `Mat4F32` (and `F64` variants) with matrix multiplication, transpose, inverse, and diagonal construction.
//! - **Quaternions**: `QuatF32`/`QuatF64` — unit quaternions are isomorphic to SU(2), the
//!   double cover of SO(3). `q` and `-q` represent the same rotation. See the [`quat`] module
//!   for the mathematical context.
//! - **Lie Groups**: Implementations of Lie groups used in robotics and vision (see the
//!   [`lie`] module for the full group hierarchy and the SU(2) → SO(3) double cover):
//!   - **SO(2)**: 2D Rotations (`SO2F32`) — topology: S¹.
//!   - **SE(2)**: 2D Rigid Body Transformations (`SE2F32`) — semi-direct product SO(2) ⋉ R².
//!   - **SO(3)**: 3D Rotations (`SO3F32`) — stored as unit quaternions (SU(2)), topology: RP³.
//!   - **SE(3)**: 3D Rigid Body Transformations (`SE3F32`) — semi-direct product SO(3) ⋉ R³.
//!   - **RxSO(3)**: 3D Rotation + Scale (`RxSO3F32`) — direct product R⁺ × SO(3).
//!   - **Sim(3)**: 3D Similarity Transforms (`Sim3F32`) — semi-direct product (R⁺ × SO(3)) ⋉ R³.
//! - **Lie Algebra Operations**: Full support for manifold operations:
//!   - Exponential (`exp`) and Logarithmic (`log`) maps.
//!   - Adjoint representation (`adjoint`).
//!   - Hat (`hat`) and Vee (`vee`) operators.
//!   - Jacobians (left and right) for optimization.
//! - **Differentiation**: Analytical Jacobians for Lie group operations, essential for non-linear least squares and Kalman filtering.
//!
//! ## Usage
//!
//! Add this to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! kornia-algebra = "0.1"
//! ```
//!
//! ### Basic Linear Algebra
//!
//! ```rust
//! use kornia_algebra::{Vec3F32, Mat3F32};
//!
//! let v = Vec3F32::new(1.0, 2.0, 3.0);
//! let m = Mat3F32::IDENTITY;
//!
//! let v_transformed = m * v;
//! assert_eq!(v_transformed, v);
//! ```
//!
//! ### 3D Rotations (SO3)
//!
//! ```rust
//! use kornia_algebra::{SO3F32, Vec3AF32};
//!
//! // Create a rotation of 90 degrees around X axis
//! let rotation = SO3F32::exp(Vec3AF32::new(std::f32::consts::FRAC_PI_2, 0.0, 0.0));
//!
//! let point = Vec3AF32::new(0.0, 1.0, 0.0);
//! let rotated_point = rotation * point;
//!
//! // (0, 1, 0) rotated 90 deg around X becomes (0, 0, 1)
//! assert!((rotated_point - Vec3AF32::new(0.0, 0.0, 1.0)).length() < 1e-6);
//! ```
//!
//! ### Rigid Body Transformations (SE3)
//!
//! ```rust
//! use kornia_algebra::{SE3F32, SO3F32, Vec3AF32};
//!
//! // Rotation + Translation
//! let rotation = SO3F32::IDENTITY;
//! let translation = Vec3AF32::new(1.0, 0.0, 0.0);
//! let pose = SE3F32::new(rotation, translation);
//!
//! let point = Vec3AF32::new(0.0, 0.0, 0.0);
//! let transformed = pose * point;
//!
//! assert_eq!(transformed, Vec3AF32::new(1.0, 0.0, 0.0));
//! ```
//!
//! ### Lie Algebra Interpolation
//!
//! ```rust
//! use kornia_algebra::{SE3F32, Vec3AF32};
//!
//! let start = SE3F32::IDENTITY;
//! // Move 1 meter in X and rotate 90 degrees around Z
//! let translation = Vec3AF32::new(1.0, 0.0, 0.0);
//! let rotation = Vec3AF32::new(0.0, 0.0, std::f32::consts::FRAC_PI_2);
//! let end = SE3F32::exp(translation, rotation);
//!
//! // Interpolate halfway
//! let t = 0.5;
//! let (delta_trans, delta_rot) = start.rminus(&end); // Logarithmic difference
//! let interpolated = start.rplus(delta_trans * t, delta_rot * t);
//! ```
//!
//! ### 2D Rigid Body Transformations (SE2)
//!
//! ```rust
//! use kornia_algebra::{SE2F32, SO2F32, Vec2F32};
//!
//! // Create from angle (radians) and translation
//! let rotation = SO2F32::exp(std::f32::consts::FRAC_PI_2); // 90 deg
//! let translation = Vec2F32::new(10.0, 0.0);
//! let pose = SE2F32::new(rotation, translation);
//!
//! // Transform a point
//! let point = Vec2F32::new(1.0, 0.0);
//! let transformed = pose * point;
//!
//! // (1,0) rotated 90 deg -> (0,1) + (10,0) -> (10,1)
//! assert!((transformed - Vec2F32::new(10.0, 1.0)).length() < 1e-6);
//! ```
//!
//! ### Optimization Tools (Jacobians)
//!
//! ```rust
//! use kornia_algebra::{SE3F32, Vec3AF32};
//!
//! let pose = SE3F32::from_random();
//! let point = Vec3AF32::new(1.0, 0.0, 0.0);
//!
//! // Compute Jacobians for optimization
//! // Right Jacobian maps tangent space variations to group variations
//! let translation_twist = Vec3AF32::new(0.1, 0.0, 0.0); // Small translation in x
//! let rotation_twist = Vec3AF32::new(0.0, 0.0, 0.0);
//! let jacobian = SE3F32::right_jacobian(translation_twist, rotation_twist);
//! // Use jacobian in optimization algorithms...
//! ```
//!
//! ### Data Conversions
//!
//! ```rust
//! use kornia_algebra::{SE3F32, SO3F32, Vec3AF32};
//!
//! // Create from array [qx, qy, qz, qw, tx, ty, tz]
//! let pose_arr = [0.0, 0.0, 0.0, 1.0, 10.0, 20.0, 30.0];
//! let pose = SE3F32::from_array(pose_arr);
//!
//! // Convert back to array
//! let arr = pose.to_array();
//! assert_eq!(arr, pose_arr);
//!
//! // Convert to 4x4 Matrix
//! let mat4 = pose.matrix();
//! ```

// Algebraic types and Lie groups
mod lie;
mod mat;
mod quat;
mod vec;

// Linear algebra operations
pub mod linalg;

//param module
pub mod param;
pub use param::{Param, ParamError};

// Optimization module
pub mod optim;

// Re-export types at crate root for convenience
pub use lie::{RxSO3F32, Sim3F32, SE2F32, SE3F32, SO2F32, SO3F32};
pub use mat::{DMatF32, DMatF64, Mat2F32, Mat2F64, Mat3AF32, Mat3F32, Mat3F64, Mat4F32, Mat4F64};
pub use quat::{QuatF32, QuatF64};
pub use vec::{DVecF32, DVecF64, Vec2F32, Vec2F64, Vec3AF32, Vec3F32, Vec3F64, Vec4F32, Vec4F64};
