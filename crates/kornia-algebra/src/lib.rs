//! Lie groups and algebras for kornia-rs.
//!
//! This crate provides:
//! - Unified algebraic types (`types` module) using newtype pattern over glam backend
//! - Lie group implementations (SO2, SO3, SE2, SE3)

// Algebraic types
mod mat2;
mod mat3;
mod mat3a;
mod mat4;
mod quat;
mod vec2;
mod vec3;
mod vec3a;
mod vec4;

// Re-export types at crate root for convenience
pub use mat2::Mat2;
pub use mat3::Mat3;
pub use mat3a::Mat3A;
pub use mat4::Mat4;
pub use quat::Quat;
pub use vec2::Vec2;
pub use vec3::Vec3;
pub use vec3a::Vec3A;
pub use vec4::Vec4;

// Type aliases for explicit precision (single precision / f32)
pub type Vec2F32 = Vec2;
pub type Vec3F32 = Vec3;
pub type Vec3AF32 = Vec3A;
pub type Vec4F32 = Vec4;
pub type Mat2F32 = Mat2;
pub type Mat3F32 = Mat3;
pub type Mat3AF32 = Mat3A;
pub type Mat4F32 = Mat4;
pub type QuatF32 = Quat;

// Isometry type aliases
pub type Isometry2F32 = lie::SE2;
pub type Isometry3F32 = lie::SE3;

// Re-export glam types that are used directly
pub use glam::Affine3A;

// Lie groups
pub mod lie;
