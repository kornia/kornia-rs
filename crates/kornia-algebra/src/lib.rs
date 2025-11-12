//! Lie groups and algebras for kornia-rs.
//!
//! This crate provides:
//! - Unified algebraic types (`types` module) using newtype pattern over glam backend
//! - Lie group implementations (SO2, SO3, SE2, SE3)

// Algebraic types
pub mod mat2;
pub mod mat3;
pub mod mat3a;
pub mod mat4;
pub mod quat;
pub mod vec2;
pub mod vec3;
pub mod vec3a;
pub mod vec4;

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

// Re-export glam types that are used directly
pub use glam::Affine3A;

// Unified algebraic types module (re-exports)
pub mod types;

// Lie groups
pub mod se2;
pub mod se3;
pub mod so2;
pub mod so3;
