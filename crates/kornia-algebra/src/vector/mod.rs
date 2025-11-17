//! Vector types module.
//!
//! This module provides vector types for kornia-rs:
//! - Vec2F32: 2D vector
//! - Vec3F32: 3D vector
//! - Vec3AF32: 3D aligned vector (SIMD-optimized)
//! - Vec4F32: 4D vector

mod vec2;
mod vec3;
mod vec3a;
mod vec4;

pub use vec2::Vec2F32;
pub use vec3::Vec3F32;
pub use vec3a::Vec3AF32;
pub use vec4::Vec4F32;
