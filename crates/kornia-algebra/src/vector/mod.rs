//! Vector types module.
//!
//! This module provides vector types for kornia-rs.
//!
//! The concrete newtype wrappers over `glam` are implemented via a small
//! `macro_rules!` helper placed in `vector.rs` so we can easily support
//! multiple precisions (e.g. `f32`, `f64`) without copy-pasting boilerplate.
//!

#[macro_use]
mod vec;

mod vec2;
mod vec3;
mod vec3a;
mod vec4;

pub use vec3a::Vec3AF32;
pub use {vec2::Vec2F32, vec2::Vec2F64};
pub use {vec3::Vec3F32, vec3::Vec3F64};
pub use {vec4::Vec4F32, vec4::Vec4F64};
