//! Shared correspondence sample types consumed by [`super::Estimator`] impls.
//!
//! These structs are deliberately f64 / column-major-friendly so that the
//! same input slice can flow into the F/E/H trio (two-view) or the PnP path
//! (3D→2D) without copying. Estimators that work in a different precision
//! (e.g. EPnP in f32) convert at the trait boundary.
//!
//! Using owned, named-field structs over tuples follows the project
//! convention for new public APIs and makes the corresponding PyO3 bindings
//! straightforward (`#[pyclass(frozen)]`).

use kornia_algebra::{Vec2F64, Vec3F64};

/// One 2D-2D correspondence between two images.
///
/// Coordinate convention is intentionally not enforced here — most estimators
/// expect *pixel* coordinates and apply Hartley normalization internally
/// (`FundamentalEstimator`, `HomographyEstimator`), while
/// `EssentialEstimator` (5-point) expects already-normalized coordinates
/// (i.e. `K⁻¹ · [x, y, 1]ᵀ`). See each estimator's docs.
///
/// `#[repr(C)]` is load-bearing: the [`crate::ransac::estimators::FundamentalEstimator`]
/// NEON path uses `vld4q_f64` to deinterleave two consecutive matches into
/// SoA lane-vectors in one instruction, which assumes the field order
/// `x1.x, x1.y, x2.x, x2.y` is exactly the in-memory layout.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Match2d2d {
    /// Point in image 1.
    pub x1: Vec2F64,
    /// Point in image 2.
    pub x2: Vec2F64,
}

impl Match2d2d {
    /// Construct a 2D-2D match.
    #[inline]
    pub fn new(x1: Vec2F64, x2: Vec2F64) -> Self {
        Self { x1, x2 }
    }
}

/// One 3D-2D correspondence (world point ↔ image pixel) for absolute pose
/// estimation (PnP).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Match2d3d {
    /// 3D point in the world frame.
    pub object: Vec3F64,
    /// Pixel observation.
    pub image: Vec2F64,
}

impl Match2d3d {
    /// Construct a 3D-2D match.
    #[inline]
    pub fn new(object: Vec3F64, image: Vec2F64) -> Self {
        Self { object, image }
    }
}
