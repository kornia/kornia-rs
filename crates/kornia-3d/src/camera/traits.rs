//! Shared traits and types for camera models.

use crate::pose::Pose3d;
use kornia_algebra::{Vec2F64, Vec3F64};
use kornia_image::ImageSize;

/// Normal projection rejection reasons common to multiple camera models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionReject {
    /// Point projection is mathematically invalid (e.g. exactly at the optical center).
    InvalidProjection,
    /// Point depth is below or equal to the requested minimum depth.
    BelowMinDepth,
    /// Projection lies outside the given image bounds.
    OutOfImage,
}

/// Unified trait bounding common operations across differing geometric camera
/// models (e.g. Pinhole, Fisheye).
pub trait CameraModel {
    /// Returns the foundational intrinsic parameters: `(fx, fy, cx, cy)`.
    fn intrinsics(&self) -> (f64, f64, f64, f64);

    /// Projects a 3D point in the camera frame to a 2D pixel coordinate.
    /// Returns `None` if the projection is mathematically invalid (for instance,
    /// behind the camera depending on the model, or at the exact optical center).
    fn project(&self, p_cam: &Vec3F64) -> Option<Vec2F64>;

    /// Projects a 3D point in the camera frame to a 2D pixel coordinate,
    /// explicitly returning an error if the point is invalid or falls outside
    /// the provided `image_size`.
    fn project_to_image(
        &self,
        p_cam: &Vec3F64,
        image_size: ImageSize,
    ) -> Result<Vec2F64, ProjectionReject>;

    /// Unprojects a 2D pixel coordinate into a 3D directional ray in the camera frame.
    /// Normally represented where Z=1 (ideal image plane).
    fn unproject(&self, pixel: &Vec2F64) -> Option<Vec3F64>;
}

/// Project a world point through a pose and any generic camera model.
///
/// Returns `(u, v, z_cam)` or `None` if the point projection is mathematically
/// invalid (for example, behind the camera or out of bounds for the model).
pub fn project_point<C: CameraModel>(
    camera: &C,
    pose: &Pose3d,
    point_world: &Vec3F64,
) -> Option<(f64, f64, f64)> {
    let p_cam = pose.transform_point(point_world);
    let pixel = camera.project(&p_cam)?;
    Some((pixel.x, pixel.y, p_cam.z))
}
