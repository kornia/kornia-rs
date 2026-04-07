//! Shared traits and types for camera models.

use crate::pose::Pose3d;
use kornia_algebra::{Vec2F64, Vec3F64};
use kornia_image::ImageSize;

/// Projection rejection reasons common to multiple camera models.
///
/// Used by [`CameraModel::project_to_image`] and model-specific projection
/// helpers to communicate why a 3D point could not be projected to a valid
/// pixel coordinate.
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
///
/// Implementors provide the core projection / unprojection pipeline so that
/// downstream algorithms (e.g. triangulation, PnP) can work generically with
/// any supported camera type.
pub trait CameraModel {
    /// Returns the foundational intrinsic parameters: `(fx, fy, cx, cy)`.
    fn intrinsics(&self) -> (f64, f64, f64, f64);

    /// Projects a 3D point in the camera frame to a 2D pixel coordinate.
    ///
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
    ///
    /// The returned vector typically lies on the unit sphere or the `z = 1`
    /// ideal image plane, depending on the model.
    fn unproject(&self, pixel: &Vec2F64) -> Option<Vec3F64>;
}

/// Project a world point through a pose and any generic camera model.
///
/// Transforms `point_world` into the camera frame via `pose`, then projects
/// using the supplied camera model.
///
/// # Arguments
///
/// * `camera` - Any type implementing [`CameraModel`].
/// * `pose` - World-to-camera rigid body transform.
/// * `point_world` - 3D point in world coordinates.
///
/// # Returns
///
/// `Some((u, v, z_cam))` on success, where `(u, v)` are pixel coordinates and
/// `z_cam` is the depth in the camera frame. Returns `None` if the projection
/// is invalid for the given model (e.g. behind the camera for pinhole).
///
/// # Example
///
/// ```rust
/// use kornia_3d::camera::{project_point, PinholeCamera};
/// use kornia_3d::pose::Pose3d;
/// use kornia_algebra::{Mat3F64, Vec3F64};
///
/// let cam = PinholeCamera {
///     fx: 200.0, fy: 200.0, cx: 320.0, cy: 240.0,
///     k1: 0.0, k2: 0.0, p1: 0.0, p2: 0.0,
/// };
/// let pose = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::ZERO);
/// let pt = Vec3F64::new(0.0, 0.0, 5.0);
/// let (u, v, z) = project_point(&cam, &pose, &pt).unwrap();
/// assert!((u - 320.0).abs() < 1e-12);
/// assert!((v - 240.0).abs() < 1e-12);
/// assert!((z - 5.0).abs() < 1e-12);
/// ```
pub fn project_point<C: CameraModel>(
    camera: &C,
    pose: &Pose3d,
    point_world: &Vec3F64,
) -> Option<(f64, f64, f64)> {
    let p_cam = pose.transform_point(point_world);
    let pixel = camera.project(&p_cam)?;
    Some((pixel.x, pixel.y, p_cam.z))
}
