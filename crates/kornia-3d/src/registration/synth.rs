//! Synthetic analytic depth renderer for the registration tests.
//!
//! Raycasts a scene of (infinite) planes and spheres from a posed pinhole camera, producing a
//! `u16` millimetre depth image. Conventions match [`super::rgbd`]: OpenCV pinhole, +Z forward,
//! camera pose given as `T_world_cam` (camera-to-world), depth is z-depth in millimetres.

use crate::camera::PinholeCamera;
use kornia_algebra::{Mat3F64, Vec3F64};
use kornia_image::{Image, ImageError, ImageSize};

/// Size of the test images.
pub(crate) const TEST_SIZE: ImageSize = ImageSize {
    width: 320,
    height: 180,
};

/// Undistorted camera for [`TEST_SIZE`] images: `fx = fy = 200`, principal point at the centre.
pub(crate) fn test_camera() -> PinholeCamera {
    PinholeCamera {
        fx: 200.0,
        fy: 200.0,
        cx: 159.5,
        cy: 89.5,
        ..PinholeCamera::IDENTITY
    }
}

/// Infinite plane `normal . p = d` in world coordinates.
pub(crate) struct Plane {
    /// Unit plane normal.
    pub normal: Vec3F64,
    /// Signed distance so that `normal . p = d` on the plane.
    pub d: f64,
}

/// Sphere in world coordinates.
pub(crate) struct Sphere {
    /// Centre of the sphere.
    pub center: Vec3F64,
    /// Radius of the sphere.
    pub radius: f64,
}

/// A raycastable test scene.
pub(crate) struct Scene {
    /// Infinite planes.
    pub planes: Vec<Plane>,
    /// Spheres.
    pub spheres: Vec<Sphere>,
}

/// Near clip: hits closer than this are ignored.
const NEAR_M: f64 = 0.05;

impl Scene {
    /// Concave corner of three orthogonal planes plus a sphere, all in front of a camera at the
    /// origin looking down +Z.
    ///
    /// Three mutually orthogonal planes constrain all six DoF for point-to-plane ICP (a single
    /// plane is degenerate); the sphere adds curvature. Designed for [`test_camera`].
    pub fn corner_and_sphere() -> Self {
        Scene {
            planes: vec![
                Plane {
                    normal: Vec3F64::new(0.0, 0.0, 1.0),
                    d: 2.0,
                },
                Plane {
                    normal: Vec3F64::new(1.0, 0.0, 0.0),
                    d: 0.8,
                },
                Plane {
                    normal: Vec3F64::new(0.0, 1.0, 0.0),
                    d: 0.6,
                },
            ],
            spheres: vec![Sphere {
                center: Vec3F64::new(-0.2, -0.15, 1.2),
                radius: 0.3,
            }],
        }
    }

    /// Closest hit along ray `origin + t * dir`, returned as the parameter `t`.
    fn raycast(&self, origin: Vec3F64, dir: Vec3F64) -> Option<f64> {
        let mut best: Option<f64> = None;
        let mut consider = |t: f64| {
            if t > NEAR_M && best.is_none_or(|b| t < b) {
                best = Some(t);
            }
        };

        for plane in &self.planes {
            let denom = plane.normal.dot(dir);
            if denom.abs() < 1e-12 {
                continue;
            }
            consider((plane.d - plane.normal.dot(origin)) / denom);
        }

        for sphere in &self.spheres {
            let oc = origin - sphere.center;
            let a = dir.dot(dir);
            let b = oc.dot(dir);
            let c = oc.dot(oc) - sphere.radius * sphere.radius;
            let disc = b * b - a * c;
            if disc < 0.0 {
                continue;
            }
            let sqrt_disc = disc.sqrt();
            consider((-b - sqrt_disc) / a);
            consider((-b + sqrt_disc) / a);
        }

        best
    }
}

/// Z-depth (in metres) of the closest surface at pixel `(u, v)`, or `None`.
///
/// The camera ray direction is scaled so its camera-frame z component is 1, making the ray
/// parameter equal to the z-depth.
fn raycast_pixel(
    scene: &Scene,
    camera: &PinholeCamera,
    rot_world_cam: &Mat3F64,
    t_world_cam: &Vec3F64,
    u: usize,
    v: usize,
) -> Option<f64> {
    let dir_cam = Vec3F64::new(
        (u as f64 - camera.cx) / camera.fx,
        (v as f64 - camera.cy) / camera.fy,
        1.0,
    );
    scene.raycast(*t_world_cam, *rot_world_cam * dir_cam)
}

/// Renders a `u16` millimetre depth image (0 = miss) from pose `T_world_cam`.
pub(crate) fn render_depth_mm(
    scene: &Scene,
    camera: &PinholeCamera,
    size: ImageSize,
    rot_world_cam: &Mat3F64,
    t_world_cam: &Vec3F64,
) -> Result<Image<u16, 1>, ImageError> {
    let mut depth = Image::from_size_val(size, 0u16)?;
    for (idx, px) in depth.as_slice_mut().iter_mut().enumerate() {
        let (u, v) = (idx % size.width, idx / size.width);
        if let Some(z) = raycast_pixel(scene, camera, rot_world_cam, t_world_cam, u, v) {
            let mm = (z * 1000.0).round();
            if mm >= 1.0 && mm <= u16::MAX as f64 {
                *px = mm as u16;
            }
        }
    }
    Ok(depth)
}
