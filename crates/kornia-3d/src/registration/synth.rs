//! Synthetic analytic depth renderer for registration tests.
//!
//! Raycasts a scene of (infinite) planes and spheres from a posed pinhole
//! camera, producing either a `u16` millimetre depth image or an exact `f32`
//! vertex map. Shared by the `rgbd` and `icp` test suites.
//!
//! Conventions match [`super::rgbd`]: OpenCV pinhole, +Z forward, camera pose
//! given as `T_world_cam` (camera-to-world), depth is z-depth in millimetres.

use super::rgbd::DepthIntrinsics;

/// Infinite plane `normal . p = d` in world coordinates.
pub(crate) struct Plane {
    /// Unit plane normal.
    pub normal: [f64; 3],
    /// Signed distance so that `normal . p = d` on the plane.
    pub d: f64,
}

/// Sphere in world coordinates.
pub(crate) struct Sphere {
    /// Center of the sphere.
    pub center: [f64; 3],
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
    /// Concave corner of three orthogonal planes plus a sphere, all in front of
    /// a camera at the origin looking down +Z.
    ///
    /// Three mutually orthogonal planes constrain all six DoF for
    /// point-to-plane ICP (a single plane is degenerate); the sphere adds
    /// curvature. Designed for the 320x180 / fx=200 test intrinsics.
    pub fn corner_and_sphere() -> Self {
        Scene {
            planes: vec![
                Plane {
                    normal: [0.0, 0.0, 1.0],
                    d: 2.0,
                },
                Plane {
                    normal: [1.0, 0.0, 0.0],
                    d: 0.8,
                },
                Plane {
                    normal: [0.0, 1.0, 0.0],
                    d: 0.6,
                },
            ],
            spheres: vec![Sphere {
                center: [-0.2, -0.15, 1.2],
                radius: 0.3,
            }],
        }
    }

    /// Closest hit along ray `origin + t * dir`, returned as the parameter `t`.
    fn raycast(&self, origin: &[f64; 3], dir: &[f64; 3]) -> Option<f64> {
        let mut best: Option<f64> = None;
        let mut consider = |t: f64| {
            if t > NEAR_M && best.is_none_or(|b| t < b) {
                best = Some(t);
            }
        };

        for plane in &self.planes {
            let denom = dot(&plane.normal, dir);
            if denom.abs() < 1e-12 {
                continue;
            }
            consider((plane.d - dot(&plane.normal, origin)) / denom);
        }

        for sphere in &self.spheres {
            let oc = [
                origin[0] - sphere.center[0],
                origin[1] - sphere.center[1],
                origin[2] - sphere.center[2],
            ];
            let a = dot(dir, dir);
            let b = dot(&oc, dir);
            let c = dot(&oc, &oc) - sphere.radius * sphere.radius;
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

fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Z-depth (in metres) of the closest surface at pixel `(u, v)`, or `None`.
///
/// The camera ray direction is scaled so its camera-frame z component is 1,
/// making the ray parameter equal to the z-depth.
fn raycast_pixel(
    scene: &Scene,
    intr: &DepthIntrinsics,
    rot_world_cam: &[[f64; 3]; 3],
    t_world_cam: &[f64; 3],
    u: usize,
    v: usize,
) -> Option<f64> {
    let dir_cam = [
        (u as f64 - intr.cx) / intr.fx,
        (v as f64 - intr.cy) / intr.fy,
        1.0,
    ];
    let dir_world = [
        dot(&rot_world_cam[0], &dir_cam),
        dot(&rot_world_cam[1], &dir_cam),
        dot(&rot_world_cam[2], &dir_cam),
    ];
    scene.raycast(t_world_cam, &dir_world)
}

/// Render a `u16` millimetre depth image (0 = miss) from pose `T_world_cam`.
pub(crate) fn render_depth_mm(
    scene: &Scene,
    intr: &DepthIntrinsics,
    rot_world_cam: &[[f64; 3]; 3],
    t_world_cam: &[f64; 3],
) -> Vec<u16> {
    let mut depth = vec![0u16; intr.width * intr.height];
    for v in 0..intr.height {
        for u in 0..intr.width {
            if let Some(z) = raycast_pixel(scene, intr, rot_world_cam, t_world_cam, u, v) {
                let mm = (z * 1000.0).round();
                if mm >= 1.0 && mm <= u16::MAX as f64 {
                    depth[v * intr.width + u] = mm as u16;
                }
            }
        }
    }
    depth
}

/// Render an exact `f32` camera-frame vertex map (no millimetre quantization),
/// `[0, 0, 0]` = miss. Same conventions as [`super::rgbd::depth_to_vertex_map`].
pub(crate) fn render_vertex_map(
    scene: &Scene,
    intr: &DepthIntrinsics,
    rot_world_cam: &[[f64; 3]; 3],
    t_world_cam: &[f64; 3],
) -> Vec<[f32; 3]> {
    let mut verts = vec![[0.0f32; 3]; intr.width * intr.height];
    for v in 0..intr.height {
        for u in 0..intr.width {
            if let Some(z) = raycast_pixel(scene, intr, rot_world_cam, t_world_cam, u, v) {
                let p = intr.unproject(u as f64, v as f64, z);
                verts[v * intr.width + u] = [p[0] as f32, p[1] as f32, p[2] as f32];
            }
        }
    }
    verts
}
