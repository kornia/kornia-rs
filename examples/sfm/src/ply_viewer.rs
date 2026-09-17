//! Visualize a PLY point cloud and the camera poses in the rerun GUI.
//!
//! Reads the file with `kornia::k3d::io::ply::read_ply_binary` (the same
//! reader the `ply_rerun` example uses), logs positions + colors, and draws
//! each registered camera as a pinhole frustum (pattern from
//! `examples/colmap_rerun`).

use std::error::Error;
use std::path::Path;

use kornia_3d::pose::Pose3d;
use kornia_algebra::QuatF64;

/// Convert a camera pose (`T_world_cam`) into `(translation, quaternion_wxyz)`
/// for rerun's `Transform3D::from_translation_rotation`.
///
/// rerun transforms are parent→child (`ChildFromParent`), so the camera's
/// world transform is the inverse of `T_world_cam`: its translation is the
/// camera center, its rotation is the cam→world rotation.
fn pose_to_rerun(pose: &Pose3d) -> ([f32; 3], [f32; 4]) {
    let cam_world = pose.inverse();
    let t = [
        cam_world.translation.x as f32,
        cam_world.translation.y as f32,
        cam_world.translation.z as f32,
    ];
    // QuatF64::to_array is [x, y, z, w] (glam); rerun wants wxyz.
    let q = QuatF64::from_mat3(&cam_world.rotation).to_array();
    (
        [t[0], t[1], t[2]],
        [q[3] as f32, q[0] as f32, q[1] as f32, q[2] as f32],
    )
}

/// Open `path` in the rerun viewer and block until the viewer is closed.
///
/// Logs the point cloud and each registered camera pose as a pinhole frustum.
/// Requires the rerun viewer (`pip install rerun-sdk` or from rerun.io).
///
/// # Arguments
///
/// * `path` - Path to a binary PLY file (XYZRgbNormals layout).
/// * `views` - Per-frame poses from the reconstruction (`None` = unregistered).
/// * `fx` / `fy` / `cx` / `cy` - Camera intrinsics (pixels).
/// * `width` / `height` - Frame size in pixels (for the pinhole frustum).
///
/// # Errors
///
/// Returns an error if the PLY cannot be read or rerun cannot spawn a viewer.
#[allow(clippy::too_many_arguments)] // view_world signature mirrors the reconstruction's inputs
pub fn view_world(
    path: &Path,
    views: &[Option<Pose3d>],
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    width: usize,
    height: usize,
) -> Result<(), Box<dyn Error>> {
    let pointcloud =
        kornia::k3d::io::ply::read_ply_binary(path, kornia::k3d::io::ply::PlyType::XYZRgbNormals)?;
    eprintln!(
        "[view] read {} points from {}",
        pointcloud.len(),
        path.display()
    );

    let rec = rerun::RecordingStreamBuilder::new("SfM Point Cloud Viewer").spawn()?;
    rec.log("/", &rerun::ViewCoordinates::RIGHT_HAND_Y_DOWN())?;

    let points: Vec<rerun::Position3D> = pointcloud
        .points()
        .iter()
        .map(|p| rerun::Position3D::new(p[0] as f32, p[1] as f32, p[2] as f32))
        .collect();

    let colors: Vec<rerun::Color> = pointcloud
        .colors()
        .map(|colors| {
            colors
                .iter()
                .map(|c| rerun::Color::from_rgb(c[0], c[1], c[2]))
                .collect()
        })
        .unwrap_or_default();

    rec.log(
        "world/points",
        &rerun::Points3D::new(points).with_colors(colors),
    )?;

    // Camera frustums: one entity per registered view.
    let mut n_cameras = 0;
    for (i, view) in views.iter().enumerate() {
        let Some(pose) = view else { continue };
        let (t, q) = pose_to_rerun(pose);
        rec.log(
            format!("world/camera_{i}"),
            &rerun::Transform3D::from_translation_rotation(
                t,
                rerun::Quaternion::from_wxyz([q[0], q[1], q[2], q[3]]),
            )
            .with_relation(rerun::TransformRelation::ChildFromParent),
        )?;
        rec.log(format!("world/camera_{i}"), &rerun::ViewCoordinates::RDF())?;
        rec.log(
            format!("world/camera_{i}/image"),
            &rerun::Pinhole::from_focal_length_and_resolution(
                [fx as f32, fy as f32],
                [width as f32, height as f32],
            )
            .with_principal_point([cx as f32, cy as f32]),
        )?;
        n_cameras += 1;
    }
    eprintln!(
        "[view] logged {} cameras and {} points. Close the viewer to exit.",
        n_cameras,
        pointcloud.len()
    );

    // Keep the process alive so the viewer stays connected.
    loop {
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_algebra::{Mat3F64, Vec3F64};

    #[test]
    fn pose_to_rerun_identity_is_origin_and_identity_quat() {
        let (t, q) = pose_to_rerun(&Pose3d::IDENTITY);
        assert_eq!(t, [0.0, 0.0, 0.0]);
        // wxyz identity = (1, 0, 0, 0)
        assert!((q[0] - 1.0).abs() < 1e-6);
        assert!(q[1].abs() < 1e-6 && q[2].abs() < 1e-6 && q[3].abs() < 1e-6);
    }

    #[test]
    fn pose_to_rerun_uses_camera_center_from_inverse() {
        // T_world_cam with a 90° yaw and t = (-5,0,0): its camera center in
        // world is -R^T * t = (0,0,5).
        let rot = Mat3F64::from_cols(
            Vec3F64::new(0.0, 0.0, -1.0),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(1.0, 0.0, 0.0),
        );
        let pose = Pose3d::new(rot, Vec3F64::new(-5.0, 0.0, 0.0));
        let (t, q) = pose_to_rerun(&pose);
        assert!((t[0]).abs() < 1e-4, "t0={}", t[0]);
        assert!((t[1]).abs() < 1e-4, "t1={}", t[1]);
        assert!((t[2] - 5.0).abs() < 1e-3, "t2={}", t[2]);
        // Rotation is a quarter turn: quaternion must be unit length.
        let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "unit quaternion, got {norm}");
    }
}
