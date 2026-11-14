//! Visualize a point cloud and the camera poses in the rerun GUI.
//!
//! Logs the in-memory vertices (positions + colours) and draws each registered
//! camera as a pinhole frustum (pattern from `examples/apriltag_board_multicam`).

use std::error::Error;

use kornia_3d::pose::Pose3d;
use kornia_algebra::QuatF64;

use crate::ply_writer::Vertex;

/// rerun transform relation used to place a camera in the world.
///
/// `ParentFromChild` means the logged transform maps the child frame into the
/// parent frame — camera→world (`T_world_cam`), which is what
/// `kornia_calib::reconstruct` returns. The same convention
/// `examples/apriltag_board_multicam` uses.
const CAMERA_TRANSFORM_RELATION: rerun::TransformRelation =
    rerun::TransformRelation::ParentFromChild;

/// Convert a camera pose (`T_world_cam` = camera→world, as returned by
/// `kornia_calib::reconstruct`) into `(translation, quaternion_wxyz)` for
/// rerun's `Transform3D::from_translation_rotation`. The translation is the
/// camera centre, passed through verbatim (the pose is already C2W).
fn pose_to_rerun(pose: &Pose3d) -> ([f32; 3], [f32; 4]) {
    let t = [
        pose.translation.x as f32,
        pose.translation.y as f32,
        pose.translation.z as f32,
    ];
    // QuatF64::to_array is [x, y, z, w] (glam); rerun wants wxyz.
    let q = QuatF64::from_mat3(&pose.rotation).to_array();
    (
        [t[0], t[1], t[2]],
        [q[3] as f32, q[0] as f32, q[1] as f32, q[2] as f32],
    )
}

/// Log colored vertices to the rerun viewer and draw each registered camera.
///
/// Requires the rerun viewer (`pip install rerun-sdk` or from rerun.io). The
/// viewer is spawned detached; this returns immediately after logging.
///
/// # Arguments
///
/// * `vertices` - Point-cloud vertices (positions, colours, normals).
/// * `views` - Per-frame poses from the reconstruction (`None` = unregistered).
/// * `fx` / `fy` / `cx` / `cy` - Camera intrinsics (pixels).
/// * `width` / `height` - Frame size in pixels (for the pinhole frustum).
///
/// # Errors
///
/// Returns an error if rerun cannot spawn a viewer.
#[allow(clippy::too_many_arguments)] // mirrors the reconstruction's inputs
pub fn view_world(
    vertices: &[Vertex],
    views: &[Option<Pose3d>],
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    width: usize,
    height: usize,
) -> Result<(), Box<dyn Error>> {
    eprintln!("[view] {} points", vertices.len());

    // Up-to-scale maps can be tiny: the orbit radius may be ~0.2 units while
    // the pinhole frustums are ~1 unit long, which makes every camera cluster
    // at the center visually. Normalize the display so the largest camera
    // radius maps to a fixed target (rotation unchanged).
    const TARGET_ORBIT_RADIUS: f64 = 5.0;
    let radii: Vec<f64> = views
        .iter()
        .filter_map(|v| v.as_ref())
        .map(|p| p.translation.length())
        .collect();
    let max_radius = radii.iter().cloned().fold(0.0, f64::max);
    let max_extent = vertices
        .iter()
        .map(|v| {
            let p = v.position;
            (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt()
        })
        .fold(0.0, f64::max);
    let scale = if max_radius > 1e-9 {
        TARGET_ORBIT_RADIUS / max_radius
    } else if max_extent > 1e-9 {
        1.0 / max_extent
    } else {
        1.0
    };

    let rec = rerun::RecordingStreamBuilder::new("SfM Point Cloud Viewer").spawn()?;
    rec.log("/", &rerun::ViewCoordinates::RIGHT_HAND_Y_DOWN())?;

    let points: Vec<rerun::Position3D> = vertices
        .iter()
        .map(|v| {
            let p = v.position;
            rerun::Position3D::new(
                (p[0] * scale) as f32,
                (p[1] * scale) as f32,
                (p[2] * scale) as f32,
            )
        })
        .collect();

    let colors: Vec<rerun::Color> = vertices
        .iter()
        .map(|v| rerun::Color::from_rgb(v.color[0], v.color[1], v.color[2]))
        .collect();

    rec.log(
        "world/points",
        &rerun::Points3D::new(points).with_colors(colors),
    )?;

    // Camera frustums: one entity per registered view (translation scaled,
    // rotation unchanged).
    let mut n_cameras = 0;
    for (i, view) in views.iter().enumerate() {
        let Some(pose) = view else { continue };
        let (t, q) = pose_to_rerun(pose);
        let t = [
            t[0] * scale as f32,
            t[1] * scale as f32,
            t[2] * scale as f32,
        ];
        rec.log(
            format!("world/camera_{i}"),
            &rerun::Transform3D::from_translation_rotation(
                t,
                rerun::Quaternion::from_wxyz([q[0], q[1], q[2], q[3]]),
            )
            .with_relation(CAMERA_TRANSFORM_RELATION),
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
    if !radii.is_empty() {
        let min = radii.iter().cloned().fold(f64::INFINITY, f64::min);
        eprintln!(
            "[view] camera centers: orbit radius min {min:.3} / max {max_radius:.3} -> scaled \
             {:.2}..{:.2}; display scale {scale:.2}",
            min * scale,
            max_radius * scale
        );
    }
    eprintln!(
        "[view] logged {} cameras and {} points.",
        n_cameras,
        vertices.len()
    );

    // The viewer is spawned detached; returning drops the `RecordingStream`,
    // which flushes and disconnects. No need to keep the process alive.
    Ok(())
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
    fn pose_to_rerun_passes_camera_to_world_through() {
        // C2W pose with a 90° yaw and translation (-5,0,0) (the camera centre).
        // ParentFromChild expects camera→world, so translation/rotation pass
        // through verbatim.
        let rot = Mat3F64::from_cols(
            Vec3F64::new(0.0, 0.0, -1.0),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(1.0, 0.0, 0.0),
        );
        let pose = Pose3d::new(rot, Vec3F64::new(-5.0, 0.0, 0.0));
        let (t, q) = pose_to_rerun(&pose);
        assert!((t[0] + 5.0).abs() < 1e-4, "t0={}", t[0]);
        assert!(t[1].abs() < 1e-4 && t[2].abs() < 1e-4, "t={t:?}");
        let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "unit quaternion, got {norm}");
    }

    #[test]
    fn camera_transform_relation_is_camera_to_world() {
        assert!(
            matches!(
                CAMERA_TRANSFORM_RELATION,
                rerun::TransformRelation::ParentFromChild
            ),
            "C2W poses must be logged with ParentFromChild, not ChildFromParent"
        );
    }
}
