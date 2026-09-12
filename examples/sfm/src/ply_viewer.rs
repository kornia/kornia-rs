//! Visualize a PLY point cloud in the rerun GUI.
//!
//! Reads the file with `kornia::k3d::io::ply::read_ply_binary` (the same
//! reader the `ply_rerun` example uses) and logs positions + colors to a
//! spawned rerun viewer.

use std::error::Error;
use std::path::Path;

/// Open `path` in the rerun viewer and block until the viewer is closed.
///
/// Requires the rerun viewer (`pip install rerun-sdk` or from rerun.io).
///
/// # Arguments
///
/// * `path` - Path to a binary PLY file (XYZRgbNormals layout).
///
/// # Errors
///
/// Returns an error if the PLY cannot be read or rerun cannot spawn a viewer.
pub fn view_ply(path: &Path) -> Result<(), Box<dyn Error>> {
    let pointcloud =
        kornia::k3d::io::ply::read_ply_binary(path, kornia::k3d::io::ply::PlyType::XYZRgbNormals)?;
    eprintln!(
        "[view] read {} points from {}",
        pointcloud.len(),
        path.display()
    );

    let rec = rerun::RecordingStreamBuilder::new("SfM Point Cloud Viewer").spawn()?;

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
        "pointcloud",
        &rerun::Points3D::new(points).with_colors(colors),
    )?;

    eprintln!("[view] point cloud opened in rerun. Close the viewer to exit.");
    // Keep the process alive so the viewer stays connected.
    loop {
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}
