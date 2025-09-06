use argh::FromArgs;
use std::path::PathBuf;

use kornia::k3d::pointcloud::PointCloud;
use kornia::k3d::voxel_grid::VoxelGrid;
use glam::DVec3;
use rerun::{RecordingStreamBuilder, Points3D, Color};

#[derive(FromArgs)]
/// Example of point cloud downsampling using VoxelGrid
struct Args {
    /// path to the input point cloud (.pcd file)
    #[argh(option, short = 'i')]
    input_path: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let args: Args = argh::from_env();

    // Load the point cloud
    let point_cloud = kornia_3d::io::pcd::read_pcd_binary(args.input_path)?;
    println!("Original point cloud: #{} points", point_cloud.len());

    // Create a Rerun recording stream
    let rec = RecordingStreamBuilder::new("VoxelGrid Downsampling").spawn()?;

    // Downsample using VoxelGrid
    let leaf_size = DVec3::from_array([0.1, 0.1, 0.1]); // Adjust leaf size as needed
    let mut voxel_grid = VoxelGrid::new(leaf_size)?;
    voxel_grid.set_downsample_all_data(true);
    let downsampled_cloud = voxel_grid.downsample(&point_cloud);
    println!("Downsampled point cloud: #{} points", downsampled_cloud.len());

    // Log both point clouds
    log_pointcloud(&rec, &point_cloud, &downsampled_cloud)?;

    Ok(())
}

fn log_pointcloud(
    rec: &rerun::RecordingStream,
    original_cloud: &PointCloud,
    downsampled_cloud: &PointCloud,
) -> Result<(), Box<dyn std::error::Error>> {
    // Convert original points to Rerun format
    let points_original = original_cloud
        .points()
        .iter()
        .map(|p| rerun::Position3D::new(p[0] as f32, p[1] as f32, p[2] as f32))
        .collect::<Vec<_>>();

    // Convert downsampled points to Rerun format
    let points_downsampled = downsampled_cloud
        .points()
        .iter()
        .map(|p| rerun::Position3D::new(p[0] as f32, p[1] as f32, p[2] as f32))
        .collect::<Vec<_>>();

    // Use original colors if available, otherwise default to blue for original
    let colors_original = if let Some(colors) = original_cloud.colors() {
        colors
            .iter()
            .map(|c| Color::from_rgb(c[0], c[1], c[2]))
            .collect::<Vec<_>>()
    } else {
        vec![Color::from_rgb(90, 145, 199); points_original.len()] // Blue marine
    };

    // Use downsampled colors if available, otherwise default to green
    let colors_downsampled = if let Some(colors) = downsampled_cloud.colors() {
        colors
            .iter()
            .map(|c| Color::from_rgb(c[0], c[1], c[2]))
            .collect::<Vec<_>>()
    } else {
        vec![Color::from_rgb(0, 255, 0); points_downsampled.len()] // Green
    };

    // Log original point cloud
    rec.log(
        "original",
        &Points3D::new(points_original).with_colors(colors_original),
    )?;

    // Log downsampled point cloud
    rec.log(
        "downsampled",
        &Points3D::new(points_downsampled).with_colors(colors_downsampled),
    )?;

    Ok(())
}