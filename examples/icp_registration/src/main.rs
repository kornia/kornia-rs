use argh::FromArgs;
use std::path::PathBuf;

use kornia::k3d;
use kornia::k3d::pointcloud::PointCloud;
use kornia_icp as kicp;

#[derive(FromArgs)]
/// Example of ICP registration
struct Args {
    /// path to the source point cloud
    #[argh(option, short = 's')]
    source_path: PathBuf,

    /// path to the target point cloud
    #[argh(option, short = 't')]
    target_path: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let args: Args = argh::from_env();

    let source_cloud = k3d::io::pcd::read_pcd_binary(args.source_path)?;
    println!("Source cloud: #{} points", source_cloud.len());

    let target_cloud = k3d::io::pcd::read_pcd_binary(args.target_path)?;
    println!("Target cloud: #{} points", target_cloud.len());

    // create a Rerun recording stream
    let rec = rerun::RecordingStreamBuilder::new("ICP Visualizer").spawn()?;

    // NOTE: ICP Vanilla needs a good initial guess for the transformation
    let initial_rot = [
        [0.862, 0.011, -0.507],
        [-0.139, 0.967, -0.215],
        [0.487, 0.255, 0.835],
    ];
    let initial_trans = [0.5, 0.7, -1.20];

    let result = kicp::icp_vanilla(
        &source_cloud,
        &target_cloud,
        initial_rot,
        initial_trans,
        kicp::ICPConvergenceCriteria {
            max_iterations: 2000,
            tolerance: 1e-6,
        },
    )?;

    println!("ICP registration result: {:?}", result);

    // log the final transformation point cloud
    let mut transformed_source = vec![[0.0; 3]; source_cloud.points().len()];
    k3d::linalg::transform_points3d(
        source_cloud.points(),
        &result.rotation,
        &result.translation,
        &mut transformed_source,
    )?;

    let source_cloud_transformed = PointCloud::new(
        transformed_source,
        source_cloud.colors().cloned(),
        source_cloud.normals().cloned(),
    );

    log_pointcloud(&rec, &source_cloud_transformed, &target_cloud)?;

    Ok(())
}

fn log_pointcloud(
    rec: &rerun::RecordingStream,
    source_cloud: &PointCloud,
    target_cloud: &PointCloud,
) -> Result<(), Box<dyn std::error::Error>> {
    let points_source = source_cloud
        .points()
        .iter()
        .map(|p| rerun::Position3D::new(p[0] as f32, p[1] as f32, p[2] as f32))
        .collect::<Vec<_>>();

    let points_target = target_cloud
        .points()
        .iter()
        .map(|p| rerun::Position3D::new(p[0] as f32, p[1] as f32, p[2] as f32))
        .collect::<Vec<_>>();

    let colors_target = target_cloud
        .colors()
        .unwrap()
        .iter()
        .map(|c| rerun::Color::from_rgb(c[0], c[1], c[2]))
        .collect::<Vec<_>>();

    // to log the source point cloud in blue
    let color_source = rerun::Color::from_rgb(90, 145, 199); // blue marine
    let colors_source = vec![color_source; points_source.len()];

    // to log the source point cloud in original color
    let colors_source_rgb = source_cloud
        .colors()
        .unwrap()
        .iter()
        .map(|c| rerun::Color::from_rgb(c[0], c[1], c[2]))
        .collect::<Vec<_>>();

    rec.log(
        "source",
        &rerun::Points3D::new(points_source.clone()).with_colors(colors_source),
    )?;

    rec.log(
        "source_rgb",
        &rerun::Points3D::new(points_source.clone()).with_colors(colors_source_rgb),
    )?;

    rec.log(
        "target",
        &rerun::Points3D::new(points_target).with_colors(colors_target),
    )?;

    Ok(())
}
