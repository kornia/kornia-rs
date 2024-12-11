use argh::FromArgs;
use std::path::PathBuf;

use kornia::k3d;
use kornia::k3d::pointcloud::PointCloud;
use kornia_icp as kicp;

#[derive(FromArgs)]
/// Example of ICP registration
struct Args {
    /// path to the source point cloud
    #[argh(option)]
    source_path: PathBuf,

    /// path to the target point cloud
    #[argh(option)]
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
    let rec = rerun::RecordingStreamBuilder::new("Ply Visualizer").spawn()?;

    //log_pointcloud(&rec, "source", &source_cloud)?;
    log_pointcloud(&rec, "target", &target_cloud, "marine")?;

    // NOTE: ICP Vanilla needs a good initial guess for the transformation
    let initial_rot = [
        [0.862, 0.011, -0.507],
        [-0.139, 0.967, -0.215],
        [0.487, 0.255, 0.835],
    ];
    let initial_trans = [0.5, 0.7, -1.4];

    let result = kicp::icp_vanilla(
        &source_cloud,
        &target_cloud,
        2000,
        1e-6,
        initial_rot,
        initial_trans,
    )?;
    println!("ICP registration result: {:?}", result);

    let mut transformed_source = vec![[0.0; 3]; source_cloud.len()];
    k3d::linalg::transform_points3d(
        source_cloud.points(),
        &result.rotation,
        &result.translation,
        &mut transformed_source,
    );

    let transformed_source = PointCloud::new(
        transformed_source,
        Some(source_cloud.colors().unwrap().to_vec()),
        None,
    );

    log_pointcloud(&rec, "transformed_source", &transformed_source, "gold")?;

    Ok(())
}

fn log_pointcloud(
    rec: &rerun::RecordingStream,
    name: &str,
    pointcloud: &PointCloud,
    color: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let points = pointcloud
        .points()
        .iter()
        .map(|p| rerun::Position3D::new(p[0] as f32, p[1] as f32, p[2] as f32))
        .collect::<Vec<_>>();

    let color = match color {
        "marine" => rerun::Color::from_rgb(90, 145, 199),
        "gold" => rerun::Color::from_rgb(255, 215, 0),
        _ => rerun::Color::from_rgb(255, 255, 255),
    };
    let colors = vec![color; points.len()];

    rec.log(name, &rerun::Points3D::new(points).with_colors(colors))?;

    Ok(())
}
