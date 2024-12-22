use argh::FromArgs;
use std::path::PathBuf;

use kornia::k3d;

#[derive(FromArgs)]
/// Read a PLY file and log it to Rerun
struct Args {
    /// path to the PLY file
    #[argh(option)]
    ply_path: PathBuf,

    /// property type to read
    #[argh(option)]
    ply_type: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    let ply_type = match args.ply_type.to_lowercase().as_str() {
        "default" => k3d::io::ply::PlyType::XYZRgbNormals,
        "opensplat" => k3d::io::ply::PlyType::OpenSplat,
        _ => return Err(format!("Unsupported property: {}", args.ply_type).into()),
    };

    // read the image
    let pointcloud = k3d::io::ply::read_ply_binary(args.ply_path, ply_type)?;
    println!("Read #{} points", pointcloud.len());

    // create a Rerun recording stream
    let rec = rerun::RecordingStreamBuilder::new("Ply Visualizer").spawn()?;

    // create a vector of points
    let points = pointcloud
        .points()
        .iter()
        .map(|p| rerun::Position3D::new(p[0] as f32, p[1] as f32, p[2] as f32))
        .collect::<Vec<_>>();

    // create a vector of colors
    let colors = pointcloud.colors().map_or(vec![], |colors| {
        colors
            .iter()
            .map(|c| rerun::Color::from_rgb(c[0], c[1], c[2]))
            .collect()
    });

    // log the pointcloud
    rec.log(
        "pointcloud",
        &rerun::Points3D::new(points).with_colors(colors),
    )?;

    Ok(())
}
