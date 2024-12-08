use argh::FromArgs;
use std::path::PathBuf;

use kornia::k3d;

#[derive(FromArgs)]
/// Read a PLY file and log it to Rerun
struct Args {
    /// path to the PLY file
    #[argh(option)]
    ply_path: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    // read the image
    let pointcloud =
        k3d::io::ply::read_ply_binary(args.ply_path, k3d::io::ply::PlyProperty::OpenSplat)?;
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
            .map(|c| {
                rerun::Color::from_rgb(
                    (c[0] * 255.0) as u8,
                    (c[1] * 255.0) as u8,
                    (c[2] * 255.0) as u8,
                )
            })
            .collect()
    });

    // log the pointcloud
    rec.log(
        "pointcloud",
        &rerun::Points3D::new(points).with_colors(colors),
    )?;

    Ok(())
}
