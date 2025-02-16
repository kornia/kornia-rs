use argh::FromArgs;
use std::path::PathBuf;

use kornia::k3d;

#[derive(FromArgs)]
/// Read a COLMAP database and log it to Rerun
struct Args {
    /// path to the COLMAP database
    #[argh(option)]
    colmap_path: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    // read the cameras
    let cameras = k3d::io::colmap::read_cameras_txt(args.colmap_path.join("cameras.txt"))?;

    // read the 3D points
    let colmap_points3d =
        k3d::io::colmap::read_points3d_txt(args.colmap_path.join("points3D.txt"))?;

    // read the images
    let colmap_images = k3d::io::colmap::read_images_txt(args.colmap_path.join("images.txt"))?;

    // create a Rerun recording stream
    let rec = rerun::RecordingStreamBuilder::new("Ply Visualizer").spawn()?;

    rec.log("/", &rerun::ViewCoordinates::RIGHT_HAND_Y_DOWN())?;

    let (points, colors) = colmap_points3d
        .iter()
        .map(|point| {
            (
                [
                    point.xyz[0] as f32,
                    point.xyz[1] as f32,
                    point.xyz[2] as f32,
                ],
                rerun::Color::from_rgb(point.rgb[0], point.rgb[1], point.rgb[2]),
            )
        })
        .collect::<(Vec<_>, Vec<_>)>();

    rec.log("points", &rerun::Points3D::new(points).with_colors(colors))?;

    // log the image camera poses
    for (i, image) in colmap_images.iter().enumerate() {
        rec.log(
            format!("camera_{}", i),
            &rerun::Transform3D::from_translation_rotation(
                image.translation.map(|x| x as f32),
                rerun::Quaternion::from_wxyz(image.rotation.map(|x| x as f32)),
            )
            .with_relation(rerun::TransformRelation::ChildFromParent),
        )?;

        rec.log(format!("camera_{}", i), &rerun::ViewCoordinates::RDF())?;

        let camera = cameras
            .iter()
            .find(|c| c.camera_id == image.camera_id)
            .unwrap_or_else(|| {
                panic!("Camera with id {} not found", image.camera_id);
            });

        rec.log(
            format!("camera_{}/image", i),
            &rerun::Pinhole::from_focal_length_and_resolution(
                [camera.params[0] as f32, camera.params[1] as f32],
                [camera.width as f32, camera.height as f32],
            )
            .with_principal_point([camera.params[2] as f32, camera.params[3] as f32]),
        )?;
    }

    Ok(())
}
