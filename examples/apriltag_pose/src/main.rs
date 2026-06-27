//! Detect AprilTag36H11 tags in an image and estimate the 6-DOF pose of each detection.

use argh::FromArgs;
use kornia::io::png::read_image_png_mono8;
use kornia_3d::camera::PinholeCamera;
use kornia_apriltag::{decoder::Detection, family::TagFamilyKind, AprilTagDecoder, DecodeTagsConfig};

#[derive(FromArgs)]
/// Detect AprilTag36H11 markers and estimate their 6-DOF poses.
struct Args {
    /// path to the input PNG image
    #[argh(option)]
    path: std::path::PathBuf,

    /// physical tag size in metres (e.g. 0.162)
    #[argh(option, default = "0.162")]
    tag_size: f64,

    /// camera focal length in x (pixels)
    #[argh(option, default = "600.0")]
    fx: f64,

    /// camera focal length in y (pixels)
    #[argh(option, default = "600.0")]
    fy: f64,

    /// principal point x (pixels)
    #[argh(option, default = "320.0")]
    cx: f64,

    /// principal point y (pixels)
    #[argh(option, default = "240.0")]
    cy: f64,

    /// number of orthogonal-iteration refinement steps
    #[argh(option, default = "50")]
    n_iters: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    let img = read_image_png_mono8(&args.path)?;
    let img_size = img.size();

    let config = DecodeTagsConfig::new(vec![TagFamilyKind::Tag36H11])?;
    let mut decoder = AprilTagDecoder::new(config, img_size)?;
    let detections = decoder.decode(&img)?;

    if detections.is_empty() {
        println!("No tags detected.");
        return Ok(());
    }

    let camera = PinholeCamera {
        fx: args.fx,
        fy: args.fy,
        cx: args.cx,
        cy: args.cy,
        k1: 0.0,
        k2: 0.0,
        p1: 0.0,
        p2: 0.0,
    };

    for det in &detections {
        print_detection(det, &camera, args.tag_size, args.n_iters);
    }

    Ok(())
}

fn print_detection(det: &Detection, camera: &PinholeCamera, tag_size: f64, n_iters: usize) {
    println!("--- Tag ID: {} (hamming: {}) ---", det.id, det.hamming);
    match det.estimate_pose(camera, tag_size, n_iters) {
        Ok(pair) => {
            let best = &pair.best;
            let r = &best.pose.rotation;
            let t = &best.pose.translation;
            println!(
                "  rotation:\n    [{:.4}, {:.4}, {:.4}]\n    [{:.4}, {:.4}, {:.4}]\n    [{:.4}, {:.4}, {:.4}]",
                r.x_axis.x, r.y_axis.x, r.z_axis.x,
                r.x_axis.y, r.y_axis.y, r.z_axis.y,
                r.x_axis.z, r.y_axis.z, r.z_axis.z,
            );
            println!("  translation: [{:.4}, {:.4}, {:.4}]", t.x, t.y, t.z);
            println!("  reprojection error (best):   {:.6}", best.error);
            println!("  reprojection error (second): {:.6}", pair.second.error);
        }
        Err(e) => println!("  pose estimation failed: {e}"),
    }
}
