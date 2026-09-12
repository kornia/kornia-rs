//! End-to-end Structure-from-Motion from a video file.
//!
//! Pipeline: read the MP4 into RGB + grayscale frames, extract features
//! (ORB or SIFT), match frames in a sliding window, chain the matches into
//! tracks via `kornia_calib::build_tracks`, reconstruct the scene with
//! `kornia_calib::reconstruct`, and export the point cloud (XYZ + RGB +
//! normals) to a binary PLY file.

mod features;
mod matching;
mod ply_writer;
mod reconstruction;
mod video;

#[cfg(test)]
mod test_util;

use std::error::Error;
use std::path::PathBuf;

use argh::FromArgs;
use kornia_calib::build_tracks;

#[derive(FromArgs)]
/// Structure-from-Motion from a video file to a point-cloud PLY.
struct Args {
    /// path to the input video file
    #[argh(positional)]
    video: PathBuf,

    /// path to the output PLY file
    #[argh(positional)]
    output: PathBuf,

    /// feature detector: "orb" or "sift"
    #[argh(option, default = "features::DetectorKind::Orb")]
    detector: features::DetectorKind,

    /// focal length X in pixels
    #[argh(option)]
    fx: f64,

    /// focal length Y in pixels
    #[argh(option)]
    fy: f64,

    /// principal point X in pixels
    #[argh(option)]
    cx: f64,

    /// principal point Y in pixels
    #[argh(option)]
    cy: f64,

    /// max features per frame
    #[argh(option, default = "2000")]
    n_features: usize,

    /// match each frame against this many following frames
    #[argh(option, default = "5")]
    match_window: usize,

    /// lowe's ratio test threshold (lower = stricter)
    #[argh(option, default = "0.8")]
    ratio: f32,

    /// process every Nth frame (1 = all frames)
    #[argh(option, default = "1")]
    frame_step: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Args = argh::from_env();

    // 1. Decode frames.
    eprintln!("Reading video: {}", args.video.display());
    let (rgb_frames, gray_frames) = video::read_frames(&args.video, args.frame_step)?;
    eprintln!("Decoded {} frames", gray_frames.len());
    if gray_frames.len() < 2 {
        return Err("need at least two frames to reconstruct".into());
    }

    // 2. Extract features per frame.
    eprintln!("Extracting features with {:?}", args.detector);
    let extractor = features::make_extractor(args.detector, args.n_features);
    let all_features: Vec<features::FrameFeatures> = gray_frames
        .iter()
        .map(|frame| extractor.extract(frame))
        .collect::<Result<_, _>>()?;
    let total_keypoints: usize = all_features.iter().map(|f| f.n_keypoints()).sum();
    eprintln!(
        "Extracted {total_keypoints} keypoints across {} frames",
        all_features.len()
    );

    // 3. Match frames in a sliding window.
    let edges = matching::match_sequential_pairs(&all_features, args.match_window, args.ratio);
    eprintln!("Found {} matched correspondences", edges.len());

    // 4. Chain matches into multi-view tracks.
    let tracks = build_tracks(&edges);
    eprintln!("Built {} tracks", tracks.len());

    // 5. Reconstruct the scene.
    let n_frames = gray_frames.len();
    let reconstruction =
        reconstruction::run_sfm(&tracks, args.fx, args.fy, args.cx, args.cy, n_frames)?;
    let registered = reconstruction.views.iter().filter(|v| v.is_some()).count();
    eprintln!(
        "Reconstructed {} points across {} registered views (scale: {:?})",
        reconstruction.points.len(),
        registered,
        reconstruction.scale,
    );

    // 6. Export the point cloud to PLY (XYZ + RGB + normals).
    let vertices = ply_writer::build_vertices(&reconstruction, &tracks, &rgb_frames);
    ply_writer::write_ply(&args.output, &vertices)?;
    eprintln!(
        "Wrote {} vertices to {}",
        vertices.len(),
        args.output.display()
    );

    Ok(())
}
