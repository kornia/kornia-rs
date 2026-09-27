//! End-to-end Structure-from-Motion from a video file.
//!
//! Pipeline: read the MP4 into RGB + grayscale frames, extract features
//! (ORB or SIFT), match frames in a sliding window, chain the matches into
//! tracks via `kornia_calib::build_tracks`, reconstruct the scene with
//! `kornia_calib::reconstruct`, and export the point cloud (XYZ + RGB +
//! normals) to a binary PLY file.
//!
//! Every pipeline stage logs its start, progress, and completion time to
//! stderr so slow stages can be identified.

mod features;
mod matching;
mod ply_viewer;
mod ply_writer;
mod reconstruction;
mod video;

#[cfg(test)]
mod test_util;

use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

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

    /// read the video asynchronously (overlaps decode with downstream work)
    #[argh(switch)]
    async_video: bool,

    /// worker threads for parallel feature extraction/matching (0 = auto)
    #[argh(option, default = "0")]
    threads: usize,

    /// channel buffer size for async video reading
    #[argh(option, default = "32")]
    buffer_size: usize,

    /// open the output PLY in the rerun viewer after writing
    #[argh(switch)]
    view: bool,

    /// disable ORB orientation-histogram filtering (helps for orbit captures)
    #[argh(switch)]
    orb_no_orientation_check: bool,

    /// max bundle adjustment iterations (default: 100)
    #[argh(option, default = "100")]
    max_ba_iterations: usize,

    /// min PnP inliers to register a view (default: 30)
    #[argh(option, default = "30")]
    min_registration_inliers: usize,

    /// motion prior sigma (0.0 = disabled)
    #[argh(option, default = "0.0")]
    motion_prior_sigma: f64,

    /// up prior sigma (0.0 = disabled)
    #[argh(option, default = "0.0")]
    up_prior_sigma: f64,

    /// max reprojection error in normalized units (default: 0.01)
    #[argh(option, default = "0.01")]
    max_reprojection_error: f64,

    /// verify matches with epipolar RANSAC (rejects false matches)
    #[argh(switch)]
    geo_verify: bool,

    /// epipolar RANSAC inlier threshold in pixels (default: 3.0)
    #[argh(option, default = "3.0")]
    geo_threshold: f64,

    /// min inliers for a pair's fundamental matrix to be trusted (default: 8)
    #[argh(option, default = "8")]
    geo_min_inliers: usize,

    /// use CUDA for SIFT extraction (requires an NVIDIA GPU)
    #[argh(switch)]
    cuda: bool,

    /// enable Wald's SPRT for PnP registration (rejects bad poses early)
    #[argh(switch)]
    sprt: bool,

    /// SPRT expected inlier ratio (default: 0.5)
    #[argh(option, default = "0.5")]
    sprt_epsilon: f64,

    /// SPRT Type-I error delta, probability of rejecting a good pose (default: 0.05)
    #[argh(option, default = "0.05")]
    sprt_delta: f64,

    /// refine focal length + radial/tangential distortion against the
    /// reconstruction (helps when intrinsics are guessed)
    #[argh(switch)]
    refine_intrinsics: bool,

    /// also match frame i against i+K, i+2K, ... (wide baseline, longer
    /// tracks, more registered views; 0 = off)
    #[argh(option, default = "0")]
    wide_baseline: usize,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args: Args = argh::from_env();

    // Configure the rayon thread pool (0 = auto-detect).
    features::configure_thread_pool(args.threads)?;

    // 1. Decode frames (async or sync).
    if args.async_video {
        eprintln!(
            "[1/6] reading video (async, buffer={}): {}",
            args.buffer_size,
            args.video.display()
        );
    } else {
        eprintln!("[1/6] reading video (sync): {}", args.video.display());
    }
    let t = Instant::now();
    let (rgb_frames, gray_frames) = if args.async_video {
        video::read_frames_async(&args.video, args.frame_step, args.buffer_size)
            .await
            .map_err(|e| -> Box<dyn Error> { e })?
    } else {
        video::read_frames(&args.video, args.frame_step)?
    };
    eprintln!(
        "[1/6] decoded {} frames in {:.1}s",
        gray_frames.len(),
        t.elapsed().as_secs_f64()
    );
    if gray_frames.len() < 2 {
        return Err("need at least two frames to reconstruct".into());
    }
    if gray_frames.len() > 200 {
        eprintln!(
            "  warning: {} frames may make reconstruction slow; consider a larger --frame-step",
            gray_frames.len()
        );
    }

    // 2. Extract features per frame (parallel via rayon; sequential for CUDA).
    if args.async_video {
        eprintln!(
            "[2/6] extracting features with {:?} (parallel)",
            args.detector
        );
    } else {
        eprintln!("[2/6] extracting features with {:?}", args.detector);
    }
    if args.cuda {
        eprintln!("[2/6] CUDA SIFT extraction (device 0)");
    }
    let t = Instant::now();
    let extractor = features::make_extractor(args.detector, args.n_features, args.cuda)?;
    // CUDA shares one device stream, so extraction must be sequential.
    let all_features = if args.cuda {
        gray_frames
            .iter()
            .map(|frame| extractor.extract(frame))
            .collect::<Result<_, _>>()?
    } else if args.async_video {
        features::extract_features_parallel(&gray_frames, extractor.as_ref())
            .map_err(|e| -> Box<dyn Error> { e.into() })?
    } else {
        gray_frames
            .iter()
            .map(|frame| extractor.extract(frame))
            .collect::<Result<_, _>>()?
    };
    let total_keypoints: usize = all_features.iter().map(|f| f.n_keypoints()).sum();
    eprintln!(
        "[2/6] extracted {total_keypoints} keypoints across {} frames in {:.1}s",
        all_features.len(),
        t.elapsed().as_secs_f64()
    );

    // 3. Match frames in a sliding window (parallel in async mode).
    if args.async_video {
        eprintln!(
            "[3/6] matching frames (window={}, parallel)",
            args.match_window
        );
    } else {
        eprintln!("[3/6] matching frames (window={})", args.match_window);
    }
    let t = Instant::now();
    let mut edges = if args.async_video {
        matching::match_pairs_parallel(
            &all_features,
            args.match_window,
            args.ratio,
            !args.orb_no_orientation_check,
        )
    } else {
        matching::match_sequential_pairs(
            &all_features,
            args.match_window,
            args.ratio,
            !args.orb_no_orientation_check,
        )
    };
    eprintln!(
        "[3/6] found {} matched correspondences in {:.1}s",
        edges.len(),
        t.elapsed().as_secs_f64()
    );

    // 3.25. Optional: wide-baseline matching (long jumps beyond the window).
    if args.wide_baseline > 0 {
        eprintln!(
            "[3/6] wide-baseline matching (stride={})",
            args.wide_baseline
        );
        let t = Instant::now();
        let added = matching::append_wide_baseline_edges(
            &all_features,
            args.match_window,
            args.wide_baseline,
            args.ratio,
            !args.orb_no_orientation_check,
            &mut edges,
        );
        eprintln!(
            "[3/6] added {added} wide-baseline matches in {:.1}s",
            t.elapsed().as_secs_f64()
        );
    }

    // 3.5. Optional: geometric verification (epipolar RANSAC) to reject false matches.
    let edges = if args.geo_verify {
        eprintln!(
            "[3.5/6] geometric verification (threshold={} px, min_inliers={})",
            args.geo_threshold, args.geo_min_inliers
        );
        let t = Instant::now();
        let before = edges.len();
        let edges = matching::verify_matches_geometrically(
            &edges,
            args.geo_threshold,
            args.geo_min_inliers,
        );
        eprintln!(
            "[3.5/6] filtered {before} -> {} matches in {:.1}s",
            edges.len(),
            t.elapsed().as_secs_f64()
        );
        edges
    } else {
        edges
    };

    // 4. Chain matches into multi-view tracks.
    eprintln!("[4/6] building tracks");
    let t = Instant::now();
    let tracks = build_tracks(&edges);
    eprintln!(
        "[4/6] built {} tracks in {:.1}s",
        tracks.len(),
        t.elapsed().as_secs_f64()
    );

    // 5. Reconstruct the scene.
    eprintln!("[5/6] reconstructing scene");
    let t = Instant::now();
    let n_frames = gray_frames.len();
    let progress_start = Instant::now();
    let progress: Arc<dyn Fn(usize, usize) + Send + Sync> = Arc::new(move |registered, n_cams| {
        eprintln!(
            "  reconstruct: {registered}/{n_cams} views registered ({:.1}s)",
            progress_start.elapsed().as_secs_f64()
        );
    });
    let overrides = reconstruction::ReconstructionOverrides {
        max_iterations: Some(args.max_ba_iterations),
        min_registration_inliers: Some(args.min_registration_inliers),
        motion_prior_sigma: Some(args.motion_prior_sigma),
        up_prior_sigma: Some(args.up_prior_sigma),
        max_reprojection_error: Some(args.max_reprojection_error),
        sprt: args.sprt.then(|| kornia_3d::ransac::SPRTConfig {
            epsilon: args.sprt_epsilon,
            delta: args.sprt_delta,
            ..Default::default()
        }),
        refine_intrinsics: args.refine_intrinsics.then_some(true),
    };
    let reconstruction = reconstruction::run_sfm(
        &tracks,
        args.fx,
        args.fy,
        args.cx,
        args.cy,
        n_frames,
        Some(progress),
        overrides,
    )?;
    let registered = reconstruction.views.iter().filter(|v| v.is_some()).count();
    eprintln!(
        "[5/6] reconstructed {} points across {} registered views (scale: {:?}, rmse {:.3} px) in {:.1}s",
        reconstruction.points.len(),
        registered,
        reconstruction.scale,
        reconstruction.reproj_rmse_px,
        t.elapsed().as_secs_f64(),
    );

    // 6. Export the point cloud to PLY (XYZ + RGB + normals).
    eprintln!("[6/6] building vertices and writing PLY");
    let t = Instant::now();
    let vertices = ply_writer::build_vertices(&reconstruction, &tracks, &rgb_frames);
    ply_writer::write_ply(&args.output, &vertices)?;
    eprintln!(
        "[6/6] wrote {} vertices to {} in {:.1}s",
        vertices.len(),
        args.output.display(),
        t.elapsed().as_secs_f64(),
    );

    // 7. Optional: visualize the result (point cloud + camera poses) in rerun.
    if args.view {
        eprintln!("[7/7] opening PLY + camera poses in rerun viewer...");
        let (frame_w, frame_h) = rgb_frames
            .first()
            .map(|f| (f.width(), f.height()))
            .unwrap_or((0, 0));
        ply_viewer::view_world(
            &args.output,
            &reconstruction.views,
            args.fx,
            args.fy,
            args.cx,
            args.cy,
            frame_w,
            frame_h,
        )?;
    }

    Ok(())
}
