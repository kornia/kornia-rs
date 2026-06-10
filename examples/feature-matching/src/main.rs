//! Classical feature-matching + homography demo — the OpenCV "find a known
//! object in a scene" tutorial, in Rust with kornia and XFeat features.
//!
//! Pipeline (mirrors OpenCV's *Features2D + Homography to find a known object*
//! tutorial; see [`pipeline`] for the full citation and parameters):
//!
//! 1. XFeat keypoints + descriptors on a reference image and on each scene
//!    image (a live webcam frame, or a static image in offline mode).
//! 2. Mutual nearest-neighbour + cosine-similarity-gate (0.82, the upstream
//!    XFeat convention) descriptor matching.
//! 3. RANSAC homography (3 px reprojection threshold) reference → scene.
//! 4. Project the reference image's four corners through H and draw the
//!    resulting quad on the scene to outline the found object.
//!
//! ## Usage
//!
//! Offline (headless, default — matches the two committed XFeat fixtures, two
//! views of the same scene):
//!
//! ```text
//! cargo run --release -p feature-matching
//! cargo run --release -p feature-matching -- \
//!     --reference path/to/book_cover.png --image path/to/scene.png
//! ```
//!
//! Live webcam (Linux + V4L only):
//!
//! ```text
//! cargo run --release -p feature-matching -- --reference path/to/book_cover.png --webcam
//! ```

mod pipeline;
mod preprocess;

use std::path::PathBuf;

use argh::FromArgs;

use kornia::io::functional::read_image_any_rgb8;

use pipeline::{estimate_homography, extract_features, match_features, Features, HomographyResult};
use preprocess::{rgb8_to_aligned_gray, AlignedGray};

/// Find a known planar object (reference image) in a scene using XFeat features,
/// descriptor matching, and a RANSAC homography. Offline by default; pass
/// `--webcam` for the live demo on Linux.
#[derive(FromArgs)]
struct Args {
    /// path to the reference (object) image. Defaults to the XFeat `ref`
    /// fixture so the example runs with no arguments.
    #[argh(option, short = 'r')]
    reference: Option<PathBuf>,

    /// path to a static scene image to match against (offline mode). Defaults
    /// to the XFeat `tgt` fixture. Ignored when `--webcam` is set.
    #[argh(option, short = 'i')]
    image: Option<PathBuf>,

    /// use the live webcam as the scene source instead of a static image
    /// (Linux + V4L only).
    #[argh(switch, short = 'w')]
    webcam: bool,

    /// webcam device id (`/dev/video<id>`), used with `--webcam`.
    #[argh(option, short = 'c', default = "0")]
    camera_id: u32,
}

/// Locate the committed XFeat fixture images so the example runs argument-free.
fn fixture(kind: &str) -> PathBuf {
    // examples/feature-matching/../../crates/kornia-xfeat/tests/fixtures/v1/<kind>/input.png
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .join("../../crates/kornia-xfeat/tests/fixtures/v1")
        .join(kind)
        .join("input.png")
}

/// Load an image file and preprocess it to an [`AlignedGray`] for XFeat.
fn load_aligned(path: &std::path::Path) -> Result<AlignedGray, Box<dyn std::error::Error>> {
    let rgb = read_image_any_rgb8(path)?;
    let h = rgb.height();
    let w = rgb.width();
    rgb8_to_aligned_gray(rgb.as_slice(), h, w)
        .ok_or_else(|| format!("image {} too small to align to 32", path.display()).into())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    let ref_path = args.reference.clone().unwrap_or_else(|| fixture("ref"));
    println!("Reference image: {}", ref_path.display());

    // ── Extract reference features once ──────────────────────────────────────
    let ref_gray = load_aligned(&ref_path)?;
    let ref_feats = extract_features(&ref_gray.data, ref_gray.height, ref_gray.width)?;
    println!(
        "Reference: {}x{} (aligned), {} keypoints",
        ref_gray.width,
        ref_gray.height,
        ref_feats.len()
    );

    if args.webcam {
        run_webcam(&args, &ref_feats, &ref_gray)
    } else {
        let scene_path = args.image.clone().unwrap_or_else(|| fixture("tgt"));
        run_offline(&scene_path, &ref_feats, &ref_gray)
    }
}

/// Pretty-print the full pipeline result for a single (reference, scene) pair.
fn report(
    ref_feats: &Features,
    scene_feats: &Features,
    matches: &[pipeline::FeatureMatch],
    homography: &Option<HomographyResult>,
) {
    if ref_feats.is_empty() || scene_feats.is_empty() {
        println!("No keypoints detected on one of the images; nothing to match.");
        return;
    }

    println!(
        "Scene: {} keypoints | matches (MNN + min_cossim {}): {}",
        scene_feats.len(),
        pipeline::MIN_COSSIM,
        matches.len()
    );

    match homography {
        None => {
            println!("Homography: FAILED (need >= 4 matches and a RANSAC consensus model)");
        }
        Some(h) => {
            // The inlier mask is aligned to `matches`; cross-check it against the
            // cached count and surface the first inlier correspondence as a spot
            // check that the mapping is sane.
            let mask_count = h.inliers.iter().filter(|&&b| b).count();
            debug_assert_eq!(mask_count, h.inlier_count);
            println!(
                "RANSAC: {} inliers / {} matches in {} iters",
                h.inlier_count,
                matches.len(),
                h.num_iters
            );
            if let Some(first) = h.inliers.iter().position(|&b| b) {
                let m = &matches[first];
                let r = &ref_feats.keypoints[m.ref_idx];
                let s = &scene_feats.keypoints[m.scene_idx];
                println!(
                    "  first inlier: ref ({:.1}, {:.1}) -> scene ({:.1}, {:.1})",
                    r.x, r.y, s.x, s.y
                );
            }
            println!("Homography H (reference -> scene), row-major:");
            for row in &h.h {
                println!("  [{:12.6} {:12.6} {:12.6}]", row[0], row[1], row[2]);
            }
            println!("Projected reference corners (scene pixels):");
            let names = [
                "top-left    ",
                "top-right   ",
                "bottom-right",
                "bottom-left ",
            ];
            for (name, c) in names.iter().zip(&h.projected_corners) {
                println!("  {name}: ({:9.2}, {:9.2})", c.x, c.y);
            }
        }
    }
}

/// Offline mode: match the reference against a single static scene image and
/// print every stage's numbers. Runs fully headless (no rerun, no camera).
fn run_offline(
    scene_path: &std::path::Path,
    ref_feats: &Features,
    ref_gray: &AlignedGray,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Scene image: {} (offline mode)", scene_path.display());

    let scene_gray = load_aligned(scene_path)?;
    let scene_feats = extract_features(&scene_gray.data, scene_gray.height, scene_gray.width)?;

    let matches = match_features(ref_feats, &scene_feats);
    let homography = estimate_homography(
        ref_feats,
        &scene_feats,
        &matches,
        ref_gray.width,
        ref_gray.height,
    );

    report(ref_feats, &scene_feats, &matches, &homography);
    Ok(())
}

// ── Live webcam mode (Linux + V4L) ───────────────────────────────────────────

#[cfg(target_os = "linux")]
fn run_webcam(
    args: &Args,
    ref_feats: &Features,
    ref_gray: &AlignedGray,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };

    use kornia::{
        image::{Image, ImageSize},
        imgproc::{self, color::YuvToRgbMode},
        io::{
            jpeg,
            v4l::{PixelFormat, V4LCameraConfig, V4lVideoCapture},
        },
        tensor::CpuAllocator,
    };

    use pipeline::Corner;

    let rec = rerun::RecordingStreamBuilder::new("kornia feature-matching").spawn()?;

    // Log the reference image once (left side of the view) as gray.
    let ref_u8: Vec<u8> = ref_gray
        .data
        .iter()
        .map(|&v| (v * 255.0).clamp(0.0, 255.0) as u8)
        .collect();
    rec.log(
        "reference",
        &rerun::Image::from_elements(
            &ref_u8,
            [ref_gray.width as u32, ref_gray.height as u32],
            rerun::ColorModel::L,
        ),
    )?;

    let webcam_size = ImageSize {
        width: 640,
        height: 480,
    };
    let mut webcam = V4lVideoCapture::new(V4LCameraConfig {
        device_path: format!("/dev/video{}", args.camera_id),
        size: webcam_size,
        ..Default::default()
    })?;

    let cancel = Arc::new(AtomicBool::new(false));
    {
        let cancel = Arc::clone(&cancel);
        if let Err(e) = ctrlc::set_handler(move || cancel.store(true, Ordering::SeqCst)) {
            eprintln!("warning: could not install Ctrl-C handler: {e}");
        }
    }

    let mut rgb_frame = Image::<u8, 3, _>::from_size_val(webcam_size, 0, CpuAllocator)?;

    println!("Webcam capture started ({}x{}); Ctrl-C to stop.", 640, 480);

    while !cancel.load(Ordering::SeqCst) {
        let Some(frame) = webcam.grab_frame()? else {
            continue;
        };
        let buf = frame.buffer.as_slice();
        match frame.pixel_format {
            PixelFormat::YUYV => {
                imgproc::color::convert_yuyv_to_rgb_u8(
                    buf,
                    &mut rgb_frame,
                    YuvToRgbMode::Bt601Full,
                )?;
            }
            PixelFormat::MJPG => {
                jpeg::decode_image_jpeg_rgb8(buf, &mut rgb_frame)?;
            }
            other => return Err(format!("unsupported pixel format: {other}").into()),
        }

        // Preprocess the frame and run the full pipeline.
        let Some(scene_gray) =
            rgb8_to_aligned_gray(rgb_frame.as_slice(), webcam_size.height, webcam_size.width)
        else {
            continue;
        };
        let scene_feats =
            match extract_features(&scene_gray.data, scene_gray.height, scene_gray.width) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("extract failed: {e}");
                    continue;
                }
            };
        let matches = match_features(ref_feats, &scene_feats);
        let homography = estimate_homography(
            ref_feats,
            &scene_feats,
            &matches,
            ref_gray.width,
            ref_gray.height,
        );

        // Log the scene frame as gray (aligned resolution, matching keypoint coords).
        let scene_u8: Vec<u8> = scene_gray
            .data
            .iter()
            .map(|&v| (v * 255.0).clamp(0.0, 255.0) as u8)
            .collect();
        rec.log(
            "scene",
            &rerun::Image::from_elements(
                &scene_u8,
                [scene_gray.width as u32, scene_gray.height as u32],
                rerun::ColorModel::L,
            ),
        )?;

        // Draw matched scene keypoints.
        let scene_pts: Vec<[f32; 2]> = matches
            .iter()
            .map(|m| {
                let k = &scene_feats.keypoints[m.scene_idx];
                [k.x, k.y]
            })
            .collect();
        rec.log("scene/matches", &rerun::Points2D::new(scene_pts))?;

        // Draw the projected object outline as a closed polygon (TL,TR,BR,BL,TL).
        if let Some(h) = &homography {
            let c = |p: &Corner| [p.x as f32, p.y as f32];
            let poly = vec![
                c(&h.projected_corners[0]),
                c(&h.projected_corners[1]),
                c(&h.projected_corners[2]),
                c(&h.projected_corners[3]),
                c(&h.projected_corners[0]),
            ];
            rec.log(
                "scene/object_outline",
                &rerun::LineStrips2D::new([poly]).with_colors([rerun::Color::from_rgb(0, 255, 0)]),
            )?;
            println!(
                "matches: {:4} | inliers: {:4} | iters: {:4}",
                matches.len(),
                h.inlier_count,
                h.num_iters
            );
        } else {
            rec.log("scene/object_outline", &rerun::Clear::flat())?;
            println!("matches: {:4} | object not found", matches.len());
        }
    }

    println!("Webcam capture stopped.");
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn run_webcam(
    _args: &Args,
    _ref_feats: &Features,
    _ref_gray: &AlignedGray,
) -> Result<(), Box<dyn std::error::Error>> {
    Err("the --webcam mode requires Video4Linux and is only supported on Linux".into())
}
