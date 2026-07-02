use argh::FromArgs;
use kornia::{
    image::{Image, ImageSize},
    imgproc::color::{gray_from_rgb_u8, YuvToRgbMode},
    io::{fps_counter::FpsCounter, functional::read_image_any_rgb8, jpeg},
};
use kornia_3d::camera::PinholeCamera;
use kornia_apriltag::{family::TagFamilyKind, AprilTagDecoder, DecodeTagsConfig};
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

/// Per-tag temporal pose state (last smoothed orientation + translation), keyed by tag id.
type PoseState = HashMap<u16, (glam::DQuat, glam::DVec3)>;

/// Detect AprilTags in an image or live camera feed and visualize in Rerun (2D + optional 3D pose).
#[derive(Debug, FromArgs)]
struct Args {
    /// path to an image file (mutually exclusive with --camera-id)
    #[argh(option, short = 'p')]
    path: Option<String>,

    /// V4L camera device index (default 0; mutually exclusive with --path)
    #[argh(option, short = 'c')]
    camera_id: Option<u32>,

    /// camera capture width in pixels
    #[argh(option, default = "640")]
    width: u32,

    /// camera capture height in pixels
    #[argh(option, default = "480")]
    height: u32,

    /// frames per second for camera mode
    #[argh(option, short = 'f', default = "30")]
    fps: u32,

    /// apriltag family kinds to detect
    #[argh(
        option,
        short = 'k',
        default = "vec![TagFamilyKind::Tag36H11]",
        from_str_fn(parse_family)
    )]
    kind: Vec<TagFamilyKind>,

    /// downscale factor (1 = none, 2 = half, …)
    #[argh(option, short = 's', default = "2")]
    downscale_factor: usize,

    /// minimum white/black pixel difference for adaptive threshold
    #[argh(option, short = 'm', default = "5")]
    min_white_black_difference: u8,

    /// tag sharpening for decode (0.0–1.0)
    #[argh(option, short = 'd', default = "0.25")]
    decode_sharpening: f32,

    /// enable sub-pixel edge refinement
    #[argh(switch, short = 'r')]
    refine_edges_enabled: bool,

    // ── 3D pose estimation ──────────────────────────────────────────────────
    /// physical tag side length in metres (enables 3D pose output)
    #[argh(option)]
    tag_size: Option<f64>,

    /// camera focal length x in pixels
    #[argh(option, default = "600.0")]
    fx: f64,

    /// camera focal length y in pixels
    #[argh(option, default = "600.0")]
    fy: f64,

    /// camera principal point x in pixels (default: width/2)
    #[argh(option)]
    cx: Option<f64>,

    /// camera principal point y in pixels (default: height/2)
    #[argh(option)]
    cy: Option<f64>,

    /// number of orthogonal-iteration refinement steps for pose
    #[argh(option, default = "50")]
    n_iters: usize,

    /// pose temporal smoothing toward each new frame (0 = frozen, 1 = none/raw)
    #[argh(option, default = "0.35")]
    pose_smoothing: f64,
}

fn parse_family(s: &str) -> Result<TagFamilyKind, String> {
    match s {
        "tag16_h5" => Ok(TagFamilyKind::Tag16H5),
        "tag36_h11" => Ok(TagFamilyKind::Tag36H11),
        "tag36_h10" => Ok(TagFamilyKind::Tag36H10),
        "tag25_h9" => Ok(TagFamilyKind::Tag25H9),
        "tagcircle21_h7" => Ok(TagFamilyKind::TagCircle21H7),
        "tagcircle49_h12" => Ok(TagFamilyKind::TagCircle49H12),
        "tagcustom48_h12" => Ok(TagFamilyKind::TagCustom48H12),
        "tagstandard41_h12" => Ok(TagFamilyKind::TagStandard41H12),
        "tagstandard52_h13" => Ok(TagFamilyKind::TagStandard52H13),
        _ => Err(format!("Unknown family '{s}'")),
    }
}

fn tag_color(kind: TagFamilyKind) -> rerun::Color {
    let [r, g, b] = match kind {
        TagFamilyKind::Tag16H5 => [255, 0, 0],
        TagFamilyKind::Tag36H11 => [0, 255, 0],
        TagFamilyKind::Tag36H10 => [0, 0, 255],
        TagFamilyKind::Tag25H9 => [255, 255, 0],
        TagFamilyKind::TagCircle21H7 => [255, 0, 255],
        TagFamilyKind::TagCircle49H12 => [0, 255, 255],
        TagFamilyKind::TagCustom48H12 => [255, 128, 0],
        TagFamilyKind::TagStandard41H12 => [128, 0, 255],
        TagFamilyKind::TagStandard52H13 => [0, 128, 255],
        TagFamilyKind::Custom(_) => [128, 128, 128],
    };
    rerun::Color::from_rgb(r, g, b)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    let rec = rerun::RecordingStreamBuilder::new("Kornia AprilTag").spawn()?;

    // Camera-space convention: X-right, Y-down, Z-forward (OpenCV/RDF).
    rec.log_static("/", &rerun::ViewCoordinates::RIGHT_HAND_Y_DOWN())?;
    rec.log_static("world/camera", &rerun::ViewCoordinates::RDF())?;

    let mut config = DecodeTagsConfig::new(args.kind.clone())?;
    config.downscale_factor = args.downscale_factor;
    config.min_white_black_difference = args.min_white_black_difference;
    config.decode_sharpening = args.decode_sharpening;
    config.refine_edges_enabled = args.refine_edges_enabled;

    match (&args.path, args.camera_id) {
        (Some(path), None) => {
            // Static image mode.
            let rgb = read_image_any_rgb8(path)?;
            let mut gray = Image::from_size_val(rgb.size(), 0u8)?;
            gray_from_rgb_u8(&rgb, &mut gray)?;

            let cx = args.cx.unwrap_or(gray.width() as f64 / 2.0);
            let cy = args.cy.unwrap_or(gray.height() as f64 / 2.0);

            let mut decoder = AprilTagDecoder::new(config, gray.size())?;
            let detections = decoder.decode(&gray)?;
            eprintln!("{} detections in {}", detections.len(), path);

            let mut pose_state = PoseState::new();
            log_frame(
                &rec,
                &rgb,
                gray.size(),
                &detections,
                args.tag_size.map(|ts| {
                    (
                        PinholeCamera {
                            fx: args.fx,
                            fy: args.fy,
                            cx,
                            cy,
                            k1: 0.0,
                            k2: 0.0,
                            p1: 0.0,
                            p2: 0.0,
                        },
                        ts,
                        args.n_iters,
                    )
                }),
                &mut pose_state,
                args.pose_smoothing,
            )?;
        }
        (None, camera) => {
            // Live camera mode (V4L, Linux only).
            #[cfg(target_os = "linux")]
            {
                use kornia::io::v4l::{PixelFormat, V4LCameraConfig, V4lVideoCapture};

                let device_id = camera.unwrap_or(0);
                let requested_size = ImageSize {
                    width: args.width as usize,
                    height: args.height as usize,
                };

                let cancel = Arc::new(AtomicBool::new(false));
                let cancel2 = Arc::clone(&cancel);
                ctrlc::set_handler(move || {
                    cancel2.store(true, Ordering::SeqCst);
                })?;

                let mut cam = V4lVideoCapture::new(V4LCameraConfig {
                    device_path: format!("/dev/video{device_id}"),
                    size: requested_size,
                    fps: args.fps,
                    format: PixelFormat::MJPG,
                    buffer_size: 4,
                })?;

                // Use the size the camera actually negotiated (some devices clamp it).
                let img_size = cam.size();

                let cx = args.cx.unwrap_or(img_size.width as f64 / 2.0);
                let cy_val = args.cy.unwrap_or(img_size.height as f64 / 2.0);
                let pinhole = args.tag_size.map(|ts| {
                    (
                        PinholeCamera {
                            fx: args.fx,
                            fy: args.fy,
                            cx,
                            cy: cy_val,
                            k1: 0.0,
                            k2: 0.0,
                            p1: 0.0,
                            p2: 0.0,
                        },
                        ts,
                        args.n_iters,
                    )
                });

                let mut rgb = Image::<u8, 3>::from_size_val(img_size, 0)?;
                let mut gray = Image::<u8, 1>::from_size_val(img_size, 0u8)?;
                let mut decoder = AprilTagDecoder::new(config, img_size)?;
                let mut fps = FpsCounter::new();
                let mut pose_state = PoseState::new();

                while !cancel.load(Ordering::SeqCst) {
                    let Some(frame) = cam.grab_frame()? else {
                        continue;
                    };
                    // For most formats we decode to RGB and then derive gray below.
                    // GREY is already single-channel luminance, so we fill `gray`
                    // directly and replicate it into `rgb` only for visualization.
                    let mut have_gray = false;
                    match frame.pixel_format {
                        PixelFormat::YUYV => {
                            kornia::imgproc::color::convert_yuyv_to_rgb_u8(
                                frame.buffer.as_slice(),
                                &mut rgb,
                                YuvToRgbMode::Bt601Full,
                            )?;
                        }
                        PixelFormat::MJPG => {
                            jpeg::decode_image_jpeg_rgb8(frame.buffer.as_slice(), &mut rgb)?;
                        }
                        // V4L2 GREY (8-bit grayscale): not a named variant, arrives as Custom.
                        PixelFormat::Custom(fourcc) if &fourcc == b"GREY" => {
                            let buf = frame.buffer.as_slice();
                            let expected = img_size.width * img_size.height;
                            if buf.len() < expected {
                                continue;
                            }
                            let buf = &buf[..expected];
                            gray.as_slice_mut().copy_from_slice(buf);
                            for (px, &g) in rgb.as_slice_mut().chunks_exact_mut(3).zip(buf) {
                                px[0] = g;
                                px[1] = g;
                                px[2] = g;
                            }
                            have_gray = true;
                        }
                        _ => continue,
                    }
                    if !have_gray {
                        gray_from_rgb_u8(&rgb, &mut gray)?;
                    }
                    let detections = decoder.decode(&gray)?;
                    decoder.clear();

                    log_frame(
                        &rec,
                        &rgb,
                        img_size,
                        &detections,
                        pinhole
                            .as_ref()
                            .map(|(cam, ts, ni)| (cam.clone(), *ts, *ni)),
                        &mut pose_state,
                        args.pose_smoothing,
                    )?;

                    fps.update();
                    eprint!("\r{} detections  {:.1} fps", detections.len(), fps.fps());
                }
                eprintln!();
            }

            #[cfg(not(target_os = "linux"))]
            {
                let _ = camera;
                return Err("V4L camera mode is only supported on Linux".into());
            }
        }
        (Some(_), Some(_)) => {
            return Err("Provide either --path or --camera-id, not both".into());
        }
    }

    Ok(())
}

/// Log one frame: image + 2D overlays + optional 3D tag poses.
fn log_frame(
    rec: &rerun::RecordingStream,
    rgb: &kornia::image::Image<u8, 3>,
    img_size: ImageSize,
    detections: &[kornia_apriltag::decoder::Detection],
    pose_args: Option<(PinholeCamera, f64, usize)>,
    pose_state: &mut PoseState,
    smooth_alpha: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    // Log camera pinhole model once (static) when pose estimation is active.
    if let Some((ref cam, _, _)) = pose_args {
        rec.log_static(
            "world/camera/image",
            &rerun::Pinhole::from_focal_length_and_resolution(
                [cam.fx as f32, cam.fy as f32],
                [img_size.width as f32, img_size.height as f32],
            )
            .with_principal_point([cam.cx as f32, cam.cy as f32]),
        )?;
    }

    // Log the RGB frame.
    rec.log(
        "world/camera/image",
        &rerun::Image::from_elements(rgb.as_slice(), img_size.into(), rerun::ColorModel::RGB),
    )?;

    // Build per-detection 2D overlays and (optionally) 3D transforms.
    let mut strips: Vec<[[f32; 2]; 5]> = Vec::with_capacity(detections.len());
    let mut labels: Vec<String> = Vec::with_capacity(detections.len());
    let mut colors: Vec<rerun::Color> = Vec::with_capacity(detections.len());

    for det in detections {
        let c = &det.quad.corners;
        strips.push([
            [c[0].x, c[0].y],
            [c[1].x, c[1].y],
            [c[2].x, c[2].y],
            [c[3].x, c[3].y],
            [c[0].x, c[0].y],
        ]);
        labels.push(format!("id={}", det.id));
        colors.push(tag_color(det.tag_family_kind.clone()));

        // 3D pose for each tag.
        if let Some((ref cam, tag_size, n_iters)) = pose_args {
            if let Ok(pair) = det.estimate_pose(cam, tag_size, n_iters) {
                // Best candidate (lower reprojection error) in the planar ambiguity.
                let q_best = dquat_from_mat3(&pair.best.pose.rotation);
                let tb = pair.best.pose.translation;
                let (mut q, mut t) = (q_best, glam::DVec3::new(tb.x, tb.y, tb.z));

                // Temporal disambiguation: when we've seen this tag before, choose the
                // candidate whose orientation is closest to last frame (|dot| is
                // hemisphere-agnostic), then low-pass toward it to damp noise. On the
                // first sighting we trust the lower-reprojection "best" as-is.
                if let Some(&(q_prev, t_prev)) = pose_state.get(&det.id) {
                    // Near-frontal tags make the two reprojection errors nearly equal,
                    // so the raw "best" flip-flops; pick the one matching last frame.
                    let q_second = dquat_from_mat3(&pair.second.pose.rotation);
                    if q_second.dot(q_prev).abs() > q_best.dot(q_prev).abs() {
                        q = q_second;
                        let ts = pair.second.pose.translation;
                        t = glam::DVec3::new(ts.x, ts.y, ts.z);
                    }
                    // Align hemispheres before slerp, then low-pass orientation + position.
                    let q_prev = if q_prev.dot(q) < 0.0 {
                        glam::DQuat::from_xyzw(-q_prev.x, -q_prev.y, -q_prev.z, -q_prev.w)
                    } else {
                        q_prev
                    };
                    q = q_prev.slerp(q, smooth_alpha).normalize();
                    t = t_prev.lerp(t, smooth_alpha);
                }
                pose_state.insert(det.id, (q, t));

                let (qx, qy, qz, qw) = (q.x as f32, q.y as f32, q.z as f32, q.w as f32);

                let tag_path = format!("world/tag_{}", det.id);
                let tag_face_path = format!("world/tag_{}/face", det.id);
                let tag_plane_path = format!("world/tag_{}/plane", det.id);
                let tag_axes_path = format!("world/tag_{}/axes", det.id);
                // Transform: tag (child) → world/camera (parent) frame.
                // p_camera = R * p_tag + t maps child→parent, i.e. ParentFromChild.
                // (The camera sits at the world origin, so p_world == p_camera.)
                rec.log(
                    tag_path.as_str(),
                    &rerun::Transform3D::from_translation_rotation(
                        [t.x as f32, t.y as f32, t.z as f32],
                        rerun::Quaternion::from_wxyz([qw, qx, qy, qz]),
                    )
                    .with_relation(rerun::TransformRelation::ParentFromChild),
                )?;

                // Tag corners in the tag frame (planar, z = 0).
                let h = (tag_size / 2.0) as f32;
                let corners3d = [[-h, -h, 0.0f32], [h, -h, 0.0], [h, h, 0.0], [-h, h, 0.0]];
                let col = tag_color(det.tag_family_kind.clone());

                // Filled planar surface (two triangles) so the tag shows as a full plane.
                rec.log(
                    tag_plane_path.as_str(),
                    &rerun::Mesh3D::new(corners3d)
                        .with_triangle_indices([[0, 1, 2], [0, 2, 3]])
                        .with_vertex_colors([col, col, col, col]),
                )?;

                // Crisp outline + id label on top of the plane.
                rec.log(
                    tag_face_path.as_str(),
                    &rerun::LineStrips3D::new([[
                        corners3d[0],
                        corners3d[1],
                        corners3d[2],
                        corners3d[3],
                        corners3d[0],
                    ]])
                    .with_colors([col])
                    .with_labels([format!("{}", det.id)]),
                )?;

                // Draw the tag's coordinate frame (X=red, Y=green, Z=blue) so the
                // 6-DOF orientation is readable in 3D. Arrows inherit the tag pose.
                let axis = tag_size as f32; // one tag-side long
                rec.log(
                    tag_axes_path.as_str(),
                    &rerun::Arrows3D::from_vectors([
                        [axis, 0.0, 0.0],
                        [0.0, axis, 0.0],
                        [0.0, 0.0, axis],
                    ])
                    .with_origins([[0.0f32, 0.0, 0.0]; 3])
                    .with_colors([
                        rerun::Color::from_rgb(255, 0, 0),
                        rerun::Color::from_rgb(0, 255, 0),
                        rerun::Color::from_rgb(0, 0, 255),
                    ]),
                )?;
            }
        }
    }

    // 2D corner overlays on the image.
    if !strips.is_empty() {
        rec.log(
            "world/camera/image",
            &rerun::LineStrips2D::new(strips)
                .with_labels(labels)
                .with_colors(colors),
        )?;
    }

    Ok(())
}

/// Convert a Mat3F64 rotation matrix to a glam quaternion.
fn dquat_from_mat3(r: &kornia_algebra::Mat3F64) -> glam::DQuat {
    // Mat3F64 wraps glam::DMat3. Build from columns.
    glam::DQuat::from_mat3(&glam::DMat3::from_cols(
        glam::DVec3::new(r.x_axis.x, r.x_axis.y, r.x_axis.z),
        glam::DVec3::new(r.y_axis.x, r.y_axis.y, r.y_axis.z),
        glam::DVec3::new(r.z_axis.x, r.z_axis.y, r.z_axis.z),
    ))
}
