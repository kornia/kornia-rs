//! Completed multi-camera calibration from a known AprilTag board.
//!
//! Run from the repository root:
//!
//! ```text
//! pixi run cargo run -p apriltag_board_multicam -- \
//!   camera_0.png camera_1.png [camera_2.png ...]
//! ```
//!
//! Set `RERUN_SAVE=/tmp/apriltag_board_multicam.rrd` to save a recording
//! instead of spawning the Rerun viewer.

use std::error::Error;
use std::path::{Path, PathBuf};

use argh::FromArgs;
use kornia::k3d::camera::PinholeCamera;
use kornia::k3d::pose::Pose3d;
use kornia_algebra::{QuatF64, Vec2F64, Vec3F64};
use kornia_apriltag::{
    decoder::Detection, family::TagFamilyKind, AprilTagDecoder, DecodeTagsConfig,
};
use kornia_calib::{
    calibrate_board, estimate_focal, BoardGeometry, CalibConfig, RigCalibration, TagObservation,
};
use kornia_imgproc::color::{ConvertColorExt, Gray8, Rgb8};
use kornia_io::functional::read_image_any_rgb8;
use rerun::blueprint::{
    Blueprint, BlueprintActivation, ContainerLike, Horizontal, Spatial2DView, Spatial3DView,
    Vertical,
};

const PRINTED_TAG_SIZE_M: f64 = 0.016;
const PRINTED_TAG_PITCH_M: f64 = 0.020;
const BOARD_SCALE: f64 = 1.6;
const TAG_SIZE_M: f64 = PRINTED_TAG_SIZE_M * BOARD_SCALE;
const TAG_PITCH_M: f64 = PRINTED_TAG_PITCH_M * BOARD_SCALE;
const ASSUMED_EQUIVALENT_FOCAL_MM: f64 = 24.0;
const FULL_FRAME_DIAGONAL_MM: f64 = 43.266_615_305_567_87;
const MIN_DECISION_MARGIN: f32 = 20.0;
const MAX_HAMMING: u8 = 1;
const DETECTOR_DOWNSCALE_FACTOR: usize = 6;
const MIN_PROJECTION_DEPTH_M: f64 = 1e-6;
const IMAGE_PLANE_DISTANCE_M: f32 = 0.1;
const TAG_CANONICAL_CORNERS: [(f32, f32); 4] = [(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)];
const TAG_IDS: [[u16; 6]; 6] = [
    [35, 34, 33, 32, 31, 30],
    [29, 28, 27, 26, 25, 24],
    [23, 22, 21, 20, 19, 18],
    [17, 16, 15, 14, 13, 12],
    [11, 10, 9, 8, 7, 6],
    [5, 4, 3, 2, 1, 0],
];

#[derive(FromArgs)]
/// Calibrate a multi-camera rig from images of the same stationary AprilTag board.
struct Args {
    /// input image paths, one per camera
    #[argh(positional)]
    image_paths: Vec<PathBuf>,
}

fn validated_image_paths(image_paths: Vec<PathBuf>) -> Result<Vec<PathBuf>, String> {
    if image_paths.len() < 2 {
        return Err("expected at least two camera image paths".to_owned());
    }
    Ok(image_paths)
}

/// Builds geometry for the
/// [MakerWorld 6x6 AprilTag board](https://makerworld.com/en/models/1552797-3d-printable-6x6-apriltag-calibration-board#profileId-1631086).
///
/// Its tag IDs are horizontally reversed relative to [`BoardGeometry::april_grid`],
/// so their physical corners are defined explicitly.
fn makerworld_6x6_board() -> BoardGeometry {
    let mut board_tag_corners: Vec<(u16, [Vec3F64; 4])> = Vec::new();
    let half_tag_size = TAG_SIZE_M / 2.0;
    let center_row_index = (TAG_IDS.len() as f64 - 1.0) / 2.0;
    let center_column_index = (TAG_IDS[0].len() as f64 - 1.0) / 2.0;

    for (row_index, row_ids) in TAG_IDS.iter().enumerate() {
        for (column_index, &tag_id) in row_ids.iter().enumerate() {
            let center_x = (column_index as f64 - center_column_index) * TAG_PITCH_M;
            let center_y = (center_row_index - row_index as f64) * TAG_PITCH_M;
            let corners = [
                Vec3F64::new(center_x - half_tag_size, center_y + half_tag_size, 0.0), // TL
                Vec3F64::new(center_x + half_tag_size, center_y + half_tag_size, 0.0), // TR
                Vec3F64::new(center_x + half_tag_size, center_y - half_tag_size, 0.0), // BR
                Vec3F64::new(center_x - half_tag_size, center_y - half_tag_size, 0.0), // BL
            ];
            board_tag_corners.push((tag_id, corners));
        }
    }

    BoardGeometry::from_corners(board_tag_corners)
}

/// Creates a simple initial camera model when calibrated intrinsics are unavailable.
fn approximate_pinhole(width: usize, height: usize) -> PinholeCamera {
    let width = width as f64;
    let height = height as f64;
    // Convert the assumed full-frame-equivalent focal length into focal pixels.
    let image_diagonal_px = width.hypot(height);
    let fx = image_diagonal_px * ASSUMED_EQUIVALENT_FOCAL_MM / FULL_FRAME_DIAGONAL_MM;

    PinholeCamera {
        fx,
        fy: fx,
        cx: width / 2.0,
        cy: height / 2.0,
        k1: 0.0,
        k2: 0.0,
        p1: 0.0,
        p2: 0.0,
    }
}

/// Projects the tag's canonical corners in the calibration API's aruco winding.
fn calibration_corners(detection: &Detection) -> [Vec2F64; 4] {
    TAG_CANONICAL_CORNERS.map(|(x, y)| {
        let corner = detection.quad.homography_project(x, y);
        Vec2F64::new(corner.x as f64, corner.y as f64)
    })
}

/// Reorganizes camera-major detections into tag-major calibration observations.
fn build_observations(
    detections_per_camera: &[Vec<Detection>],
    board: &BoardGeometry,
) -> Vec<TagObservation> {
    let mut observations = Vec::new();

    for row_ids in &TAG_IDS {
        for &tag_id in row_ids {
            let mut per_camera = Vec::new();
            for (camera_index, detections) in detections_per_camera.iter().enumerate() {
                let detection = detections.iter().find(|detection| {
                    detection.id == tag_id
                        && detection.hamming <= MAX_HAMMING
                        && detection.decision_margin >= MIN_DECISION_MARGIN
                });
                if let Some(detection) = detection {
                    per_camera.push((camera_index, calibration_corners(detection)));
                }
            }

            if board.contains(tag_id) && !per_camera.is_empty() {
                observations.push(TagObservation { tag_id, per_camera });
            }
        }
    }

    observations
}

/// Refines one square-pixel focal length per camera from the known planar board.
fn estimate_camera_focals(
    initial_cameras: &[PinholeCamera],
    observations: &[TagObservation],
    board: &BoardGeometry,
) -> Vec<PinholeCamera> {
    initial_cameras
        .iter()
        .enumerate()
        .map(|(camera_index, initial)| {
            let mut board_points = Vec::new();
            let mut image_points = Vec::new();
            for observation in observations {
                let Some(object_corners) = board.object_points(observation.tag_id) else {
                    continue;
                };
                let Some((_, measured_corners)) = observation
                    .per_camera
                    .iter()
                    .find(|(index, _)| *index == camera_index)
                else {
                    continue;
                };
                for (object, measured) in object_corners.iter().zip(measured_corners) {
                    board_points.push(Vec2F64::new(object.x, object.y));
                    image_points.push(*measured);
                }
            }

            let mut camera = initial.clone();
            if let Some(focal) =
                estimate_focal(&board_points, &image_points, initial.cx, initial.cy)
            {
                camera.fx = focal;
                camera.fy = focal;
            }
            camera
        })
        .collect()
}

struct CalibrationSolution {
    calibration: RigCalibration,
    num_observations: usize,
    cameras: Vec<PinholeCamera>,
}

/// Calibrates every camera from the decoder's orientation-aware tag corners.
fn solve_calibration(
    cameras: &[PinholeCamera],
    detections_per_camera: &[Vec<Detection>],
    board: &BoardGeometry,
) -> Result<CalibrationSolution, Box<dyn Error>> {
    let config = CalibConfig::new(TAG_SIZE_M);
    let observations = build_observations(detections_per_camera, board);
    let refined_cameras = estimate_camera_focals(cameras, &observations, board);
    let calibration = calibrate_board(&refined_cameras, &observations, &[], board, &config)?;
    let registered = calibration.poses.iter().flatten().count();
    println!(
        "calibration: RMS={:.3} px, registered={registered}/{}",
        calibration.reproj_rmse_px,
        cameras.len(),
    );

    let usable = registered == cameras.len()
        && calibration.reproj_rmse_px.is_finite()
        && calibration.reproj_rmse_px >= 0.0;
    if !usable {
        return Err("calibration did not produce a valid pose for every camera".into());
    }

    Ok(CalibrationSolution {
        calibration,
        num_observations: observations.len(),
        cameras: refined_cameras,
    })
}

/// Converts a Kornia rotation matrix into a Rerun quaternion.
fn rerun_quaternion(pose: &Pose3d) -> rerun::Quaternion {
    let [x, y, z, w] = QuatF64::from_mat3(&pose.rotation).to_array();
    rerun::Quaternion::from_wxyz([w as f32, x as f32, y as f32, z as f32])
}

fn open_recording_stream() -> Result<rerun::RecordingStream, Box<dyn Error>> {
    let builder = rerun::RecordingStreamBuilder::new("AprilTag Board Multicam Calibration");
    if let Some(path) = std::env::var_os("RERUN_SAVE") {
        Ok(builder.save(Path::new(&path))?)
    } else {
        Ok(builder.spawn()?)
    }
}

/// Gives the calibration viewer one 3D view and one image view per camera.
fn send_calibration_blueprint(
    rec: &rerun::RecordingStream,
    num_cameras: usize,
) -> Result<(), Box<dyn Error>> {
    let camera_views = (0..num_cameras)
        .map(|camera_index| {
            Spatial2DView::new(format!("Camera {camera_index}"))
                .with_origin(format!("world/camera_{camera_index}/image"))
                .into()
        })
        .collect::<Vec<ContainerLike>>();
    let camera_views = Vertical::new(camera_views).with_row_shares(vec![1.0; num_cameras]);

    let layout = Horizontal::new([
        Spatial3DView::new("Calibrated rig")
            .with_origin("world")
            .with_defaults(
                &rerun::Pinhole::update_fields().with_image_plane_distance(IMAGE_PLANE_DISTANCE_M),
            )
            .into(),
        camera_views.into(),
    ])
    .with_column_shares([1.0, 1.0]);

    Blueprint::new(layout).send(rec, BlueprintActivation::default())?;
    Ok(())
}

/// Logs the physical board, calibrated cameras, images, detections, and reprojections.
fn log_rerun(
    rec: &rerun::RecordingStream,
    board: &BoardGeometry,
    cameras: &[PinholeCamera],
    images: &[Rgb8],
    detections_per_camera: &[Vec<Detection>],
    calibration: &RigCalibration,
) -> Result<(), Box<dyn Error>> {
    rec.log_static("/", &rerun::ViewCoordinates::RIGHT_HAND_Z_UP())?;

    let mut board_strips = Vec::new();
    let mut board_labels = Vec::new();
    for row_ids in &TAG_IDS {
        for &tag_id in row_ids {
            if let Some(corners) = board.object_points(tag_id) {
                board_strips.push([
                    corners[0].to_array().map(|value| value as f32),
                    corners[1].to_array().map(|value| value as f32),
                    corners[2].to_array().map(|value| value as f32),
                    corners[3].to_array().map(|value| value as f32),
                    corners[0].to_array().map(|value| value as f32),
                ]);
                board_labels.push(format!("id={tag_id}"));
            }
        }
    }
    rec.log_static(
        "world/board/tags",
        &rerun::LineStrips3D::new(board_strips)
            .with_colors([rerun::Color::from_rgb(255, 200, 0)])
            .with_labels(board_labels),
    )?;

    for (camera_index, ((camera, image), detections)) in cameras
        .iter()
        .zip(images)
        .zip(detections_per_camera)
        .enumerate()
    {
        let Some(camera_to_world) = calibration.poses.get(camera_index).copied().flatten() else {
            continue;
        };
        let camera_path = format!("world/camera_{camera_index}");
        let image_path = format!("{camera_path}/image");

        rec.log_static(camera_path.as_str(), &rerun::ViewCoordinates::RDF())?;
        rec.log(
            camera_path.as_str(),
            &rerun::Transform3D::from_translation_rotation(
                camera_to_world
                    .translation
                    .to_array()
                    .map(|value| value as f32),
                rerun_quaternion(&camera_to_world),
            )
            .with_relation(rerun::TransformRelation::ParentFromChild),
        )?;
        rec.log_static(
            image_path.as_str(),
            &rerun::Pinhole::from_focal_length_and_resolution(
                [camera.fx as f32, camera.fy as f32],
                [image.width() as f32, image.height() as f32],
            )
            .with_principal_point([camera.cx as f32, camera.cy as f32]),
        )?;
        rec.log(
            image_path.as_str(),
            &rerun::Image::from_elements(
                image.as_slice(),
                image.size().into(),
                rerun::ColorModel::RGB,
            ),
        )?;

        let mut measured_strips = Vec::new();
        let mut measured_labels = Vec::new();
        for detection in detections {
            if !board.contains(detection.id) {
                continue;
            }
            let corners = &detection.quad.corners;
            measured_strips.push([
                [corners[0].x, corners[0].y],
                [corners[1].x, corners[1].y],
                [corners[2].x, corners[2].y],
                [corners[3].x, corners[3].y],
                [corners[0].x, corners[0].y],
            ]);
            measured_labels.push(format!("id={}", detection.id));
        }
        rec.log(
            image_path.as_str(),
            &rerun::LineStrips2D::new(measured_strips)
                .with_colors([rerun::Color::from_rgb(0, 220, 255)])
                .with_labels(measured_labels),
        )?;

        let world_to_camera = camera_to_world.inverse();
        let mut projected_strips = Vec::new();
        for row_ids in &TAG_IDS {
            for &tag_id in row_ids {
                let Some(object_corners) = board.object_points(tag_id) else {
                    continue;
                };
                let projected: Option<Vec<[f32; 2]>> = object_corners
                    .iter()
                    .chain(std::iter::once(&object_corners[0]))
                    .map(|point_world| {
                        let point_camera = world_to_camera.transform_point(point_world);
                        camera
                            .project_to_pixel(&point_camera, MIN_PROJECTION_DEPTH_M)
                            .map(|pixel| [pixel.x as f32, pixel.y as f32])
                    })
                    .collect();
                if let Some(projected) = projected {
                    projected_strips.push(projected);
                }
            }
        }
        rec.log(
            image_path.as_str(),
            &rerun::LineStrips2D::new(projected_strips)
                .with_colors([rerun::Color::from_rgb(0, 255, 80)])
                .with_radii([rerun::Radius::new_ui_points(1.0)]),
        )?;
    }

    Ok(())
}

fn load_camera_image(path: &Path) -> Result<(Rgb8, PinholeCamera, Vec<Detection>), Box<dyn Error>> {
    let image = read_image_any_rgb8(path)?;
    let gray: Gray8 = image.cvt()?;

    let mut config = DecodeTagsConfig::new(vec![TagFamilyKind::Tag36H11])?;
    config.downscale_factor = DETECTOR_DOWNSCALE_FACTOR;
    config.refine_edges_enabled = true;
    let mut detector = AprilTagDecoder::new(config, gray.size())?;
    let detections = detector.decode(&gray)?;
    let camera = approximate_pinhole(image.width(), image.height());
    Ok((image, camera, detections))
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Args = argh::from_env();
    let image_paths = validated_image_paths(args.image_paths)?;
    let board = makerworld_6x6_board();
    let mut cameras = Vec::new();
    let mut images = Vec::new();
    let mut detections_per_camera = Vec::new();

    for (camera_index, path) in image_paths.iter().enumerate() {
        let (image, camera, detections) = load_camera_image(path)?;
        println!(
            "camera {camera_index}: {} detections, {}x{}",
            detections.len(),
            image.width(),
            image.height()
        );
        cameras.push(camera);
        images.push(image);
        detections_per_camera.push(detections);
    }

    let solution = solve_calibration(&cameras, &detections_per_camera, &board)?;
    println!(
        "\nreprojection RMS: {:.3} px\ntag observations: {}",
        solution.calibration.reproj_rmse_px, solution.num_observations
    );

    let relative_poses = solution.calibration.rebased(0, None)?;
    for (camera_index, camera_in_camera_0) in relative_poses.iter().enumerate().skip(1) {
        let Some(camera_in_camera_0) = camera_in_camera_0 else {
            continue;
        };
        println!(
            "camera {camera_index} position in camera 0 frame: [{:.4}, {:.4}, {:.4}] m",
            camera_in_camera_0.translation.x,
            camera_in_camera_0.translation.y,
            camera_in_camera_0.translation.z
        );
        println!(
            "camera {camera_index} baseline: {:.4} m",
            camera_in_camera_0.translation.length()
        );
    }
    for (stats, camera) in solution
        .calibration
        .per_camera
        .iter()
        .zip(&solution.cameras)
    {
        println!(
            "camera {}: focal={:.1} px, registered={}, observations={}, RMS={:.3} px",
            stats.camera, camera.fx, stats.registered, stats.num_obs, stats.reproj_rmse_px
        );
    }

    let rec = open_recording_stream()?;
    send_calibration_blueprint(&rec, images.len())?;
    log_rerun(
        &rec,
        &board,
        &solution.cameras,
        &images,
        &detections_per_camera,
        &solution.calibration,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_requires_at_least_two_image_paths() -> Result<(), Box<dyn Error>> {
        let args = Args::from_args(&["apriltag_board_multicam"], &["camera_0.png"])
            .map_err(|error| std::io::Error::other(error.output))?;

        assert!(validated_image_paths(args.image_paths).is_err());
        Ok(())
    }

    #[test]
    fn cli_accepts_all_camera_image_paths() -> Result<(), Box<dyn Error>> {
        let args = Args::from_args(
            &["apriltag_board_multicam"],
            &["camera_0.png", "camera_1.png", "camera_2.png"],
        )
        .map_err(|error| std::io::Error::other(error.output))?;

        let paths = validated_image_paths(args.image_paths).map_err(std::io::Error::other)?;
        assert_eq!(
            paths,
            [
                PathBuf::from("camera_0.png"),
                PathBuf::from("camera_1.png"),
                PathBuf::from("camera_2.png"),
            ]
        );
        Ok(())
    }

    #[test]
    fn makerworld_board_has_expected_scaled_geometry() -> Result<(), Box<dyn Error>> {
        const GEOMETRY_TOLERANCE_M: f64 = 1e-12;

        let board = makerworld_6x6_board();

        let tag_35 = board
            .object_points(35)
            .ok_or_else(|| std::io::Error::other("board is missing tag 35"))?;
        let tag_0 = board
            .object_points(0)
            .ok_or_else(|| std::io::Error::other("board is missing tag 0"))?;

        assert!((tag_35[0].x - -0.0928).abs() < GEOMETRY_TOLERANCE_M);
        assert!((tag_35[0].y - 0.0928).abs() < GEOMETRY_TOLERANCE_M);
        assert!((tag_0[2].x - 0.0928).abs() < GEOMETRY_TOLERANCE_M);
        assert!((tag_0[2].y - -0.0928).abs() < GEOMETRY_TOLERANCE_M);
        Ok(())
    }

    #[test]
    fn calibration_corners_follow_decoded_orientation() -> Result<(), Box<dyn Error>> {
        const NO_DOWNSCALE_FACTOR: usize = 1;
        const CORNER_TOLERANCE_PX: f64 = 1e-4;

        use kornia_io::png::read_image_png_mono8;

        let image_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/apriltag.png");
        let image = read_image_png_mono8(image_path)?;
        let mut config = DecodeTagsConfig::new(vec![TagFamilyKind::Tag36H11])?;
        config.downscale_factor = NO_DOWNSCALE_FACTOR;
        let mut detector = AprilTagDecoder::new(config, image.size())?;
        let detections = detector.decode(&image)?;
        assert_eq!(detections.len(), 1);

        let detection = detections
            .first()
            .ok_or_else(|| std::io::Error::other("fixture did not contain an AprilTag"))?;
        let corners = calibration_corners(detection);
        let detected = &detection.quad.corners;
        // The fixture's decoded tag frame is half a turn from the raw geometric winding.
        let expected = [detected[1], detected[2], detected[3], detected[0]]
            .map(|corner| Vec2F64::new(corner.x as f64, corner.y as f64));
        for (corner, expected) in corners.iter().zip(expected) {
            assert!(
                (*corner - expected).length() < CORNER_TOLERANCE_PX,
                "corner {corner:?}, expected {expected:?}"
            );
        }

        Ok(())
    }

    #[test]
    fn sample_images_produce_valid_two_camera_calibration() -> Result<(), Box<dyn Error>> {
        const MAX_REPROJECTION_RMS_PX: f64 = 1.5;
        const MIN_BASELINE_M: f64 = 0.34;
        const MAX_BASELINE_M: f64 = 0.38;

        let image_paths = [
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../tests/data/apriltag_board_multicam/camera_0.jpg"
            ),
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../tests/data/apriltag_board_multicam/camera_1.jpg"
            ),
        ];
        let board = makerworld_6x6_board();
        let mut cameras = Vec::new();
        let mut detections_per_camera = Vec::new();
        for path in image_paths {
            let (_, camera, detections) = load_camera_image(Path::new(path))?;
            cameras.push(camera);
            detections_per_camera.push(detections);
        }

        let solution = solve_calibration(&cameras, &detections_per_camera, &board)?;
        assert!(
            solution.calibration.reproj_rmse_px < MAX_REPROJECTION_RMS_PX,
            "reprojection RMS was {} px",
            solution.calibration.reproj_rmse_px
        );

        let relative_poses = solution.calibration.rebased(0, None)?;
        let camera_1 = relative_poses
            .get(1)
            .copied()
            .flatten()
            .ok_or_else(|| std::io::Error::other("camera 1 was not registered"))?;
        let baseline = camera_1.translation.length();
        assert!(
            (MIN_BASELINE_M..=MAX_BASELINE_M).contains(&baseline),
            "baseline was {baseline} m"
        );
        Ok(())
    }

    #[test]
    fn calibration_blueprint_sets_larger_image_plane_distance() -> Result<(), Box<dyn Error>> {
        let (rec, storage) = rerun::RecordingStreamBuilder::new("blueprint-test").memory()?;

        send_calibration_blueprint(&rec, 3)?;
        rec.flush_blocking()?;

        let distance_component = rerun::Pinhole::descriptor_image_plane_distance().component;
        let name_component = rerun::external::re_sdk_types::blueprint::archetypes::ViewBlueprint::
            descriptor_display_name()
            .component;
        let mut distances = Vec::new();
        let mut num_views = 0;
        for message in storage.take() {
            let rerun::log::LogMsg::ArrowMsg(_, message) = message else {
                continue;
            };
            let chunk = rerun::log::Chunk::from_arrow_msg(&message)?;
            for batch in
                chunk.iter_component::<rerun::components::ImagePlaneDistance>(distance_component)
            {
                distances.extend(batch.iter().copied().map(f32::from));
            }
            for batch in chunk.iter_component::<rerun::components::Name>(name_component) {
                num_views += batch.len();
            }
        }

        assert_eq!(distances, [IMAGE_PLANE_DISTANCE_M]);
        assert_eq!(num_views, 4);
        Ok(())
    }
}
