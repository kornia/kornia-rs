//! Data-driven regression tests for AprilTag detector.
//!
//! Loads a manifest (`manifest.json`) referencing test images already tracked in the repo.
//! Validates detection count, tag IDs, and geometric properties (center + corners).
//!
//! NOT a performance benchmark. IS a regression guard against lost tags, false positives,
//! and geometric drift.

use kornia_apriltag::family::TagFamilyKind;
use kornia_apriltag::{AprilTagDecoder, DecodeTagsConfig};
use kornia_image::allocator::CpuAllocator;
use kornia_image::color_spaces::Gray8;
use kornia_image::Image;
use kornia_imgproc::color::ConvertColor;
use kornia_io::{
    jpeg::{read_image_jpeg_mono8, read_image_jpeg_rgb8},
    png::{read_image_png_mono8, read_image_png_rgb8},
};
use serde::Deserialize;
use std::fs::File;
use std::path::{Path, PathBuf};

const MANIFEST_PATH: &str = "tests/data/manifest.json";

#[derive(Deserialize, Debug)]
struct TestCase {
    filename: String,
    family: String,
    #[serde(rename = "type")]
    test_type: TestType,
    expect_ids: Option<Vec<usize>>,
    expect_detections: Option<Vec<ExpectedDetection>>,
    tolerance_px: Option<f32>,
    min_detections: Option<usize>,
    #[serde(default)]
    description: String,
}

#[derive(Deserialize, Debug, Clone)]
struct ExpectedDetection {
    id: usize,
    center: [f32; 2],
    corners: [[f32; 2]; 4],
}

#[derive(Deserialize, Debug, PartialEq)]
#[serde(rename_all = "lowercase")]
enum TestType {
    Golden,
    Edge,
    Negative,
}

fn get_tag_family_kind(name: &str) -> Result<TagFamilyKind, String> {
    match name {
        "tag16h5" => Ok(TagFamilyKind::Tag16H5),
        "tag25h9" => Ok(TagFamilyKind::Tag25H9),
        "tag36h10" => Ok(TagFamilyKind::Tag36H10),
        "tag36h11" => Ok(TagFamilyKind::Tag36H11),
        "tagCircle21h7" => Ok(TagFamilyKind::TagCircle21H7),
        "tagCircle49h12" => Ok(TagFamilyKind::TagCircle49H12),
        "tagCustom48h12" => Ok(TagFamilyKind::TagCustom48H12),
        "tagStandard41h12" => Ok(TagFamilyKind::TagStandard41H12),
        "tagStandard52h13" => Ok(TagFamilyKind::TagStandard52H13),
        _ => Err(format!("Unknown tag family: {}", name)),
    }
}

fn load_image_mono8(path: &Path) -> Result<Image<u8, 1, CpuAllocator>, Box<dyn std::error::Error>> {
    if !path.exists() {
        return Err(format!("Image file not found: {:?}", path).into());
    }

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_lowercase())
        .ok_or("No extension")?;

    match ext.as_str() {
        "png" => match read_image_png_mono8(path) {
            Ok(img) => Ok(img.into_inner()),
            Err(_) => {
                let img = read_image_png_rgb8(path)?;
                let mut gray = Gray8::from_size_val(img.size(), 0, CpuAllocator)?;
                img.convert(&mut gray)?;
                Ok(gray.into_inner())
            }
        },
        "jpg" | "jpeg" => match read_image_jpeg_mono8(path) {
            Ok(img) => Ok(img.into_inner()),
            Err(_) => {
                let img = read_image_jpeg_rgb8(path)?;
                let mut gray = Gray8::from_size_val(img.size(), 0, CpuAllocator)?;
                img.convert(&mut gray)?;
                Ok(gray.into_inner())
            }
        },
        _ => Err(format!("Unsupported extension: {}", ext).into()),
    }
}

#[test]
fn run_data_driven_tests() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(MANIFEST_PATH);
    if !manifest_path.exists() {
        return Err(format!(
            "Manifest not found at {:?}. Please ensure the test infrastructure is set up correctly.",
            manifest_path
        )
        .into());
    }

    let file = File::open(&manifest_path)?;
    let manifest: Vec<TestCase> = serde_json::from_reader(file)?;

    for case in manifest {
        println!(
            "Running test case: {} ({})",
            case.filename, case.description
        );
        let image_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(&case.filename);

        if !image_path.exists() {
            return Err(
                format!("Test image not found at {:?}", image_path).into()
            );
        }

        let image = load_image_mono8(&image_path)
            .map_err(|e| format!("Failed to load {}: {}", case.filename, e))?;
        let family_kind = get_tag_family_kind(&case.family)?;

        let config = DecodeTagsConfig::new(vec![family_kind.clone()])?;
        let mut decoder = AprilTagDecoder::new(config, image.size())?;
        let detections = decoder.decode(&image)?;

        match case.test_type {
            TestType::Golden => {
                // Validate tag IDs (sorted comparison)
                if let Some(ref expect_ids) = case.expect_ids {
                    let mut detected_ids: Vec<usize> =
                        detections.iter().map(|d| d.id as usize).collect();
                    detected_ids.sort();
                    let mut sorted_expected = expect_ids.clone();
                    sorted_expected.sort();

                    assert_eq!(
                        detected_ids, sorted_expected,
                        "ID mismatch for '{}': got {:?}, expected {:?}",
                        case.description, detected_ids, sorted_expected
                    );
                }

                // Validate geometric properties (center + corners)
                if let Some(ref expected_dets) = case.expect_detections {
                    let tol = case.tolerance_px.unwrap_or(5.0);
                    let mut candidates: Vec<_> = detections.iter().collect();

                    for expected in expected_dets {
                        let best_idx = candidates
                            .iter()
                            .enumerate()
                            .filter(|(_, d)| d.id as usize == expected.id)
                            .min_by(|(_, a), (_, b)| {
                                let da = (a.center.x - expected.center[0]).powi(2)
                                    + (a.center.y - expected.center[1]).powi(2);
                                let db = (b.center.x - expected.center[0]).powi(2)
                                    + (b.center.y - expected.center[1]).powi(2);
                                da.partial_cmp(&db).unwrap()
                            })
                            .map(|(idx, _)| idx)
                            .ok_or_else(|| {
                                format!(
                                    "No detection with id {} near ({}, {}) in '{}'",
                                    expected.id,
                                    expected.center[0],
                                    expected.center[1],
                                    case.description
                                )
                            })?;

                        let matched = candidates.remove(best_idx);

                        // Validate center coordinates
                        assert!(
                            (matched.center.x - expected.center[0]).abs() <= tol
                                && (matched.center.y - expected.center[1]).abs() <= tol,
                            "Center mismatch in '{}': got ({:.1}, {:.1}), expected ({:.1}, {:.1}), tol={}",
                            case.description,
                            matched.center.x,
                            matched.center.y,
                            expected.center[0],
                            expected.center[1],
                            tol
                        );

                        // Validate corner coordinates
                        for (i, (actual, exp)) in matched
                            .quad
                            .corners
                            .iter()
                            .zip(expected.corners.iter())
                            .enumerate()
                        {
                            assert!(
                                (actual.x - exp[0]).abs() <= tol
                                    && (actual.y - exp[1]).abs() <= tol,
                                "Corner {} mismatch in '{}': got ({:.1}, {:.1}), expected ({:.1}, {:.1}), tol={}",
                                i,
                                case.description,
                                actual.x,
                                actual.y,
                                exp[0],
                                exp[1],
                                tol
                            );
                        }
                    }
                }
            }
            TestType::Edge => {
                let min = case
                    .min_detections
                    .expect("Edge test must have min_detections");
                assert!(
                    detections.len() >= min,
                    "Edge test '{}': expected >= {} detections, got {}",
                    case.description,
                    min,
                    detections.len()
                );
            }
            TestType::Negative => {
                assert_eq!(
                    detections.len(),
                    0,
                    "Negative test '{}': expected 0 detections, got {}",
                    case.description,
                    detections.len()
                );
            }
        }
    }

    Ok(())
}
