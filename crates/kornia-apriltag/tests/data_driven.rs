//! Data-driven regression tests for AprilTag detector.
//!
//! This test suite loads a manifest file (`manifest.json`) and a corresponding dataset
//! to verify the detector's behavior against ground truth.
//!
//! It is NOT intended as a performance benchmark or accuracy competition.
//! It IS intended to prevent regressions in detection (e.g. lost tags, false positives).
//!
//! Requirements:
//! - Manifest must be at `crates/kornia-apriltag/tests/data/manifest.json`.
//! - Dataset is optional. If `crates/kornia-apriltag/tests/data/apriltag_dataset` is missing,
//!   tests will be skipped (CI-safe).

use kornia_apriltag::{AprilTagDecoder, DecodeTagsConfig};
use kornia_apriltag::family::TagFamilyKind;
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

const DATASET_DIR: &str = "tests/data/apriltag_dataset";
const MANIFEST_PATH: &str = "tests/data/manifest.json";

#[derive(Deserialize, Debug)]
struct TestCase {
    filename: String,
    family: String,
    #[serde(rename = "type")]
    test_type: TestType,
    expect_ids: Option<Vec<usize>>,
    min_detections: Option<usize>,
    #[serde(default)]
    description: String,
}

#[derive(Deserialize, Debug, PartialEq)]
#[serde(rename_all = "lowercase")]
enum TestType {
    Golden,
    Edge,
    Negative,
}

fn get_tag_family_kind(name: &str) -> TagFamilyKind {
    match name {
        "tag16h5" => TagFamilyKind::Tag16H5,
        "tag25h9" => TagFamilyKind::Tag25H9,
        "tag36h10" => TagFamilyKind::Tag36H10,
        "tag36h11" => TagFamilyKind::Tag36H11,
        "tagCircle21h7" => TagFamilyKind::TagCircle21H7,
        "tagCircle49h12" => TagFamilyKind::TagCircle49H12,
        "tagCustom48h12" => TagFamilyKind::TagCustom48H12,
        "tagStandard41h12" => TagFamilyKind::TagStandard41H12,
        "tagStandard52h13" => TagFamilyKind::TagStandard52H13,
        _ => panic!("Unknown tag family: {}", name),
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
        panic!(
            "Manifest not found at {:?}. Please ensure the test infrastructure is set up correctly.",
            manifest_path
        );
    }

    let file = File::open(&manifest_path)?;
    let manifest: Vec<TestCase> = serde_json::from_reader(file)?;

    let dataset_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(DATASET_DIR);
    if !dataset_root.exists() {
        eprintln!("AprilTag dataset not found at {:?}, skipping data-driven tests", dataset_root);
        return Ok(());
    }

    for case in manifest {
        println!("Running test case: {} ({})", case.filename, case.description);
        let image_path = dataset_root.join(&case.filename);
        
        // Fail if image is missing
        if !image_path.exists() {
            panic!("Test failed: Image file missing at {:?}", image_path);
        }

        let image = load_image_mono8(&image_path).map_err(|e| format!("Failed to load {}: {}", case.filename, e))?;
        let family_kind = get_tag_family_kind(&case.family);

        let config = DecodeTagsConfig::new(vec![family_kind.clone()])?;
        // Use default image size or read from image
        let img_size = image.size();
        let mut decoder = AprilTagDecoder::new(config, img_size)?;
        
        let detections = decoder.decode(&image)?;

        match case.test_type {
            TestType::Golden => {
                let expect_ids = case.expect_ids.clone().expect("Golden test must have expect_ids");
                let detected_ids: Vec<usize> = detections.iter().map(|d| d.id as usize).collect();
                
                // Sort for comparison
                let mut sorted_detected = detected_ids.clone();
                sorted_detected.sort();
                let mut sorted_expected = expect_ids.clone();
                sorted_expected.sort();

                assert_eq!(sorted_detected, sorted_expected, "Golden test failed for {}", case.filename);
            }
            TestType::Edge => {
                 let min_detections = case.min_detections.expect("Edge test must have min_detections");
                 assert!(detections.len() >= min_detections, "Edge test failed for {}, expected >= {} detections, got {}", case.filename, min_detections, detections.len());
            }
            TestType::Negative => {
                assert_eq!(detections.len(), 0, "Negative test failed for {}, expected 0 detections", case.filename);
            }
        }
    }

    Ok(())
}
