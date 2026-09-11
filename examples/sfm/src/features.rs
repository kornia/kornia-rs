//! Feature detection and descriptor extraction.
//!
//! Provides a common [`FeatureExtractor`] interface over the ORB and SIFT
//! detectors in `kornia-imgproc`, so the rest of the pipeline is agnostic to
//! which detector produced the features. A detector is selected at runtime via
//! the [`DetectorKind`] CLI argument.

use std::error::Error;
use std::str::FromStr;

use kornia_image::Image;
use kornia_imgproc::features::{
    sift_detect_and_compute, FirstOctave, OrbDetector, SiftConfig, SiftWorkspace,
};

/// The feature detector/descriptor to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectorKind {
    /// Oriented FAST + rotated BRIEF. Binary 256-bit descriptors, fast.
    Orb,
    /// Scale-invariant feature transform. 128-D float descriptors, robust.
    Sift,
}

impl FromStr for DetectorKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "orb" => Ok(DetectorKind::Orb),
            "sift" => Ok(DetectorKind::Sift),
            _ => Err(format!("unknown detector: {s} (expected 'orb' or 'sift')")),
        }
    }
}

// `argh::FromArgValue` has a blanket impl for any `T: FromStr`, so `DetectorKind`
// is automatically accepted as an `argh` CLI argument type.

/// Unified feature output from any detector.
///
/// Exactly one of `descriptors_orb` / `descriptors_sift` is populated,
/// depending on which detector produced the features.
#[derive(Debug, Clone)]
pub struct FrameFeatures {
    /// Keypoint positions as `[col, row]` in full-resolution pixel coordinates.
    pub keypoints: Vec<[f32; 2]>,
    /// ORB descriptors: 256-bit packed binary, one per keypoint.
    pub descriptors_orb: Option<Vec<[u8; 32]>>,
    /// SIFT descriptors: flat buffer, `keypoints.len() * 128` floats (row-major).
    pub descriptors_sift: Option<Vec<f32>>,
}

impl FrameFeatures {
    /// Number of keypoints detected in the frame.
    pub fn n_keypoints(&self) -> usize {
        self.keypoints.len()
    }
}

/// Extracts keypoints and descriptors from a single grayscale frame.
pub trait FeatureExtractor {
    /// Extract features from a grayscale image.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying detector fails.
    fn extract(&self, gray: &Image<u8, 1>) -> Result<FrameFeatures, Box<dyn Error>>;
}

/// ORB feature extractor (binary descriptors, Hamming matching).
pub struct OrbExtractor {
    /// Maximum number of keypoints to retain per frame.
    pub n_features: usize,
}

impl FeatureExtractor for OrbExtractor {
    fn extract(&self, gray: &Image<u8, 1>) -> Result<FrameFeatures, Box<dyn Error>> {
        let orb = OrbDetector {
            n_keypoints: self.n_features,
            ..OrbDetector::new()
        };
        let features = orb.detect_and_extract_u8(gray)?;
        Ok(FrameFeatures {
            keypoints: features.keypoints_xy,
            descriptors_orb: Some(features.descriptors),
            descriptors_sift: None,
        })
    }
}

/// SIFT feature extractor (float descriptors, L2 matching).
pub struct SiftExtractor {
    /// Maximum number of keypoints to retain per frame (`0` = unlimited).
    pub n_features: usize,
}

impl FeatureExtractor for SiftExtractor {
    fn extract(&self, gray: &Image<u8, 1>) -> Result<FrameFeatures, Box<dyn Error>> {
        let w = gray.width();
        let h = gray.height();
        // SIFT's pipeline scales input by 1/255 internally, so feed the raw
        // 0..255 pixel values (not normalized to 0..1).
        let buf: Vec<f32> = gray.as_slice().iter().map(|&v| v as f32).collect();

        let cfg = SiftConfig {
            n_features: self.n_features,
            ..SiftConfig::default()
        };
        let mut ws = SiftWorkspace::new();
        let feats = sift_detect_and_compute(
            &mut ws,
            &buf,
            w,
            h,
            &cfg,
            FirstOctave::Native,
            usize::MAX,
            false,
        )?;

        let keypoints: Vec<[f32; 2]> = feats
            .keypoints
            .iter()
            .map(|kp| [kp.x, kp.y]) // x = col, y = row
            .collect();

        Ok(FrameFeatures {
            keypoints,
            descriptors_orb: None,
            descriptors_sift: Some(feats.descriptors),
        })
    }
}

/// Build an extractor for the given detector kind.
pub fn make_extractor(kind: DetectorKind, n_features: usize) -> Box<dyn FeatureExtractor> {
    match kind {
        DetectorKind::Orb => Box::new(OrbExtractor { n_features }),
        DetectorKind::Sift => Box::new(SiftExtractor { n_features }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A checkerboard image: corners everywhere, so ORB and SIFT both fire.
    fn make_checkerboard(width: usize, height: usize, block: usize) -> Image<u8, 1> {
        use kornia_image::ImageSize;
        let mut img = Image::<u8, 1>::from_size_val(ImageSize { width, height }, 0).unwrap();
        let data = img.as_slice_mut();
        for y in 0..height {
            for x in 0..width {
                let on = (x / block + y / block).is_multiple_of(2);
                data[y * width + x] = if on { 255 } else { 0 };
            }
        }
        img
    }

    #[test]
    fn detector_kind_parses_orb() {
        assert_eq!("orb".parse::<DetectorKind>(), Ok(DetectorKind::Orb));
        assert_eq!("ORB".parse::<DetectorKind>(), Ok(DetectorKind::Orb));
    }

    #[test]
    fn detector_kind_parses_sift() {
        assert_eq!("sift".parse::<DetectorKind>(), Ok(DetectorKind::Sift));
        assert_eq!("SIFT".parse::<DetectorKind>(), Ok(DetectorKind::Sift));
    }

    #[test]
    fn detector_kind_rejects_invalid() {
        assert!("bogus".parse::<DetectorKind>().is_err());
        assert!("".parse::<DetectorKind>().is_err());
    }

    #[test]
    fn orb_extractor_finds_features_on_checkerboard() {
        let img = make_checkerboard(200, 200, 20);
        let extractor = OrbExtractor { n_features: 500 };
        let feats = extractor.extract(&img).unwrap();

        assert!(feats.n_keypoints() > 0, "ORB should find corners");
        let desc = feats
            .descriptors_orb
            .as_ref()
            .expect("ORB descriptors present");
        assert_eq!(desc.len(), feats.n_keypoints());
        assert!(feats.descriptors_sift.is_none());
        // Every descriptor is a 256-bit packed binary descriptor.
        assert!(desc.iter().all(|d| d.len() == 32));
    }

    #[test]
    fn sift_extractor_finds_features_on_checkerboard() {
        let img = make_checkerboard(200, 200, 20);
        let extractor = SiftExtractor { n_features: 0 };
        let feats = extractor.extract(&img).unwrap();

        assert!(feats.n_keypoints() > 0, "SIFT should find corners");
        let desc = feats
            .descriptors_sift
            .as_ref()
            .expect("SIFT descriptors present");
        assert_eq!(desc.len(), feats.n_keypoints() * 128);
        assert!(feats.descriptors_orb.is_none());
    }

    #[test]
    fn make_extractor_dispatches_on_kind() {
        let orb = make_extractor(DetectorKind::Orb, 100);
        let sift = make_extractor(DetectorKind::Sift, 0);
        // Smoke-check the extractors run on a tiny checkerboard without panicking.
        let img = make_checkerboard(64, 64, 16);
        assert!(orb.extract(&img).is_ok());
        assert!(sift.extract(&img).is_ok());
    }
}
