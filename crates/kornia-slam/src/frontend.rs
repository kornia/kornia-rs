use kornia_image::{allocator::CpuAllocator, Image, ImageError, ImageSize};
use kornia_imgproc::features::orb::{match_orb_descriptors, OrbDetector, OrbFeatures, OrbMatchConfig};

/// Features extracted from a single frame.
#[derive(Debug, Clone)]
pub struct FrameFeatures {
    pub keypoints: Vec<[f32; 2]>,
    pub angles: Vec<f32>,
    pub descriptors: Vec<[u8; 32]>,
}

/// Extract ORB features from a grayscale image buffer.
pub fn extract_features(
    image: &[u8],
    width: usize,
    height: usize,
) -> Result<FrameFeatures, ImageError> {
    let size = ImageSize { width, height };
    let gray = Image::from_size_slice(size, image, CpuAllocator)?;
    let gray_f32 = gray.cast::<f32>()?;

    let detector = OrbDetector::default();
    let orb_feats = detector.detect_and_extract(&gray_f32)?;

    Ok(FrameFeatures {
        keypoints: orb_feats.keypoints_xy,
        angles: orb_feats.orientations,
        descriptors: orb_feats.descriptors,
    })
}

/// Match ORB features between two frames.
pub fn match_features(a: &FrameFeatures, b: &FrameFeatures) -> Vec<(usize, usize)> {
    match_orb_descriptors(
        &a.angles,
        &a.descriptors,
        &b.angles,
        &b.descriptors,
        OrbMatchConfig::default(),
    )
}
