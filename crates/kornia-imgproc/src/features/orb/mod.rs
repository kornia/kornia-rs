mod extractor;
mod matcher;
mod pattern;

pub use extractor::{OrbDetector, OrbFeatures};
pub use matcher::{match_orb_descriptors, OrbMatchConfig};

#[cfg(all(test, feature = "opencv_bench"))]
mod opencv_tests {
    use super::*;
    use kornia_image::Image;
    use kornia_tensor::CpuAllocator;
    use opencv::{
        core::{no_array, Mat, Vector},
        features2d::{ORB_ScoreType, ORB},
        imgcodecs,
        prelude::*,
    };

    fn u8_to_f32_image(src: &Image<u8, 1, CpuAllocator>) -> Image<f32, 1, CpuAllocator> {
        let mut dst = Image::from_size_val(src.size(), 0.0, CpuAllocator).unwrap();
        src.as_slice()
            .iter()
            .zip(dst.as_slice_mut())
            .for_each(|(&s, d)| *d = s as f32 / 255.0);
        dst
    }

    fn keypoints_from_opencv(kps: &Vector<opencv::core::KeyPoint>) -> Vec<(f32, f32)> {
        kps.iter().map(|kp| (kp.pt().y, kp.pt().x)).collect()
    }

    fn has_nearby_keypoint(kps: &[(f32, f32)], target: (f32, f32), radius: f32) -> bool {
        let r2 = radius * radius;
        kps.iter().any(|&(y, x)| {
            let dy = y - target.0;
            let dx = x - target.1;
            dy * dy + dx * dx <= r2
        })
    }

    #[test]
    fn test_orb_compare_with_opencv() -> Result<(), Box<dyn std::error::Error>> {
        let img_rgb = kornia_io::jpeg::read_image_jpeg_rgb8("../../tests/data/dog.jpeg")?;
        let mut img_gray = Image::from_size_val(img_rgb.size(), 0u8, CpuAllocator)?;
        crate::color::gray_from_rgb_u8(&img_rgb, &mut img_gray)?;
        let img_f32 = u8_to_f32_image(&img_gray);

        // Kornia ORB.
        let orb = OrbDetector::default();
        let (kps, scales, orientations, _responses) = orb.detect(&img_f32)?;
        let (descriptors, _mask) = orb.extract(&img_f32, &kps, &scales, &orientations)?;

        let mut kps_xy: Vec<(f32, f32)> = kps.iter().copied().collect();

        // OpenCV ORB.
        let mat = imgcodecs::imread("../../tests/data/dog.jpeg", imgcodecs::IMREAD_GRAYSCALE)?;

        let mut orb_cv = ORB::create(500, 1.2, 8, 31, 0, 2, ORB_ScoreType::HARRIS_SCORE, 31, 20)?;
        let mut kps_cv = Vector::<opencv::core::KeyPoint>::new();
        let mut desc_cv = Mat::default();
        orb_cv.detect_and_compute(&mat, &no_array(), &mut kps_cv, &mut desc_cv, false)?;

        let kps_cv_xy = keypoints_from_opencv(&kps_cv);

        assert!(kps_xy.len() >= 100, "kornia ORB found too few keypoints");
        assert!(kps_cv_xy.len() >= 100, "OpenCV ORB found too few keypoints");

        kps_xy.truncate(500);
        let kps_cv_xy = &kps_cv_xy[..kps_cv_xy.len().min(1000)];

        let mut overlap = 0usize;
        for &kp in &kps_xy {
            if has_nearby_keypoint(kps_cv_xy, kp, 3.0) {
                overlap += 1;
            }
        }

        let overlap_ratio = overlap as f32 / kps_xy.len() as f32;
        assert!(overlap_ratio >= 0.15, "overlap too low: {overlap_ratio:.2}");

        // Descriptor size comparison (both use packed 32-byte descriptors).
        assert_eq!(desc_cv.cols(), 32);
        if !descriptors.is_empty() {
            assert_eq!(descriptors[0].len(), 32);
        }

        Ok(())
    }
}
