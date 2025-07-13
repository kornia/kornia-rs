#![deny(missing_docs)]
//! # Kornia AprilTag

use std::collections::HashMap;

use kornia_image::{
    allocator::{CpuAllocator, ImageAllocator},
    Image, ImageSize,
};

use crate::{
    decoder::{decode_tags, Detection, GrayModelPair},
    errors::AprilTagError,
    family::TagFamily,
    quad::{fit_quads, FitQuadConfig},
    segmentation::{find_connected_components, find_gradient_clusters, GradientInfo},
    threshold::{adaptive_threshold, TileMinMax},
    union_find::UnionFind,
    utils::Pixel,
};

/// Error types for AprilTag detection.
pub mod errors;

/// Utility functions for AprilTag detection.
pub mod utils;

/// Thresholding utilities for AprilTag detection.
pub mod threshold;

/// image iteration utilities module.
pub(crate) mod iter;

/// Segmentation utilities for AprilTag detection.
pub mod segmentation;

/// Union-find utilities for AprilTag detection.
pub mod union_find;

/// AprilTag family definitions and utilities.
pub mod family;

/// Quad detection utilities for AprilTag detection.
pub mod quad;

/// Decoding utilities for AprilTag detection.
pub mod decoder;

#[derive(Debug, Clone, PartialEq)]
/// Configuration for decoding AprilTags.
pub struct DecodeTagsConfig {
    /// List of tag families to detect.
    pub tag_families: Vec<TagFamily>,
    /// Configuration for quad fitting.
    pub fit_quad_config: FitQuadConfig,
    /// Whether to enable edge refinement before decoding.
    pub refine_edges_enabled: bool,
    /// Sharpening factor applied during decoding.
    pub decode_sharpening: f32,
    /// Whether normal border tags are present.
    pub normal_border: bool,
    /// Whether reversed border tags are present.
    pub reversed_border: bool,
    /// Minimum tag width at border among all families.
    pub min_tag_width: usize,
    /// Minimum difference between white and black pixels for thresholding.
    pub min_white_black_difference: u8,
}

impl DecodeTagsConfig {
    /// Creates a new `DecodeTagsConfig` with the given tag families.
    pub fn new(tag_families: Vec<TagFamily>) -> Self {
        let mut normal_border = false;
        let mut reversed_border = false;
        let mut min_tag_width = usize::MAX;

        tag_families.iter().for_each(|family| {
            if family.width_at_border < min_tag_width {
                min_tag_width = family.width_at_border;
            }
            normal_border |= !family.reversed_border;
            reversed_border |= family.reversed_border;
        });

        if min_tag_width < 3 {
            min_tag_width = 3;
        }

        Self {
            tag_families,
            fit_quad_config: Default::default(),
            normal_border,
            refine_edges_enabled: true,
            decode_sharpening: 0.25,
            reversed_border,
            min_tag_width,
            min_white_black_difference: 20,
        }
    }

    /// Creates a `DecodeTagsConfig` with all supported tag families.
    pub fn all() -> Self {
        let tag_families = vec![
            TagFamily::tag16_h5(),
            TagFamily::tag25_h9(),
            TagFamily::tag36_h10(),
            TagFamily::tag36_h11(),
            TagFamily::tagcircle21_h7(),
            TagFamily::tagcircle49_h12(),
            TagFamily::tagstandard41_h12(),
            TagFamily::tagstandard52_h13(),
            TagFamily::tagcustom48_h12(),
        ];

        Self::new(tag_families)
    }

    /// Adds a tag family to the configuration.
    pub fn add(&mut self, family: TagFamily) {
        if family.width_at_border < self.min_tag_width {
            self.min_tag_width = family.width_at_border;
        }
        self.normal_border |= !family.reversed_border;
        self.reversed_border |= family.reversed_border;

        self.tag_families.push(family);
    }

    /// Resets the sharpening buffer for each tag family in the configuration.
    pub fn reset(&mut self) {
        self.tag_families.iter_mut().for_each(|family| {
            family.sharpening_buffer.reset();
        });
    }
}

/// Decoder for AprilTag detection and decoding.
pub struct AprilTagDecoder {
    /// Configuration for decoding AprilTags.
    pub config: DecodeTagsConfig,
    bin_img: Image<Pixel, 1, CpuAllocator>,
    tile_min_max: TileMinMax,
    uf: UnionFind,
    clusters: HashMap<(usize, usize), Vec<GradientInfo>>,
    gray_model_pair: GrayModelPair,
}

impl AprilTagDecoder {
    /// Adds a tag family to the decoder configuration.
    #[inline]
    pub fn add(&mut self, family: TagFamily) {
        self.config.add(family);
    }

    /// Creates a new `AprilTagDecoder` with the given configuration and image size.
    ///
    /// # Arguments
    ///
    /// * `config` - The configuration for decoding AprilTags.
    /// * `img_size` - The size of the image to be processed.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the new `AprilTagDecoder` or an `AprilTagError`.
    pub fn new(config: DecodeTagsConfig, img_size: ImageSize) -> Result<Self, AprilTagError> {
        let bin_img = Image::from_size_val(img_size, Pixel::Skip, CpuAllocator)?;
        let tile_min_max = TileMinMax::new(img_size, 4);
        let uf = UnionFind::new(img_size.width * img_size.height);

        Ok(Self {
            config,
            bin_img,
            tile_min_max,
            uf,
            clusters: HashMap::new(),
            gray_model_pair: GrayModelPair::new(),
        })
    }

    /// Decodes AprilTags from the provided grayscale image.
    ///
    /// # Arguments
    ///
    /// * `src` - The source grayscale image to decode tags from.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing a vector of `Detection` or an `AprilTagError`.
    ///
    /// # Note
    ///
    /// If you are running this method multiple times on the same decoder instance,
    /// you should call [`AprilTagDecoder::clear`] between runs to reset internal state.
    pub fn decode<A: ImageAllocator>(
        &mut self,
        src: &Image<u8, 1, A>,
    ) -> Result<Vec<Detection>, AprilTagError> {
        // TODO: Add support for downscaling image

        // Step 1: Adaptive Threshold
        adaptive_threshold(
            src,
            &mut self.bin_img,
            &mut self.tile_min_max,
            self.config.min_white_black_difference,
        )?;

        // Step 2(a): Find Connected Components
        find_connected_components(&self.bin_img, &mut self.uf)?;

        // Step 2(b): Find Clusters
        find_gradient_clusters(&self.bin_img, &mut self.uf, &mut self.clusters);

        // Step 3: Quad Fitting
        let mut quads = fit_quads(&self.bin_img, &mut self.clusters, &self.config);

        // Step 4: Tag Decoding
        Ok(decode_tags(
            src,
            &mut quads,
            &mut self.config,
            &mut self.gray_model_pair,
        ))
    }

    /// Clears the internal state of the decoder for reuse.
    pub fn clear(&mut self) {
        self.uf.reset();
        self.clusters.clear();
        self.gray_model_pair.reset();
        self.config.reset();
    }

    /// Returns a slice of tag families configured for detection.
    pub fn tag_families(&self) -> &[TagFamily] {
        &self.config.tag_families
    }
}

#[cfg(test)]
mod tests {
    use kornia_io::png::read_image_png_mono8;

    use crate::{
        family::{TagFamily, TagFamilyKind},
        utils::Point2d,
        AprilTagDecoder, DecodeTagsConfig,
    };

    fn test_tags(
        decoder: &mut AprilTagDecoder,
        expected_tag: TagFamilyKind,
        expected_quads: [Point2d<f32>; 4],
        images_dir: &str,
        file_name_starts_with: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let tag_images = std::fs::read_dir(images_dir)?;

        for img in tag_images {
            let img = img?;
            let file_name = img.file_name();
            let file_name = file_name
                .to_str()
                .ok_or("Failed to convert file name to str")?;

            if file_name.starts_with(file_name_starts_with) {
                let file_path = img.path();

                let expected_id = file_name.strip_prefix(file_name_starts_with).unwrap();
                let expected_id = expected_id.strip_suffix(".png").unwrap();
                let Ok(expected_id) = expected_id.parse::<u16>() else {
                    // Currently we only support decoding id upto 65535 (u16::MAX) while some tag families
                    // like `TagCircle49H12` can support more than that.
                    continue;
                };

                if expected_id == u16::MAX {
                    continue;
                }

                let original_img = read_image_png_mono8(file_path)?;

                decoder.clear();
                let detection = decoder.decode(&original_img)?;

                assert_eq!(detection.len(), 1, "Tag: {}", file_name);
                let detection = &detection[0];

                assert_eq!(detection.id, expected_id);
                assert_eq!(detection.tag_family_kind, expected_tag);

                for (point, expected) in detection.quad.corners.iter().zip(expected_quads.iter()) {
                    assert!(
                        (point.y - expected.y).abs() <= 0.001,
                        "Tag: {}, Got y: {}, Expected: {}",
                        file_name,
                        point.y,
                        expected.y
                    );
                    assert!(
                        (point.x - expected.x).abs() <= 0.001,
                        "Tag: {}, Got x: {}, Expected: {}",
                        file_name,
                        point.x,
                        expected.x
                    );
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_tag16_h5() -> Result<(), Box<dyn std::error::Error>> {
        let config = DecodeTagsConfig::new(vec![TagFamily::tag16_h5()]);
        let mut decoder = AprilTagDecoder::new(config, [50, 50].into())?;

        let expected_quad = [
            Point2d { x: 40.0, y: 10.0 },
            Point2d { x: 40.0, y: 40.0 },
            Point2d { x: 10.0, y: 40.0 },
            Point2d { x: 10.0, y: 10.0 },
        ];

        test_tags(
            &mut decoder,
            TagFamilyKind::Tag16H5,
            expected_quad,
            "../../tests/data/apriltag-imgs/tag16h5/",
            "tag16_05_",
        )?;

        Ok(())
    }

    #[test]
    fn test_tag25_h9() -> Result<(), Box<dyn std::error::Error>> {
        let config = DecodeTagsConfig::new(vec![TagFamily::tag25_h9()]);
        let mut decoder = AprilTagDecoder::new(config, [55, 55].into())?;

        let expected_quad = [
            Point2d { x: 45.0, y: 10.0 },
            Point2d { x: 45.0, y: 45.0 },
            Point2d { x: 10.0, y: 45.0 },
            Point2d { x: 10.0, y: 10.0 },
        ];

        test_tags(
            &mut decoder,
            TagFamilyKind::Tag25H9,
            expected_quad,
            "../../tests/data/apriltag-imgs/tag25h9/",
            "tag25_09_",
        )?;

        Ok(())
    }

    #[test]
    fn test_tag36_h11() -> Result<(), Box<dyn std::error::Error>> {
        let config = DecodeTagsConfig::new(vec![TagFamily::tag36_h11()]);
        let mut decoder = AprilTagDecoder::new(config, [60, 60].into())?;

        let expected_quad = [
            Point2d { x: 50.0, y: 10.0 },
            Point2d { x: 50.0, y: 50.0 },
            Point2d { x: 10.0, y: 50.0 },
            Point2d { x: 10.0, y: 10.0 },
        ];

        test_tags(
            &mut decoder,
            TagFamilyKind::Tag36H11,
            expected_quad,
            "../../tests/data/apriltag-imgs/tag36h11/",
            "tag36_11_",
        )?;

        Ok(())
    }

    #[test]
    fn test_tagcircle21h7() -> Result<(), Box<dyn std::error::Error>> {
        let config = DecodeTagsConfig::new(vec![TagFamily::tagcircle21_h7()]);
        let mut decoder = AprilTagDecoder::new(config, [55, 55].into())?;

        let expected_quad = [
            Point2d { x: 40.0, y: 15.0 },
            Point2d { x: 40.0, y: 40.0 },
            Point2d { x: 15.0, y: 40.0 },
            Point2d { x: 15.0, y: 15.0 },
        ];

        test_tags(
            &mut decoder,
            TagFamilyKind::TagCircle21H7,
            expected_quad,
            "../../tests/data/apriltag-imgs/tagCircle21h7/",
            "tag21_07_",
        )?;

        Ok(())
    }

    #[test]
    fn test_tagcircle49h12() -> Result<(), Box<dyn std::error::Error>> {
        let config = DecodeTagsConfig::new(vec![TagFamily::tagcircle49_h12()]);
        let mut decoder = AprilTagDecoder::new(config, [65, 65].into())?;
        decoder.config.refine_edges_enabled = false;

        let expected_quad = [
            Point2d { x: 45.0, y: 20.0 },
            Point2d { x: 45.0, y: 45.0 },
            Point2d { x: 20.0, y: 45.0 },
            Point2d { x: 20.0, y: 20.0 },
        ];

        test_tags(
            &mut decoder,
            TagFamilyKind::TagCircle49H12,
            expected_quad,
            "../../tests/data/apriltag-imgs/tagCircle49h12/",
            "tag49_12_",
        )?;

        Ok(())
    }

    #[test]
    fn test_tagcustom48_h12() -> Result<(), Box<dyn std::error::Error>> {
        let config = DecodeTagsConfig::new(vec![TagFamily::tagcustom48_h12()]);
        let mut decoder = AprilTagDecoder::new(config, [60, 60].into())?;

        let expected_quad = [
            Point2d { x: 45.0, y: 15.0 },
            Point2d { x: 45.0, y: 45.0 },
            Point2d { x: 15.0, y: 45.0 },
            Point2d { x: 15.0, y: 15.0 },
        ];

        test_tags(
            &mut decoder,
            TagFamilyKind::TagCustom48H12,
            expected_quad,
            "../../tests/data/apriltag-imgs/tagCustom48h12/",
            "tag48_12_",
        )?;

        Ok(())
    }

    #[test]
    fn test_tagstandard41_h12() -> Result<(), Box<dyn std::error::Error>> {
        let config = DecodeTagsConfig::new(vec![TagFamily::tagstandard41_h12()]);
        let mut decoder = AprilTagDecoder::new(config, [55, 55].into())?;

        let expected_quad = [
            Point2d { x: 40.0, y: 15.0 },
            Point2d { x: 40.0, y: 40.0 },
            Point2d { x: 15.0, y: 40.0 },
            Point2d { x: 15.0, y: 15.0 },
        ];

        test_tags(
            &mut decoder,
            TagFamilyKind::TagStandard41H12,
            expected_quad,
            "../../tests/data/apriltag-imgs/tagStandard41h12/",
            "tag41_12_",
        )?;

        Ok(())
    }

    #[test]
    #[ignore]
    fn test_tagstandard52_h13() -> Result<(), Box<dyn std::error::Error>> {
        let config = DecodeTagsConfig::new(vec![TagFamily::tagstandard52_h13()]);
        let mut decoder = AprilTagDecoder::new(config, [60, 60].into())?;

        let expected_quad = [
            Point2d { x: 45.0, y: 15.0 },
            Point2d { x: 45.0, y: 45.0 },
            Point2d { x: 15.0, y: 45.0 },
            Point2d { x: 15.0, y: 15.0 },
        ];

        test_tags(
            &mut decoder,
            TagFamilyKind::TagStandard52H13,
            expected_quad,
            "../../tests/data/apriltag-imgs/tagStandard52h13/",
            "tag52_13_",
        )?;

        Ok(())
    }
}
