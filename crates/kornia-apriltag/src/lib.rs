#![deny(missing_docs)]
//! # Kornia AprilTag

use std::collections::HashMap;

use kornia_image::{
    allocator::{CpuAllocator, ImageAllocator},
    Image, ImageSize,
};

use crate::{
    decoder::{decode_tags, Detection, GrayModelPair, SharpeningBuffer},
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

/// TODO
pub struct DecodeTagsConfig {
    /// TODO
    pub tag_families: Vec<TagFamily>,
    /// TODO
    pub fit_quad_config: FitQuadConfig,
    /// Whether to enable edge refinement before decoding.
    pub refine_edges_enabled: bool,
    /// Sharpening factor applied during decoding.
    pub decode_sharpening: f32,
    /// TODO
    pub normal_border: bool,
    /// TODO
    pub reversed_border: bool,
    /// TODO
    pub min_tag_width: usize,
    /// TODO
    pub sharpening_buffer_len: usize,
    /// TODO
    pub min_white_black_difference: u8,
}

impl DecodeTagsConfig {
    /// TODO
    pub fn new(tag_families: Vec<TagFamily>) -> Self {
        let mut normal_border = false;
        let mut reversed_border = false;
        let mut min_tag_width = usize::MAX;
        let mut min_sharpening_buffer_size = 0;

        tag_families.iter().for_each(|family| {
            if family.width_at_border < min_tag_width {
                min_tag_width = family.width_at_border;
            }
            normal_border |= !family.reversed_border;
            reversed_border |= family.reversed_border;

            if min_sharpening_buffer_size < family.total_width {
                min_sharpening_buffer_size = family.total_width;
            }
        });

        min_tag_width = min_tag_width.min(3);

        Self {
            tag_families,
            fit_quad_config: Default::default(),
            normal_border,
            refine_edges_enabled: true,
            decode_sharpening: 0.25,
            reversed_border,
            min_tag_width,
            sharpening_buffer_len: min_sharpening_buffer_size * min_sharpening_buffer_size,
            min_white_black_difference: 20,
        }
    }

    /// TODO
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

    /// TODO
    pub fn add(&mut self, family: TagFamily) {
        if family.width_at_border < self.min_tag_width {
            self.min_tag_width = family.width_at_border;
        }
        self.normal_border |= !family.reversed_border;
        self.reversed_border |= family.reversed_border;

        let len = family.total_width * family.total_width;
        if self.sharpening_buffer_len < len {
            self.sharpening_buffer_len = len;
        }

        self.tag_families.push(family);
    }
}

/// TODO
pub struct AprilTagDecoder {
    /// TODO
    pub config: DecodeTagsConfig,
    bin_img: Image<Pixel, 1, CpuAllocator>,
    tile_min_max: TileMinMax,
    uf: UnionFind,
    clusters: HashMap<(usize, usize), Vec<GradientInfo>>,
    gray_model_pair: GrayModelPair,
    sharpening_buffer: SharpeningBuffer,
}

impl AprilTagDecoder {
    /// TODO
    #[inline]
    pub fn add(&mut self, family: TagFamily) {
        self.config.add(family);
    }

    /// TODO
    pub fn new(config: DecodeTagsConfig, img_size: ImageSize) -> Result<Self, AprilTagError> {
        let bin_img = Image::from_size_val(img_size, Pixel::Skip, CpuAllocator)?;
        let tile_min_max = TileMinMax::new(img_size, 4);
        let uf = UnionFind::new(img_size.width * img_size.height);
        let sharpening_buffer = SharpeningBuffer::new(config.sharpening_buffer_len);

        Ok(Self {
            config,
            bin_img,
            tile_min_max,
            uf,
            clusters: HashMap::new(),
            gray_model_pair: GrayModelPair::new(),
            sharpening_buffer,
        })
    }

    /// TODO
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
            &self.config,
            &mut self.gray_model_pair,
            &mut self.sharpening_buffer,
        ))
    }

    /// TODO
    pub fn clear(&mut self) {
        self.uf.reset();
        self.clusters.clear();
        self.gray_model_pair.reset();
        self.sharpening_buffer.reset();
    }
}

#[cfg(test)]
mod tests {

    use kornia_image::{allocator::CpuAllocator, Image, ImageSize};
    use kornia_imgproc::color::gray_from_rgb_u8;
    use kornia_io::png::read_image_png_rgba8;

    use crate::{
        family::{TagFamily, TagFamilyKind},
        utils::Point2d,
        AprilTagDecoder, DecodeTagsConfig,
    };

    fn rgb_from_rgba(src: &Image<u8, 4, CpuAllocator>, dst: &mut Image<u8, 3, CpuAllocator>) {
        src.as_slice()
            .chunks(4)
            .zip(dst.as_slice_mut().chunks_mut(3))
            .for_each(|(src, dst)| {
                dst.copy_from_slice(&src[..3]);
            });
    }

    fn scale_image(
        src: &Image<u8, 1, CpuAllocator>,
        dst: &mut Image<u8, 1, CpuAllocator>,
        factor: usize,
    ) {
        let src_slice = src.as_slice();
        let dst_width = dst.width();
        let dst_slice = dst.as_slice_mut();

        for src_y in 0..src.height() {
            for src_x in 0..src.width() {
                let src_idx = src_y * src.width() + src_x;
                let src_px = src_slice[src_idx];

                for dy in 0..factor {
                    let dst_y = src_y * factor + dy;
                    let row_offset = dst_y * dst_width;

                    for dx in 0..factor {
                        let dst_x = src_x * factor + dx;
                        let dst_idx = row_offset + dst_x;

                        dst_slice[dst_idx] = src_px;
                    }
                }
            }
        }
    }

    fn test_tags(
        decoder: &mut AprilTagDecoder,
        original_image_size: ImageSize,
        upscale_image_size: ImageSize,
        expected_tag: TagFamilyKind,
        expected_quads: [Point2d<f32>; 4],
        images_dir: &str,
        file_name_starts_with: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        decoder.config.refine_edges_enabled = false;
        let mut rgb_img = Image::from_size_val(original_image_size, 0u8, CpuAllocator)?;
        let mut grayscale_img = Image::from_size_val(original_image_size, 0u8, CpuAllocator)?;
        let mut grayscale_upscale = Image::from_size_val(upscale_image_size, 0u8, CpuAllocator)?;

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
                let expected_id = expected_id.parse::<u16>()?;

                let original_img = read_image_png_rgba8(file_path)?;
                rgb_from_rgba(&original_img, &mut rgb_img);
                gray_from_rgb_u8(&rgb_img, &mut grayscale_img)?;
                scale_image(&grayscale_img, &mut grayscale_upscale, 5);

                let detection = decoder.decode(&grayscale_upscale)?;

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

                decoder.clear();
            }
        }

        Ok(())
    }

    #[test]
    fn test_tag16_h5() -> Result<(), Box<dyn std::error::Error>> {
        let config = DecodeTagsConfig::new(vec![TagFamily::tag16_h5()]);
        let mut decoder = AprilTagDecoder::new(config, [40, 40].into())?;

        let expected_quad = [
            Point2d { x: 35.0, y: 5.0 },
            Point2d { x: 35.0, y: 35.0 },
            Point2d { x: 5.0, y: 35.0 },
            Point2d { x: 5.0, y: 5.0 },
        ];

        test_tags(
            &mut decoder,
            [8, 8].into(),
            [40, 40].into(),
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
        let mut decoder = AprilTagDecoder::new(config, [45, 45].into())?;

        let expected_quad = [
            Point2d { x: 40.0, y: 5.0 },
            Point2d { x: 40.0, y: 40.0 },
            Point2d { x: 5.0, y: 40.0 },
            Point2d { x: 5.0, y: 5.0 },
        ];

        test_tags(
            &mut decoder,
            [9, 9].into(),
            [45, 45].into(),
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
        let mut decoder = AprilTagDecoder::new(config, [50, 50].into())?;

        let expected_quad = [
            Point2d { x: 45.0, y: 5.0 },
            Point2d { x: 45.0, y: 45.0 },
            Point2d { x: 5.0, y: 45.0 },
            Point2d { x: 5.0, y: 5.0 },
        ];

        test_tags(
            &mut decoder,
            [10, 10].into(),
            [50, 50].into(),
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
        let mut decoder = AprilTagDecoder::new(config, [50, 50].into())?;

        let expected_quad = [
            Point2d { x: 35.0, y: 10.0 },
            Point2d { x: 35.0, y: 35.0 },
            Point2d { x: 10.0, y: 35.0 },
            Point2d { x: 10.0, y: 10.0 },
        ];

        test_tags(
            &mut decoder,
            [9, 9].into(),
            [50, 50].into(),
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
        let mut decoder = AprilTagDecoder::new(config, [55, 55].into())?;

        let expected_quad = [
            Point2d { x: 40.0, y: 15.0 },
            Point2d { x: 40.0, y: 40.0 },
            Point2d { x: 15.0, y: 40.0 },
            Point2d { x: 15.0, y: 15.0 },
        ];

        test_tags(
            &mut decoder,
            [11, 11].into(),
            [55, 55].into(),
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
        let mut decoder = AprilTagDecoder::new(config, [50, 50].into())?;

        let expected_quad = [
            Point2d { x: 40.0, y: 10.0 },
            Point2d { x: 40.0, y: 40.0 },
            Point2d { x: 10.0, y: 40.0 },
            Point2d { x: 10.0, y: 10.0 },
        ];

        test_tags(
            &mut decoder,
            [10, 10].into(),
            [50, 50].into(),
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
        let mut decoder = AprilTagDecoder::new(config, [45, 45].into())?;

        let expected_quad = [
            Point2d { x: 35.0, y: 10.0 },
            Point2d { x: 35.0, y: 35.0 },
            Point2d { x: 10.0, y: 35.0 },
            Point2d { x: 10.0, y: 10.0 },
        ];

        test_tags(
            &mut decoder,
            [9, 9].into(),
            [45, 45].into(),
            TagFamilyKind::TagStandard41H12,
            expected_quad,
            "../../tests/data/apriltag-imgs/tagStandard41h12/",
            "tag41_12_",
        )?;

        Ok(())
    }

    #[test]
    fn test_tagstandard52_h13() -> Result<(), Box<dyn std::error::Error>> {
        let config = DecodeTagsConfig::new(vec![TagFamily::tagstandard52_h13()]);
        let mut decoder = AprilTagDecoder::new(config, [50, 50].into())?;

        let expected_quad = [
            Point2d { x: 40.0, y: 10.0 },
            Point2d { x: 40.0, y: 40.0 },
            Point2d { x: 10.0, y: 40.0 },
            Point2d { x: 10.0, y: 10.0 },
        ];

        test_tags(
            &mut decoder,
            [10, 10].into(),
            [50, 50].into(),
            TagFamilyKind::TagStandard52H13,
            expected_quad,
            "../../tests/data/apriltag-imgs/tagStandard52h13/",
            "tag52_13_",
        )?;

        Ok(())
    }
}
