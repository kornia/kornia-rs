#![deny(missing_docs)]
// SIMD/perf-oriented kernels: explicit (slice, width, height, …) argument lists and
// index-based loops are intentional and clearer than the clippy-preferred forms.
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
//! # Kornia AprilTag

use rustc_hash::FxHashMap;

use crate::{
    decoder::{decode_tags, dedup_detections, Detection},
    errors::AprilTagError,
    family::{TagFamily, TagFamilyKind},
    quad::{fit_quads, FitQuadConfig},
    rle_cc::RleCC,
    segmentation::{find_gradient_clusters_with_cache, GradientInfo},
    threshold::{adaptive_threshold_with_split, TileMinMax},
    utils::Pixel,
};
use kornia_image::{Image, ImageSize};

/// Error types for AprilTag detection.
pub mod errors;

/// Architecture-dispatched SIMD kernels (NEON / AVX2 / scalar) + feature detection.
pub mod ops;

/// Utility functions for AprilTag detection.
pub mod utils;

/// Thresholding utilities for AprilTag detection.
pub mod threshold;

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

/// RLE-based connected components (internal).
pub(crate) mod rle_cc;

/// AprilTag 6-DOF pose estimation (built on kornia-3d geometry primitives).
pub mod pose;

/// Rigid AprilGrid target: a planar grid of AprilTags with a known metric layout.
pub mod board;

/// Configuration for decoding AprilTags.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodeTagsConfig {
    /// List of tag families to detect.
    pub tag_families: Vec<TagFamilyKind>,
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
    /// Downscale factor for input images.
    pub downscale_factor: usize,
    /// Fraction between each tile's local min and max at which the adaptive binarization
    /// splits black from white (`threshold = min + (max - min) * threshold_split`).
    ///
    /// `0.5` is the classic AprilTag midpoint. A lower value (default `0.33`) biases toward
    /// white, preserving thin bright quiet-zone margins around small or glary tags so their
    /// black border does not merge with adjacent dark regions. Clamped to `[0.0, 1.0]`.
    pub threshold_split: f32,
}

impl DecodeTagsConfig {
    /// Creates a new `DecodeTagsConfig` with the given tag family kinds.
    pub fn new(tag_family_kinds: Vec<TagFamilyKind>) -> Result<Self, AprilTagError> {
        const DEFAULT_DOWNSCALE_FACTOR: usize = 2;
        // Slightly below the classic 0.5 midpoint: biases binarization toward white so thin
        // bright margins around small/glary tags survive and the black border stays isolated.
        const DEFAULT_THRESHOLD_SPLIT: f32 = 0.33;

        if tag_family_kinds.is_empty() {
            return Err(AprilTagError::EmptyTagFamilies);
        }

        let mut tag_families = Vec::with_capacity(tag_family_kinds.len());
        let mut normal_border = false;
        let mut reversed_border = false;
        let mut min_tag_width = usize::MAX;

        for family_kind in tag_family_kinds {
            let family: TagFamily = family_kind.clone().try_into()?;
            if family.width_at_border < min_tag_width {
                min_tag_width = family.width_at_border;
            }
            normal_border |= !family.reversed_border;
            reversed_border |= family.reversed_border;

            tag_families.push(family_kind);
        }
        min_tag_width /= DEFAULT_DOWNSCALE_FACTOR;

        if min_tag_width < 3 {
            min_tag_width = 3;
        }

        Ok(Self {
            tag_families,
            fit_quad_config: Default::default(),
            normal_border,
            refine_edges_enabled: true,
            decode_sharpening: 0.25,
            reversed_border,
            min_tag_width,
            min_white_black_difference: 5,
            downscale_factor: DEFAULT_DOWNSCALE_FACTOR,
            threshold_split: DEFAULT_THRESHOLD_SPLIT,
        })
    }

    /// Creates a `DecodeTagsConfig` with all supported tag families.
    pub fn all() -> Result<Self, AprilTagError> {
        Self::new(TagFamilyKind::all())
    }

    /// Adds a tag family to the configuration.
    pub fn add(&mut self, kind: TagFamilyKind) {
        // Inspect properties
        let (width, reversed) = match &kind {
            TagFamilyKind::Custom(arc) => (arc.width_at_border, arc.reversed_border),
            _ => match TagFamily::try_from(&kind) {
                Ok(family) => (family.width_at_border, family.reversed_border),
                Err(err) => panic!("builtin tag family conversion failed: {err}"),
            },
        };

        let search_width = (width / self.downscale_factor).max(3);
        if search_width < self.min_tag_width {
            self.min_tag_width = search_width;
        }
        self.normal_border |= !reversed;
        self.reversed_border |= reversed;

        self.tag_families.push(kind);
    }
}

/// Stride-based decimation matching C's `image_u8_decimate` (top-left pixel of each factor×factor block).
fn stride_decimate(src: &Image<u8, 1>, dst: &mut Image<u8, 1>, factor: usize) {
    let src_w = src.width();
    let dst_w = dst.width();
    let dst_h = dst.height();
    let src_data = src.as_slice();
    let dst_data = dst.as_slice_mut();

    #[cfg(target_arch = "aarch64")]
    if factor == 2 {
        use std::arch::aarch64::*;
        for sy in 0..dst_h {
            let src_row = &src_data[(sy * 2) * src_w..];
            let dst_row = &mut dst_data[sy * dst_w..];
            let mut sx = 0usize;
            // Process 16 output pixels (32 source pixels) per iteration.
            while sx + 16 <= dst_w && (sx * 2 + 32) <= src_w {
                let deinterleaved = unsafe { vld2q_u8(src_row.as_ptr().add(sx * 2)) };
                // val0 = even-indexed source pixels = the ones we want (factor=2, top-left)
                unsafe {
                    vst1q_u8(dst_row.as_mut_ptr().add(sx), deinterleaved.0);
                }
                sx += 16;
            }
            // Scalar tail.
            while sx < dst_w {
                dst_row[sx] = src_row[sx * 2];
                sx += 1;
            }
        }
        return;
    }

    for sy in 0..dst_h {
        for sx in 0..dst_w {
            dst_data[sy * dst_w + sx] = src_data[(sy * factor) * src_w + sx * factor];
        }
    }
}

/// Decoder for AprilTag detection and decoding.
pub struct AprilTagDecoder {
    config: DecodeTagsConfig,
    cached_families: Vec<(TagFamilyKind, TagFamily)>,
    downscale_img: Option<Image<u8, 1>>,
    bin_img: Image<Pixel, 1>,
    tile_min_max: TileMinMax,
    rle_cc: RleCC,
    /// Pre-allocated buffer reused each frame; filled by `rle_cc.process`.
    /// Encoding: `u32::MAX` = skip/small component; otherwise run-root index as u32.
    rep_cache: Vec<u32>,
    clusters: Vec<FxHashMap<(usize, usize), Vec<GradientInfo>>>,
}

impl AprilTagDecoder {
    /// Returns a reference to the decoder configuration.
    #[inline]
    pub fn config(&self) -> &DecodeTagsConfig {
        &self.config
    }

    /// Adds a tag family to the decoder configuration and cached families.
    #[inline]
    pub fn add(&mut self, kind: TagFamilyKind) {
        if let Ok(family) = TagFamily::try_from(&kind) {
            self.cached_families.push((kind.clone(), family));
        }
        self.config.add(kind);
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
        let (img_size, downscale_img) = if config.downscale_factor <= 1 {
            (img_size, None)
        } else {
            // Match C's image_u8_decimate: swidth = 1 + (w-1)/factor (ceiling division).
            let new_size = ImageSize {
                width: 1 + (img_size.width - 1) / config.downscale_factor,
                height: 1 + (img_size.height - 1) / config.downscale_factor,
            };

            (new_size, Some(Image::from_size_val(new_size, 0)?))
        };

        // Build the tag family cache once
        let cached_families: Vec<(TagFamilyKind, TagFamily)> = config
            .tag_families
            .iter()
            .filter_map(|kind| {
                TagFamily::try_from(kind)
                    .ok()
                    .map(|family| (kind.clone(), family))
            })
            .collect();

        let n_pixels = img_size.width * img_size.height;
        let bin_img = Image::from_size_val(img_size, Pixel::Skip)?;
        let tile_min_max = TileMinMax::new(img_size, 4);
        let rle_cc = RleCC::new(img_size.height, img_size.width);

        Ok(Self {
            config,
            cached_families,
            downscale_img,
            bin_img,
            tile_min_max,
            rle_cc,
            rep_cache: vec![u32::MAX; n_pixels],
            clusters: Vec::new(),
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
    pub fn decode(&mut self, src: &Image<u8, 1>) -> Result<Vec<Detection>, AprilTagError> {
        if let Some(downscale_img) = self.downscale_img.as_mut() {
            // Stride-based subsample matching C's image_u8_decimate: dst[sy][sx] = src[sy*f][sx*f].
            stride_decimate(src, downscale_img, self.config.downscale_factor);

            // Step 1: Adaptive Threshold
            adaptive_threshold_with_split(
                downscale_img,
                &mut self.bin_img,
                &mut self.tile_min_max,
                self.config.min_white_black_difference,
                self.config.threshold_split,
            )?;
        } else {
            // Step 1: Adaptive Threshold
            adaptive_threshold_with_split(
                src,
                &mut self.bin_img,
                &mut self.tile_min_max,
                self.config.min_white_black_difference,
                self.config.threshold_split,
            )?;
        }

        // Step 2(a): Find Connected Components + path-compress + build rep_cache (one fused pass).
        self.rle_cc.process(&self.bin_img, &mut self.rep_cache, 25);

        // Step 2(b): Find Clusters (NEON fast-path on aarch64)
        self.clusters = find_gradient_clusters_with_cache(&self.bin_img, &self.rep_cache);

        // Step 3: Quad Fitting
        let mut quads = fit_quads(&self.bin_img, &self.clusters, &self.config);

        // Step 4: Tag Decoding
        // D4 fix: refine_edges search range matches C's (quad_decimate + 1).
        let refine_edges_range = self.config.downscale_factor as f32 + 1.0;
        let all = decode_tags(
            src,
            &mut quads,
            &self.cached_families,
            self.config.refine_edges_enabled,
            self.config.decode_sharpening,
            refine_edges_range,
        );
        Ok(dedup_detections(all))
    }

    /// Decodes all valid tags in the image without deduplication.
    ///
    /// Returns every detection (including multiple copies of the same id if several quads
    /// decode to it). Use this when you need the full candidate set — e.g. for parity
    /// testing where you want to find the detection closest to a known reference.
    pub fn decode_all(&mut self, src: &Image<u8, 1>) -> Result<Vec<Detection>, AprilTagError> {
        if let Some(downscale_img) = self.downscale_img.as_mut() {
            stride_decimate(src, downscale_img, self.config.downscale_factor);
            adaptive_threshold_with_split(
                downscale_img,
                &mut self.bin_img,
                &mut self.tile_min_max,
                self.config.min_white_black_difference,
                self.config.threshold_split,
            )?;
        } else {
            adaptive_threshold_with_split(
                src,
                &mut self.bin_img,
                &mut self.tile_min_max,
                self.config.min_white_black_difference,
                self.config.threshold_split,
            )?;
        }
        self.rle_cc.process(&self.bin_img, &mut self.rep_cache, 25);
        self.clusters = find_gradient_clusters_with_cache(&self.bin_img, &self.rep_cache);
        let mut quads = fit_quads(&self.bin_img, &self.clusters, &self.config);
        let refine_edges_range = self.config.downscale_factor as f32 + 1.0;
        Ok(decode_tags(
            src,
            &mut quads,
            &self.cached_families,
            self.config.refine_edges_enabled,
            self.config.decode_sharpening,
            refine_edges_range,
        ))
    }

    /// Decodes tags and returns per-stage timing (µs) for profiling.
    /// Returns `(detections, [decimate, threshold, conn_comp, gradient, fit_quads, decode_tags])`.
    pub fn decode_timed(
        &mut self,
        src: &Image<u8, 1>,
    ) -> Result<(Vec<Detection>, [u64; 6]), AprilTagError> {
        let mut us = [0u64; 6];
        let t = std::time::Instant::now();
        if let Some(downscale_img) = self.downscale_img.as_mut() {
            stride_decimate(src, downscale_img, self.config.downscale_factor);
            us[0] = t.elapsed().as_micros() as u64;
            let t = std::time::Instant::now();
            adaptive_threshold_with_split(
                downscale_img,
                &mut self.bin_img,
                &mut self.tile_min_max,
                self.config.min_white_black_difference,
                self.config.threshold_split,
            )?;
            us[1] = t.elapsed().as_micros() as u64;
        } else {
            us[0] = 0;
            let t = std::time::Instant::now();
            adaptive_threshold_with_split(
                src,
                &mut self.bin_img,
                &mut self.tile_min_max,
                self.config.min_white_black_difference,
                self.config.threshold_split,
            )?;
            us[1] = t.elapsed().as_micros() as u64;
        }
        let t = std::time::Instant::now();
        self.rle_cc.process(&self.bin_img, &mut self.rep_cache, 25);
        us[2] = t.elapsed().as_micros() as u64;
        let t = std::time::Instant::now();
        self.clusters = find_gradient_clusters_with_cache(&self.bin_img, &self.rep_cache);
        us[3] = t.elapsed().as_micros() as u64;
        let t = std::time::Instant::now();
        let mut quads = fit_quads(&self.bin_img, &self.clusters, &self.config);
        us[4] = t.elapsed().as_micros() as u64;
        let refine_edges_range = self.config.downscale_factor as f32 + 1.0;
        let t = std::time::Instant::now();
        let all = decode_tags(
            src,
            &mut quads,
            &self.cached_families,
            self.config.refine_edges_enabled,
            self.config.decode_sharpening,
            refine_edges_range,
        );
        us[5] = t.elapsed().as_micros() as u64;
        Ok((dedup_detections(all), us))
    }

    /// Clears the internal state of the decoder for reuse.
    pub fn clear(&mut self) {
        // RleCC resets itself at the start of each process() call — no-op here.
        self.clusters.clear();
    }

    /// Returns a slice of tag families configured for detection.
    pub fn tag_families(&self) -> &[TagFamilyKind] {
        &self.config.tag_families
    }
}

#[cfg(test)]
mod tests {
    use kornia_io::png::read_image_png_mono8;

    use crate::{errors::AprilTagError, family::TagFamilyKind, AprilTagDecoder, DecodeTagsConfig};
    use kornia_algebra::Vec2F32;

    fn test_tags(
        decoder: &mut AprilTagDecoder,
        expected_tag: TagFamilyKind,
        expected_quads: [Vec2F32; 4],
        images_dir: &str,
        file_name_starts_with: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // The tag images come from the `apriltag-imgs` git submodule; skip cleanly
        // when it isn't initialized rather than failing with a NotFound error.
        if !std::path::Path::new(images_dir).exists() {
            eprintln!("skipping: tag image dir '{images_dir}' not found (run `git submodule update --init`)");
            return Ok(());
        }
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
                let detection = decoder.decode(&original_img)?;

                assert_eq!(detection.len(), 1, "Tag: {file_name}");
                let detection = &detection[0];

                assert_eq!(detection.id, expected_id);
                assert_eq!(detection.tag_family_kind, expected_tag);

                // Tolerance widened to 0.3px to accommodate C-equivalent corner scaling
                // ((c - 0.5) * factor + 0.5) which shifts initial corners by -0.5px before refine.
                for (point, expected) in detection.quad.corners.iter().zip(expected_quads.iter()) {
                    assert!(
                        (point.y - expected.y).abs() <= 0.3,
                        "Tag: {}, Got y: {}, Expected: {}",
                        file_name,
                        point.y,
                        expected.y
                    );
                    assert!(
                        (point.x - expected.x).abs() <= 0.3,
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
        let config = DecodeTagsConfig::new(vec![TagFamilyKind::Tag16H5])?;
        let mut decoder = AprilTagDecoder::new(config, [50, 50].into())?;

        let expected_quad = [
            Vec2F32::new(40.0, 10.0),
            Vec2F32::new(40.0, 40.0),
            Vec2F32::new(10.0, 40.0),
            Vec2F32::new(10.0, 10.0),
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
    fn test_empty_tag_families_config_rejected() {
        let err = DecodeTagsConfig::new(Vec::new()).unwrap_err();
        assert!(matches!(err, AprilTagError::EmptyTagFamilies));
    }

    #[test]
    fn test_tag25_h9() -> Result<(), Box<dyn std::error::Error>> {
        let config = DecodeTagsConfig::new(vec![TagFamilyKind::Tag25H9])?;
        let mut decoder = AprilTagDecoder::new(config, [55, 55].into())?;

        let expected_quad = [
            Vec2F32::new(45.0, 10.0),
            Vec2F32::new(45.0, 45.0),
            Vec2F32::new(10.0, 45.0),
            Vec2F32::new(10.0, 10.0),
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
        let config = DecodeTagsConfig::new(vec![TagFamilyKind::Tag36H11])?;
        let mut decoder = AprilTagDecoder::new(config, [60, 60].into())?;

        let expected_quad = [
            Vec2F32::new(50.0, 10.0),
            Vec2F32::new(50.0, 50.0),
            Vec2F32::new(10.0, 50.0),
            Vec2F32::new(10.0, 10.0),
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
        let config = DecodeTagsConfig::new(vec![TagFamilyKind::TagCircle21H7])?;
        let mut decoder = AprilTagDecoder::new(config, [55, 55].into())?;

        let expected_quad = [
            Vec2F32::new(40.0, 15.0),
            Vec2F32::new(40.0, 40.0),
            Vec2F32::new(15.0, 40.0),
            Vec2F32::new(15.0, 15.0),
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
        let config = DecodeTagsConfig::new(vec![TagFamilyKind::TagCircle49H12])?;
        let mut decoder = AprilTagDecoder::new(config, [65, 65].into())?;

        let expected_quad = [
            Vec2F32::new(45.0, 20.0),
            Vec2F32::new(45.0, 45.0),
            Vec2F32::new(20.0, 45.0),
            Vec2F32::new(20.0, 20.0),
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
        let config = DecodeTagsConfig::new(vec![TagFamilyKind::TagCustom48H12])?;
        let mut decoder = AprilTagDecoder::new(config, [60, 60].into())?;

        let expected_quad = [
            Vec2F32::new(45.0, 15.0),
            Vec2F32::new(45.0, 45.0),
            Vec2F32::new(15.0, 45.0),
            Vec2F32::new(15.0, 15.0),
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
        let config = DecodeTagsConfig::new(vec![TagFamilyKind::TagStandard41H12])?;
        let mut decoder = AprilTagDecoder::new(config, [55, 55].into())?;

        let expected_quad = [
            Vec2F32::new(40.0, 15.0),
            Vec2F32::new(40.0, 40.0),
            Vec2F32::new(15.0, 40.0),
            Vec2F32::new(15.0, 15.0),
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
    fn test_tagstandard52_h13() -> Result<(), Box<dyn std::error::Error>> {
        let config = DecodeTagsConfig::new(vec![TagFamilyKind::TagStandard52H13])?;
        let mut decoder = AprilTagDecoder::new(config, [60, 60].into())?;

        let expected_quad = [
            Vec2F32::new(45.0, 15.0),
            Vec2F32::new(45.0, 45.0),
            Vec2F32::new(15.0, 45.0),
            Vec2F32::new(15.0, 15.0),
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
