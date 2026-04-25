use kornia_image::{allocator::ImageAllocator, Image, ImageError, ImageSize};
use kornia_tensor::CpuAllocator;

use crate::{
    features::FastDetector,
    filter::gaussian_blur,
    padding::{spatial_padding, Padding2D, PaddingMode},
};

use super::pattern::{POS0, POS1};

/// ORB features extracted from a single frame.
#[derive(Debug, Clone)]
pub struct OrbFeatures {
    /// Keypoints as `[col, row]` in pixel coordinates.
    pub keypoints_xy: Vec<[f32; 2]>,
    /// Keypoint orientation angles (radians).
    pub orientations: Vec<f32>,
    /// Binary descriptors (256-bit, packed as 32 bytes each).
    pub descriptors: Vec<[u8; 32]>,
    /// Scale factor (downscale^octave) at which each keypoint was detected.
    pub scales: Vec<f32>,
}

/// Which FAST threshold produced a debug candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrbDebugThresholdKind {
    /// Initial FAST threshold.
    Initial,
    /// Fallback minimum FAST threshold.
    Minimum,
}

/// Debug record for a single ORB candidate or survivor.
#[derive(Debug, Clone)]
pub struct OrbDebugCandidate {
    /// Keypoint x coordinate in original-image pixel space.
    pub x: f32,
    /// Keypoint y coordinate in original-image pixel space.
    pub y: f32,
    /// Detector response used for ranking.
    pub response: f32,
    /// Pyramid octave.
    pub octave: usize,
    /// Cell row used during cell-local FAST detection.
    pub cell_row: i32,
    /// Cell column used during cell-local FAST detection.
    pub cell_col: i32,
    /// Threshold source for this candidate.
    pub threshold_kind: OrbDebugThresholdKind,
}

/// Debug data for a single octave.
#[derive(Debug, Clone)]
pub struct OrbDebugOctave {
    /// Pyramid octave.
    pub octave: usize,
    /// Octave image size before padding.
    pub image_size: ImageSize,
    /// Raw FAST candidates before nonmax suppression.
    pub raw_candidates: Vec<OrbDebugCandidate>,
    /// Post-NMS FAST candidates before octree distribution.
    pub candidates: Vec<OrbDebugCandidate>,
    /// Final survivors after octree distribution.
    pub survivors: Vec<OrbDebugCandidate>,
}

/// Debug data for one ORB detection pass.
#[derive(Debug, Clone)]
pub struct OrbDebugFrame {
    /// Per-octave debug data.
    pub octaves: Vec<OrbDebugOctave>,
}

/// ORB (Oriented FAST and Rotated BRIEF) feature detector and descriptor extractor.
///
/// Detects keypoints using a multi-scale FAST detector with Harris response scoring
/// and octree-based spatial distribution, then computes rotation-aware BRIEF descriptors.
pub struct OrbDetector {
    /// Maximum number of keypoints to retain (default 500).
    pub n_keypoints: usize,
    /// Number of contiguous pixels required by the FAST detector (default 9).
    pub fast_n: usize,
    /// Initial FAST threshold (higher, for speed). Used first in each grid cell.
    pub ini_fast_threshold: f32,
    /// Minimum FAST threshold (lower, fallback). Used if no keypoints found with ini threshold.
    pub min_fast_threshold: f32,
    /// Harris corner response sensitivity parameter (default 0.04).
    pub harris_k: f32,
    /// Scale factor between pyramid levels (default 1.2).
    pub downscale: f32,
    /// Number of pyramid levels (default 8).
    pub n_scales: usize,
    /// Grid cell size for two-tier FAST detection (default 35 like ORB-SLAM3).
    pub cell_size: usize,
}

#[derive(Clone, Copy, Debug)]
struct OrbCandidate {
    row: usize,
    col: usize,
    response: f32,
    angle: f32,
    cell_row: i32,
    cell_col: i32,
    threshold_kind: OrbDebugThresholdKind,
}

#[derive(Clone, Debug)]
struct ExtractorNode {
    id: usize,
    ul: (i32, i32),
    ur: (i32, i32),
    bl: (i32, i32),
    br: (i32, i32),
    keys: Vec<usize>,
    no_more: bool,
}

impl ExtractorNode {
    fn divide(&self, candidates: &[OrbCandidate], next_id: &mut usize) -> [ExtractorNode; 4] {
        let half_x = ((self.ur.0 - self.ul.0) as f32 / 2.0).ceil() as i32;
        let half_y = ((self.br.1 - self.ul.1) as f32 / 2.0).ceil() as i32;
        let mid_x = self.ul.0 + half_x;
        let mid_y = self.ul.1 + half_y;

        let mut n1 = ExtractorNode {
            id: alloc_node_id(next_id),
            ul: self.ul,
            ur: (mid_x, self.ul.1),
            bl: (self.ul.0, mid_y),
            br: (mid_x, mid_y),
            keys: Vec::new(),
            no_more: false,
        };
        let mut n2 = ExtractorNode {
            id: alloc_node_id(next_id),
            ul: (mid_x, self.ul.1),
            ur: self.ur,
            bl: (mid_x, mid_y),
            br: (self.ur.0, mid_y),
            keys: Vec::new(),
            no_more: false,
        };
        let mut n3 = ExtractorNode {
            id: alloc_node_id(next_id),
            ul: (self.ul.0, mid_y),
            ur: (mid_x, mid_y),
            bl: self.bl,
            br: (mid_x, self.bl.1),
            keys: Vec::new(),
            no_more: false,
        };
        let mut n4 = ExtractorNode {
            id: alloc_node_id(next_id),
            ul: (mid_x, mid_y),
            ur: (self.ur.0, mid_y),
            bl: (mid_x, self.bl.1),
            br: self.br,
            keys: Vec::new(),
            no_more: false,
        };

        for &idx in &self.keys {
            let kp = candidates[idx];
            let x = kp.col as i32;
            let y = kp.row as i32;
            if x < mid_x {
                if y < mid_y {
                    n1.keys.push(idx);
                } else {
                    n3.keys.push(idx);
                }
            } else if y < mid_y {
                n2.keys.push(idx);
            } else {
                n4.keys.push(idx);
            }
        }

        n1.no_more = n1.keys.len() == 1;
        n2.no_more = n2.keys.len() == 1;
        n3.no_more = n3.keys.len() == 1;
        n4.no_more = n4.keys.len() == 1;

        [n1, n2, n3, n4]
    }
}

fn alloc_node_id(next_id: &mut usize) -> usize {
    let id = *next_id;
    *next_id += 1;
    id
}

impl Default for OrbDetector {
    fn default() -> Self {
        Self {
            downscale: 1.2,
            n_scales: 8,
            n_keypoints: 500,
            fast_n: 9,
            // ORB-SLAM3 uses 20/255 ≈ 0.078 for ini and 7/255 ≈ 0.027 for min
            ini_fast_threshold: 20.0 / 255.0,
            min_fast_threshold: 7.0 / 255.0,
            harris_k: 0.04,
            cell_size: 35,
        }
    }
}

impl OrbDetector {
    /// Create a new `OrbDetector` with default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    fn features_per_level(&self) -> Vec<usize> {
        let nlevels = self.n_scales;
        let scale_factor = self.downscale;
        let mut features_per_level = vec![0usize; nlevels];

        let factor = 1.0f32 / scale_factor;
        let mut n_desired =
            (self.n_keypoints as f32) * (1.0 - factor) / (1.0 - factor.powi(nlevels as i32));

        let mut sum = 0usize;
        for item in features_per_level.iter_mut().take(nlevels - 1) {
            let n = n_desired.round().max(0.0) as usize;
            *item = n;
            sum += n;
            n_desired *= factor;
        }
        features_per_level[nlevels - 1] = self.n_keypoints.saturating_sub(sum);

        features_per_level
    }

    fn build_pyramid<A: ImageAllocator>(
        &self,
        img: &Image<f32, 1, A>,
    ) -> Result<Vec<Image<f32, 1, CpuAllocator>>, ImageError> {
        let orig_w = img.width() as f32;
        let orig_h = img.height() as f32;

        let mut pyramid: Vec<Image<f32, 1, CpuAllocator>> = Vec::with_capacity(self.n_scales);
        pyramid.push(Image::from_size_slice(
            img.size(),
            img.as_slice(),
            CpuAllocator,
        )?);

        for level in 1..self.n_scales {
            let inv_scale = 1.0f32 / self.downscale.powi(level as i32);
            let new_w = (orig_w * inv_scale).round().max(1.0) as usize;
            let new_h = (orig_h * inv_scale).round().max(1.0) as usize;

            let prev = &pyramid[level - 1];
            if new_w == prev.width() && new_h == prev.height() {
                break;
            }

            let mut resized = Image::from_size_val(
                ImageSize {
                    width: new_w,
                    height: new_h,
                },
                0.0,
                CpuAllocator,
            )?;
            resize_linear_pixel_center(prev, &mut resized);
            pyramid.push(resized);
        }

        Ok(pyramid)
    }

    #[allow(clippy::type_complexity)]
    fn detect_octave_debug<A: ImageAllocator>(
        &self,
        octave_image: &Image<f32, 1, A>,
        octave: usize,
    ) -> Result<
        (
            Vec<[usize; 2]>,
            Vec<f32>,
            Vec<f32>,
            Vec<OrbDebugCandidate>,
            Vec<OrbDebugCandidate>,
        ),
        ImageError,
    > {
        const EDGE_THRESHOLD: i32 = 19;
        let cell_w = self.cell_size as i32;
        let (min_x, max_x, min_y, max_y) = octave_bounds(octave_image);
        let region_w = (max_x - min_x).max(0);
        let region_h = (max_y - min_y).max(0);
        if region_w <= 0 || region_h <= 0 {
            return Ok((vec![], vec![], vec![], vec![], vec![]));
        }
        let n_cols = (region_w / cell_w).max(1) as usize;
        let n_rows = (region_h / cell_w).max(1) as usize;
        let w_cell = ((region_w as f32) / (n_cols as f32)).ceil() as i32;
        let h_cell = ((region_h as f32) / (n_rows as f32)).ceil() as i32;
        let scale = self.downscale.powi(octave as i32);

        let mut keypoints: Vec<[usize; 2]> = Vec::new();
        let mut responses: Vec<f32> = Vec::new();
        let mut raw_debug_candidates = Vec::new();
        let mut debug_candidates = Vec::new();
        let padded = reflect101_pad(octave_image, EDGE_THRESHOLD as usize)?;
        for row in 0..n_rows {
            let ini_y = min_y + row as i32 * h_cell;
            if ini_y >= max_y - 3 {
                continue;
            }
            let max_y_cell = (ini_y + h_cell + 6).min(max_y);

            for col in 0..n_cols {
                let ini_x = min_x + col as i32 * w_cell;
                if ini_x >= max_x - 6 {
                    continue;
                }
                let max_x_cell = (ini_x + w_cell + 6).min(max_x);

                let roi = extract_subimage(
                    &padded,
                    (ini_y + EDGE_THRESHOLD) as usize,
                    (ini_x + EDGE_THRESHOLD) as usize,
                    (max_y_cell - ini_y) as usize,
                    (max_x_cell - ini_x) as usize,
                )?;
                let mut fast_ini =
                    FastDetector::new(roi.size(), self.ini_fast_threshold, self.fast_n, 1)?;
                fast_ini.compute_corner_response(&roi);
                let mut raw_cell_keypoints = fast_ini.extract_raw_keypoints();
                let mut cell_keypoints = fast_ini.extract_keypoints()?;
                let mut threshold_kind = OrbDebugThresholdKind::Initial;
                let mut cell_responses: Vec<f32> = cell_keypoints
                    .iter()
                    .map(|&[r, c]| {
                        fast_ini.corner_response().as_slice()
                            [fast_ini.corner_response().size().index(r, c)]
                    })
                    .collect();

                if cell_keypoints.is_empty() {
                    let mut fast_min =
                        FastDetector::new(roi.size(), self.min_fast_threshold, self.fast_n, 1)?;
                    fast_min.compute_corner_response(&roi);
                    raw_cell_keypoints = fast_min.extract_raw_keypoints();
                    cell_keypoints = fast_min.extract_keypoints()?;
                    threshold_kind = OrbDebugThresholdKind::Minimum;
                    cell_responses = cell_keypoints
                        .iter()
                        .map(|&[r, c]| {
                            fast_min.corner_response().as_slice()
                                [fast_min.corner_response().size().index(r, c)]
                        })
                        .collect();

                    raw_debug_candidates.extend(raw_cell_keypoints.into_iter().map(|[r, c]| {
                        let abs_r = (r as i32 + ini_y) as usize;
                        let abs_c = (c as i32 + ini_x) as usize;
                        OrbDebugCandidate {
                            x: abs_c as f32 * scale,
                            y: abs_r as f32 * scale,
                            response: fast_min.corner_response().as_slice()
                                [fast_min.corner_response().size().index(r, c)],
                            octave,
                            cell_row: row as i32,
                            cell_col: col as i32,
                            threshold_kind,
                        }
                    }));
                } else {
                    raw_debug_candidates.extend(raw_cell_keypoints.into_iter().map(|[r, c]| {
                        let abs_r = (r as i32 + ini_y) as usize;
                        let abs_c = (c as i32 + ini_x) as usize;
                        OrbDebugCandidate {
                            x: abs_c as f32 * scale,
                            y: abs_r as f32 * scale,
                            response: fast_ini.corner_response().as_slice()
                                [fast_ini.corner_response().size().index(r, c)],
                            octave,
                            cell_row: row as i32,
                            cell_col: col as i32,
                            threshold_kind,
                        }
                    }));
                }

                for ([r, c], response) in cell_keypoints.into_iter().zip(cell_responses.into_iter())
                {
                    let abs_r = (r as i32 + ini_y) as usize;
                    let abs_c = (c as i32 + ini_x) as usize;
                    keypoints.push([abs_r, abs_c]);
                    responses.push(response);
                    debug_candidates.push(OrbDebugCandidate {
                        x: abs_c as f32 * scale,
                        y: abs_r as f32 * scale,
                        response,
                        octave,
                        cell_row: row as i32,
                        cell_col: col as i32,
                        threshold_kind,
                    });
                }
            }
        }

        if keypoints.is_empty() {
            return Ok((vec![], vec![], vec![], vec![], raw_debug_candidates));
        }

        let mask = mask_border_keypoints(octave_image.size(), &keypoints, EDGE_THRESHOLD);
        let filtered_keypoints: Vec<_> = keypoints
            .iter()
            .zip(mask.iter())
            .filter_map(|(kp, &m)| if m { Some(*kp) } else { None })
            .collect();

        if filtered_keypoints.is_empty() {
            return Ok((vec![], vec![], vec![], vec![], raw_debug_candidates));
        }

        let padded_keypoints: Vec<_> = keypoints
            .iter()
            .map(|&[r, c]| [r + EDGE_THRESHOLD as usize, c + EDGE_THRESHOLD as usize])
            .collect();
        let orientations = corner_orientations(&padded, &padded_keypoints);
        let filtered_orientations: Vec<_> = orientations
            .iter()
            .zip(mask.iter())
            .filter_map(|(&ori, &m)| if m { Some(ori) } else { None })
            .collect();

        let filtered_responses: Vec<_> = responses
            .iter()
            .zip(mask.iter())
            .filter_map(|(&response, &m)| if m { Some(response) } else { None })
            .collect();

        let filtered_debug_candidates: Vec<_> = debug_candidates
            .into_iter()
            .zip(mask.iter())
            .filter_map(|(candidate, &m)| if m { Some(candidate) } else { None })
            .collect();

        Ok((
            filtered_keypoints,
            filtered_orientations,
            filtered_responses,
            filtered_debug_candidates,
            raw_debug_candidates,
        ))
    }

    /// Detect ORB keypoints in a grayscale image.
    ///
    /// Returns `(keypoints, scales, orientations, responses)` where each vector
    /// has the same length and keypoints are in `(row, col)` coordinates.
    #[allow(clippy::type_complexity)]
    pub fn detect<A: ImageAllocator>(
        &self,
        src: &Image<f32, 1, A>,
    ) -> Result<(Vec<(f32, f32)>, Vec<f32>, Vec<f32>, Vec<f32>), ImageError> {
        let pyramid = self.build_pyramid(src)?;
        let features_per_level = self.features_per_level();

        let mut keypoints_list = vec![];
        let mut orientations_list = vec![];
        let mut scales_list = vec![];
        let mut responses_list = vec![];

        for (octave, octave_image) in pyramid.iter().enumerate() {
            let (keypoints, orientations, responses, _, _) =
                self.detect_octave_debug(octave_image, octave)?;
            if keypoints.is_empty() {
                continue;
            }

            let candidates: Vec<OrbCandidate> = keypoints
                .iter()
                .zip(orientations.iter())
                .zip(responses.iter())
                .map(|((&[r, c], &angle), &response)| OrbCandidate {
                    row: r,
                    col: c,
                    response,
                    angle,
                    cell_row: -1,
                    cell_col: -1,
                    threshold_kind: OrbDebugThresholdKind::Initial,
                })
                .collect();

            let (min_x, max_x, min_y, max_y) = octave_bounds(octave_image);
            let distributed = distribute_octree(
                &candidates,
                min_x,
                max_x,
                min_y,
                max_y,
                features_per_level[octave],
            );

            let scale = self.downscale.powi(octave as i32);

            for cand in &distributed {
                keypoints_list.push((cand.row as f32 * scale, cand.col as f32 * scale));
                orientations_list.push(cand.angle);
                scales_list.push(scale);
                responses_list.push(cand.response);
            }
        }

        Ok((
            keypoints_list,
            scales_list,
            orientations_list,
            responses_list,
        ))
    }

    /// Detect ORB keypoints and return per-octave debug data.
    pub fn detect_debug<A: ImageAllocator>(
        &self,
        src: &Image<f32, 1, A>,
    ) -> Result<OrbDebugFrame, ImageError> {
        let pyramid = self.build_pyramid(src)?;
        let features_per_level = self.features_per_level();
        let mut octaves = Vec::with_capacity(pyramid.len());

        for (octave, octave_image) in pyramid.iter().enumerate() {
            let (keypoints, orientations, responses, debug_candidates, raw_debug_candidates) =
                self.detect_octave_debug(octave_image, octave)?;
            let candidates: Vec<OrbCandidate> = keypoints
                .iter()
                .zip(orientations.iter())
                .zip(responses.iter())
                .zip(debug_candidates.iter())
                .map(|(((&[r, c], &angle), &response), debug)| OrbCandidate {
                    row: r,
                    col: c,
                    response,
                    angle,
                    cell_row: debug.cell_row,
                    cell_col: debug.cell_col,
                    threshold_kind: debug.threshold_kind,
                })
                .collect();

            let (min_x, max_x, min_y, max_y) = octave_bounds(octave_image);
            let distributed = distribute_octree(
                &candidates,
                min_x,
                max_x,
                min_y,
                max_y,
                features_per_level[octave],
            );
            let scale = self.downscale.powi(octave as i32);
            let survivors = distributed
                .into_iter()
                .map(|cand| OrbDebugCandidate {
                    x: cand.col as f32 * scale,
                    y: cand.row as f32 * scale,
                    response: cand.response,
                    octave,
                    cell_row: cand.cell_row,
                    cell_col: cand.cell_col,
                    threshold_kind: cand.threshold_kind,
                })
                .collect();

            octaves.push(OrbDebugOctave {
                octave,
                image_size: octave_image.size(),
                raw_candidates: raw_debug_candidates,
                candidates: debug_candidates,
                survivors,
            });
        }

        Ok(OrbDebugFrame { octaves })
    }

    fn extract_octave<A: ImageAllocator>(
        &self,
        octave_image: &Image<f32, 1, A>,
        keypoints: &[(i32, i32)],
        orientations: &[f32],
    ) -> Result<(Vec<[u8; 32]>, Vec<bool>), ImageError> {
        let mask = mask_border_keypoints_i32(octave_image.size(), keypoints, 20);

        // Filter keypoints and orientations by mask
        let filtered_keypoints: Vec<_> = keypoints
            .iter()
            .zip(mask.iter())
            .filter_map(|(kp, &m)| if m { Some(*kp) } else { None })
            .collect();

        let filtered_orientations: Vec<f32> = orientations
            .iter()
            .zip(mask.iter())
            .filter_map(|(&o, &m)| if m { Some(o) } else { None })
            .collect();

        // Apply Gaussian blur before computing descriptors (matches ORB-SLAM3).
        // ORB-SLAM3 uses 7x7 kernel with sigma=2.
        let mut blurred = Image::from_size_val(octave_image.size(), 0.0f32, CpuAllocator)?;
        gaussian_blur(octave_image, &mut blurred, (7, 7), (2.0, 2.0))?;

        let descriptors = orb_loop(&blurred, &filtered_keypoints, &filtered_orientations);

        Ok((descriptors, mask))
    }

    /// Compute rotated BRIEF descriptors for previously detected keypoints.
    ///
    /// Returns `(descriptors, mask)` where `mask[i]` indicates whether descriptor `i`
    /// was successfully computed (keypoints too close to the image border are skipped).
    pub fn extract<A: ImageAllocator>(
        &self,
        src: &Image<f32, 1, A>,
        keypoints: &[(f32, f32)],
        scales: &[f32],
        orientations: &[f32],
    ) -> Result<(Vec<[u8; 32]>, Vec<bool>), ImageError> {
        let pyramid = self.build_pyramid(src)?;

        let mut descriptors_list: Vec<[u8; 32]> = Vec::new();
        let mut mask_list: Vec<bool> = Vec::new();

        let octaves: Vec<usize> = scales
            .iter()
            .map(|&s| (s.ln() / self.downscale.ln()).round() as usize)
            .collect();

        for (octave, octave_image) in pyramid.iter().enumerate() {
            let octave_mask: Vec<bool> = octaves.iter().map(|&o| o == octave).collect();

            let n_in_octave = octave_mask.iter().filter(|&&b| b).count();
            if n_in_octave == 0 {
                continue;
            }

            let mut octave_keypoints = Vec::with_capacity(n_in_octave);
            let mut octave_orientations = Vec::with_capacity(n_in_octave);

            for ((&(y, x), &ori), &is_in_octave) in
                keypoints.iter().zip(orientations).zip(&octave_mask)
            {
                if is_in_octave {
                    let scale = self.downscale.powi(octave as i32);
                    octave_keypoints.push(((y / scale).round() as i32, (x / scale).round() as i32));
                    octave_orientations.push(ori);
                }
            }

            let (descriptors, mask) =
                self.extract_octave(octave_image, &octave_keypoints, &octave_orientations)?;

            descriptors_list.extend(descriptors);
            mask_list.extend(mask);
        }

        Ok((descriptors_list, mask_list))
    }

    /// Detect keypoints and compute descriptors in one call.
    ///
    /// Equivalent to calling [`detect`](Self::detect) then [`extract`](Self::extract),
    /// but filters out border keypoints and returns an [`OrbFeatures`] with matching
    /// `keypoints_xy`, `orientations`, and `descriptors` vectors.
    pub fn detect_and_extract<A: ImageAllocator>(
        &self,
        src: &Image<f32, 1, A>,
    ) -> Result<OrbFeatures, ImageError> {
        let (kps_rc, scales, orientations, _responses) = self.detect(src)?;
        let (descriptors, mask) = self.extract(src, &kps_rc, &scales, &orientations)?;

        let mut keypoints_xy = Vec::with_capacity(descriptors.len());
        let mut valid_orientations = Vec::with_capacity(descriptors.len());
        let mut valid_descriptors = Vec::with_capacity(descriptors.len());
        let mut valid_scales = Vec::with_capacity(descriptors.len());

        // `mask` has one entry per keypoint; `descriptors` has entries only
        // for keypoints where mask is true, so we track a separate index.
        let mut desc_idx = 0;
        for (i, (((row, col), &ori), &sc)) in kps_rc
            .iter()
            .zip(orientations.iter())
            .zip(scales.iter())
            .enumerate()
        {
            if mask.get(i).copied().unwrap_or(false) {
                keypoints_xy.push([*col, *row]);
                valid_orientations.push(ori);
                valid_descriptors.push(descriptors[desc_idx]);
                valid_scales.push(sc);
                desc_idx += 1;
            }
        }

        Ok(OrbFeatures {
            keypoints_xy,
            orientations: valid_orientations,
            descriptors: valid_descriptors,
            scales: valid_scales,
        })
    }
}

fn reflect101_pad<A: ImageAllocator>(
    src: &Image<f32, 1, A>,
    border: usize,
) -> Result<Image<f32, 1, CpuAllocator>, ImageError> {
    let size = ImageSize {
        width: src.width() + 2 * border,
        height: src.height() + 2 * border,
    };
    let mut dst = Image::from_size_val(size, 0.0f32, CpuAllocator)?;
    spatial_padding(
        src,
        &mut dst,
        Padding2D {
            top: border,
            bottom: border,
            left: border,
            right: border,
        },
        PaddingMode::Reflect101,
        [0.0f32],
    )?;
    Ok(dst)
}

fn extract_subimage<A: ImageAllocator>(
    src: &Image<f32, 1, A>,
    top: usize,
    left: usize,
    height: usize,
    width: usize,
) -> Result<Image<f32, 1, CpuAllocator>, ImageError> {
    let mut dst = Image::from_size_val(ImageSize { width, height }, 0.0f32, CpuAllocator)?;
    for row in 0..height {
        let src_row = top + row;
        let dst_offset = row * width;
        let src_offset = src_row * src.width() + left;
        dst.as_slice_mut()[dst_offset..dst_offset + width]
            .copy_from_slice(&src.as_slice()[src_offset..src_offset + width]);
    }
    Ok(dst)
}

fn resize_linear_pixel_center<A: ImageAllocator>(
    src: &Image<f32, 1, A>,
    dst: &mut Image<f32, 1, CpuAllocator>,
) {
    let src_w = src.width();
    let src_h = src.height();
    let dst_w = dst.width();
    let dst_h = dst.height();

    if src_w == dst_w && src_h == dst_h {
        dst.as_slice_mut().copy_from_slice(src.as_slice());
        return;
    }

    let sx_ratio = src_w as f32 / dst_w as f32;
    let sy_ratio = src_h as f32 / dst_h as f32;
    let sx_max = (src_w - 1) as i32;
    let sy_max = (src_h - 1) as i32;

    let src_slice = src.as_slice();
    let dst_slice = dst.as_slice_mut();

    for row in 0..dst_h {
        let sy = (row as f32 + 0.5) * sy_ratio - 0.5;
        let y0f = sy.floor();
        let wy = sy - y0f;
        let y0 = (y0f as i32).clamp(0, sy_max) as usize;
        let y1 = ((y0f as i32) + 1).clamp(0, sy_max) as usize;
        let row0 = y0 * src_w;
        let row1 = y1 * src_w;

        for col in 0..dst_w {
            let sx = (col as f32 + 0.5) * sx_ratio - 0.5;
            let x0f = sx.floor();
            let wx = sx - x0f;
            let x0 = (x0f as i32).clamp(0, sx_max) as usize;
            let x1 = ((x0f as i32) + 1).clamp(0, sx_max) as usize;

            let p00 = src_slice[row0 + x0];
            let p01 = src_slice[row0 + x1];
            let p10 = src_slice[row1 + x0];
            let p11 = src_slice[row1 + x1];

            let top = p00 + wx * (p01 - p00);
            let bot = p10 + wx * (p11 - p10);
            dst_slice[row * dst_w + col] = top + wy * (bot - top);
        }
    }
}

fn mask_border_keypoints(size: ImageSize, keypoints: &[[usize; 2]], distance: i32) -> Vec<bool> {
    let rows = size.height;
    let cols = size.width;

    keypoints
        .iter()
        .map(|[r, c]| {
            let min = distance.saturating_sub(1);
            let max_row = rows as isize - distance as isize + 1;
            let max_col = cols as isize - distance as isize + 1;
            let r = *r as isize;
            let c = *c as isize;

            (min as isize) < r && r < max_row && (min as isize) < c && c < max_col
        })
        .collect()
}

fn octave_bounds<A: ImageAllocator>(image: &Image<f32, 1, A>) -> (i32, i32, i32, i32) {
    const EDGE_THRESHOLD: i32 = 19;
    let min_x = EDGE_THRESHOLD - 3;
    let min_y = EDGE_THRESHOLD - 3;
    let max_x = image.width() as i32 - EDGE_THRESHOLD + 3;
    let max_y = image.height() as i32 - EDGE_THRESHOLD + 3;
    (min_x, max_x, min_y, max_y)
}

fn distribute_octree(
    candidates: &[OrbCandidate],
    min_x: i32,
    max_x: i32,
    min_y: i32,
    max_y: i32,
    n_features: usize,
) -> Vec<OrbCandidate> {
    if candidates.is_empty() || n_features == 0 {
        return Vec::new();
    }

    let width = (max_x - min_x).max(1) as f32;
    let height = (max_y - min_y).max(1) as f32;
    let mut n_ini = (width / height).round() as usize;
    if n_ini == 0 {
        n_ini = 1;
    }
    let h_x = width / n_ini as f32;

    let mut next_node_id = 0usize;
    let mut nodes: Vec<ExtractorNode> = Vec::with_capacity(n_ini);
    for i in 0..n_ini {
        let ul = (min_x + (h_x * i as f32) as i32, min_y);
        let ur = (min_x + (h_x * (i + 1) as f32) as i32, min_y);
        let bl = (ul.0, max_y);
        let br = (ur.0, max_y);
        nodes.push(ExtractorNode {
            id: alloc_node_id(&mut next_node_id),
            ul,
            ur,
            bl,
            br,
            keys: Vec::new(),
            no_more: false,
        });
    }

    for (idx, cand) in candidates.iter().enumerate() {
        let x = cand.col as f32 - min_x as f32;
        let mut bin = (x / h_x) as usize;
        if bin >= nodes.len() {
            bin = nodes.len() - 1;
        }
        nodes[bin].keys.push(idx);
    }

    nodes.retain(|n| !n.keys.is_empty());
    for node in nodes.iter_mut() {
        if node.keys.len() == 1 {
            node.no_more = true;
        }
    }

    let mut finished = false;
    while !finished {
        let prev_size = nodes.len();
        let mut n_to_expand = 0usize;
        let mut size_and_node_ids: Vec<(usize, i32, usize)> = Vec::new();
        let to_expand: Vec<usize> = nodes
            .iter()
            .filter(|n| !n.no_more && n.keys.len() > 1)
            .map(|n| n.id)
            .collect();

        for node_id in to_expand {
            let Some(pos) = nodes.iter().position(|n| n.id == node_id) else {
                continue;
            };

            let node = nodes.remove(pos);
            let children = node.divide(candidates, &mut next_node_id);
            for child in children {
                if child.keys.is_empty() {
                    continue;
                }
                if child.keys.len() > 1 {
                    n_to_expand += 1;
                    size_and_node_ids.push((child.keys.len(), child.ul.0, child.id));
                }
                nodes.insert(0, child);
            }
        }

        if nodes.len() >= n_features || nodes.len() == prev_size {
            finished = true;
        } else if nodes.len() + n_to_expand * 3 > n_features {
            while !finished {
                let prev_size = nodes.len();
                let mut prev = size_and_node_ids.clone();
                size_and_node_ids.clear();

                prev.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
                for (_, _, node_id) in prev.into_iter().rev() {
                    let Some(pos) = nodes.iter().position(|n| n.id == node_id) else {
                        continue;
                    };

                    let node = nodes.remove(pos);
                    let children = node.divide(candidates, &mut next_node_id);
                    for child in children {
                        if child.keys.is_empty() {
                            continue;
                        }
                        if child.keys.len() > 1 {
                            size_and_node_ids.push((child.keys.len(), child.ul.0, child.id));
                        }
                        nodes.insert(0, child);
                    }

                    if nodes.len() >= n_features {
                        break;
                    }
                }

                if nodes.len() >= n_features || nodes.len() == prev_size {
                    finished = true;
                }
            }
        }
    }

    let mut result = Vec::with_capacity(nodes.len().min(n_features));
    for node in nodes {
        let mut best_idx = node.keys[0];
        let mut best_response = candidates[best_idx].response;
        for &idx in &node.keys[1..] {
            if candidates[idx].response > best_response {
                best_response = candidates[idx].response;
                best_idx = idx;
            }
        }
        result.push(candidates[best_idx]);
    }

    result
}

/// Check which keypoints fall outside the image border margin.
/// Variant that accepts `(i32, i32)` keypoints (used by extract_octave).
fn mask_border_keypoints_i32(
    size: ImageSize,
    keypoints: &[(i32, i32)],
    distance: i32,
) -> Vec<bool> {
    let rows = size.height;
    let cols = size.width;

    keypoints
        .iter()
        .map(|(r, c)| {
            let min = distance.saturating_sub(1);
            let max_row = rows as isize - distance as isize + 1;
            let max_col = cols as isize - distance as isize + 1;
            let r = *r as isize;
            let c = *c as isize;

            (min as isize) < r && r < max_row && (min as isize) < c && c < max_col
        })
        .collect()
}

fn corner_orientations<A: ImageAllocator>(
    src: &Image<f32, 1, A>,
    corners: &[[usize; 2]],
) -> Vec<f32> {
    const HALF_PATCH_SIZE: i32 = 15;
    let src_slice = src.as_slice();
    let height = src.height() as i32;
    let width = src.width() as i32;
    let umax = orb_umax();

    let mut orientations = Vec::with_capacity(corners.len());

    for &[r0, c0] in corners {
        let mut m01 = 0f32;
        let mut m10 = 0f32;
        let row = r0 as i32;
        let col = c0 as i32;

        for u in -HALF_PATCH_SIZE..=HALF_PATCH_SIZE {
            let rr = row;
            let cc = col + u;
            if rr >= 0 && rr < height && cc >= 0 && cc < width {
                let curr = src_slice[src.size().index(rr as usize, cc as usize)];
                m10 += u as f32 * curr;
            }
        }

        for v in 1..=HALF_PATCH_SIZE {
            let mut v_sum = 0f32;
            let d = umax[v as usize];
            for u in -d..=d {
                let rr_plus = row + v;
                let rr_minus = row - v;
                let cc = col + u;
                if rr_plus >= 0
                    && rr_plus < height
                    && rr_minus >= 0
                    && rr_minus < height
                    && cc >= 0
                    && cc < width
                {
                    let val_plus = src_slice[src.size().index(rr_plus as usize, cc as usize)];
                    let val_minus = src_slice[src.size().index(rr_minus as usize, cc as usize)];
                    v_sum += val_plus - val_minus;
                    m10 += u as f32 * (val_plus + val_minus);
                }
            }
            m01 += v as f32 * v_sum;
        }
        orientations.push(m01.atan2(m10));
    }

    orientations
}

fn orb_umax() -> [i32; 16] {
    const HALF_PATCH_SIZE: i32 = 15;
    let mut umax = [0i32; 16];
    let vmax = ((HALF_PATCH_SIZE as f32) * std::f32::consts::SQRT_2 / 2.0 + 1.0).floor() as i32;
    let vmin = ((HALF_PATCH_SIZE as f32) * std::f32::consts::SQRT_2 / 2.0).ceil() as i32;
    let hp2 = (HALF_PATCH_SIZE * HALF_PATCH_SIZE) as f32;

    for v in 0..=vmax {
        umax[v as usize] = (hp2 - (v * v) as f32).sqrt().round() as i32;
    }

    let mut v0 = 0;
    for v in (vmin..=HALF_PATCH_SIZE).rev() {
        while umax[v0 as usize] == umax[(v0 + 1) as usize] {
            v0 += 1;
        }
        umax[v as usize] = v0;
        v0 += 1;
    }

    umax
}

/// Compute ORB descriptors packed into 32 bytes (256 bits) per keypoint.
/// This matches the ORB-SLAM3/OpenCV descriptor format.
fn orb_loop<A: ImageAllocator>(
    src: &Image<f32, 1, A>,
    keypoints: &[(i32, i32)],
    orientation: &[f32],
) -> Vec<[u8; 32]> {
    let n_keypoints = keypoints.len();
    let height = src.height() as i32;
    let width = src.width() as i32;

    let mut descriptors = vec![[0u8; 32]; n_keypoints];

    for (desc, (&(kr, kc), &angle)) in descriptors
        .iter_mut()
        .zip(keypoints.iter().zip(orientation.iter()))
    {
        let sin_a = angle.sin();
        let cos_a = angle.cos();

        // Process 8 bit comparisons at a time to pack into one byte
        for (byte_idx, byte_out) in desc.iter_mut().enumerate() {
            let mut byte_val = 0u8;
            for bit_idx in 0..8 {
                let j = byte_idx * 8 + bit_idx;
                let pr0 = POS0[j][0] as f32;
                let pc0 = POS0[j][1] as f32;
                let pr1 = POS1[j][0] as f32;
                let pc1 = POS1[j][1] as f32;

                let spr0 = (sin_a * pr0 + cos_a * pc0).round() as i32;
                let spc0 = (cos_a * pr0 - sin_a * pc0).round() as i32;
                let spr1 = (sin_a * pr1 + cos_a * pc1).round() as i32;
                let spc1 = (cos_a * pr1 - sin_a * pc1).round() as i32;

                let r0 = kr + spr0;
                let c0 = kc + spc0;
                let r1 = kr + spr1;
                let c1 = kc + spc1;

                if r0 >= 0
                    && r0 < height
                    && c0 >= 0
                    && c0 < width
                    && r1 >= 0
                    && r1 < height
                    && c1 >= 0
                    && c1 < width
                {
                    let v0 = src.as_slice()[src.size().index(r0 as usize, c0 as usize)];
                    let v1 = src.as_slice()[src.size().index(r1 as usize, c1 as usize)];
                    if v0 < v1 {
                        byte_val |= 1 << bit_idx;
                    }
                }
            }
            *byte_out = byte_val;
        }
    }

    descriptors
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::Image;
    use kornia_tensor::CpuAllocator;

    fn make_gradient_x(size: usize) -> Image<f32, 1, CpuAllocator> {
        let mut img = Image::from_size_val([size, size].into(), 0.0, CpuAllocator).unwrap();
        let width = img.width();
        let height = img.height();
        let denom = (width.saturating_sub(1)).max(1) as f32;

        for y in 0..height {
            for x in 0..width {
                img.as_slice_mut()[y * width + x] = x as f32 / denom;
            }
        }

        img
    }

    fn make_gradient_y(size: usize) -> Image<f32, 1, CpuAllocator> {
        let mut img = Image::from_size_val([size, size].into(), 0.0, CpuAllocator).unwrap();
        let width = img.width();
        let height = img.height();
        let denom = (height.saturating_sub(1)).max(1) as f32;

        for y in 0..height {
            let value = y as f32 / denom;
            for x in 0..width {
                img.as_slice_mut()[y * width + x] = value;
            }
        }

        img
    }

    #[test]
    fn test_corner_orientations_gradient() {
        let size = 31;
        let center = [[size / 2, size / 2]];

        let img_x = make_gradient_x(size);
        let ori_x = corner_orientations(&img_x, &center)[0];
        assert!(ori_x.abs() < 0.1, "expected ~0 rad, got {ori_x}");

        let img_y = make_gradient_y(size);
        let ori_y = corner_orientations(&img_y, &center)[0].abs();
        let expected = std::f32::consts::FRAC_PI_2;
        assert!(
            (ori_y - expected).abs() < 0.1,
            "expected ~pi/2 rad, got {ori_y}"
        );
    }
}
