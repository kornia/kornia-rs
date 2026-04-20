use kornia_image::{allocator::ImageAllocator, Image, ImageError, ImageSize};
use kornia_tensor::CpuAllocator;
use rayon::prelude::*;

use crate::{
    features::{FastDetector, HarrisResponse},
    filter::{gaussian_blur, gaussian_blur_u8},
    interpolation::InterpolationMode,
    resize::{resize_fast_u8, resize_native},
};

use super::pattern::{rotated_pattern, N_BUCKETS, POS0, POS1};

/// ORB features extracted from a single frame.
#[derive(Debug, Clone)]
pub struct OrbFeatures {
    /// Keypoints as `[col, row]` in pixel coordinates.
    pub keypoints_xy: Vec<[f32; 2]>,
    /// Keypoint orientation angles (radians).
    pub orientations: Vec<f32>,
    /// Binary descriptors (256-bit, packed as 32 bytes each).
    pub descriptors: Vec<[u8; 32]>,
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
}

#[derive(Clone, Debug)]
struct ExtractorNode {
    ul: (i32, i32),
    ur: (i32, i32),
    bl: (i32, i32),
    br: (i32, i32),
    keys: Vec<usize>,
    no_more: bool,
}

impl ExtractorNode {
    fn divide(&self, candidates: &[OrbCandidate]) -> [ExtractorNode; 4] {
        let mid_x = (self.ul.0 + self.ur.0) / 2;
        let mid_y = (self.ul.1 + self.bl.1) / 2;

        let mut n1 = ExtractorNode {
            ul: self.ul,
            ur: (mid_x, self.ul.1),
            bl: (self.ul.0, mid_y),
            br: (mid_x, mid_y),
            keys: Vec::new(),
            no_more: false,
        };
        let mut n2 = ExtractorNode {
            ul: (mid_x, self.ul.1),
            ur: self.ur,
            bl: (mid_x, mid_y),
            br: (self.ur.0, mid_y),
            keys: Vec::new(),
            no_more: false,
        };
        let mut n3 = ExtractorNode {
            ul: (self.ul.0, mid_y),
            ur: (mid_x, mid_y),
            bl: self.bl,
            br: (mid_x, self.bl.1),
            keys: Vec::new(),
            no_more: false,
        };
        let mut n4 = ExtractorNode {
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

        [n1, n2, n3, n4]
    }
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
        let img = Image::from_size_slice(img.size(), img.as_slice(), CpuAllocator)?;

        let mut pyramid = Vec::with_capacity(self.n_scales);
        let mut current = img.clone();
        pyramid.push(current.clone());

        for _ in 1..self.n_scales {
            let next = pyramid_reduce(&current, self.downscale)?;
            if next.size() == current.size() {
                break;
            }

            pyramid.push(next.clone());
            current = next;
        }

        Ok(pyramid)
    }

    #[allow(clippy::type_complexity)]
    fn detect_octave<A: ImageAllocator>(
        &self,
        octave_image: &Image<f32, 1, A>,
    ) -> Result<(Vec<[usize; 2]>, Vec<f32>, Vec<f32>), ImageError> {
        // Two-tier FAST detection: first try with higher threshold, then lower if needed.
        // This matches ORB-SLAM3's approach of using iniThFAST then falling back to minThFAST.
        let mut fast_detector =
            FastDetector::new(octave_image.size(), self.ini_fast_threshold, self.fast_n, 1)?;
        fast_detector.compute_corner_response(octave_image);
        let mut keypoints = fast_detector.extract_keypoints()?;

        // If very few keypoints found, retry with lower threshold.
        // ORB-SLAM3 does this per-cell; we do it globally for simplicity.
        if keypoints.len() < 10 {
            let mut fast_detector_low =
                FastDetector::new(octave_image.size(), self.min_fast_threshold, self.fast_n, 1)?;
            fast_detector_low.compute_corner_response(octave_image);
            let keypoints_low = fast_detector_low.extract_keypoints()?;

            // Merge keypoints, avoiding duplicates (within 3 pixel radius).
            for kp in keypoints_low {
                let dominated = keypoints.iter().any(|&[r, c]| {
                    let dr = (r as i32 - kp[0] as i32).abs();
                    let dc = (c as i32 - kp[1] as i32).abs();
                    dr <= 3 && dc <= 3
                });
                if !dominated {
                    keypoints.push(kp);
                }
            }
        }

        if keypoints.is_empty() {
            return Ok((vec![], vec![], vec![]));
        }

        const EDGE_THRESHOLD: i32 = 19;
        let mask = mask_border_keypoints(octave_image.size(), &keypoints, EDGE_THRESHOLD);
        let filtered_keypoints: Vec<_> = keypoints
            .iter()
            .zip(mask.iter())
            .filter_map(|(kp, &m)| if m { Some(*kp) } else { None })
            .collect();

        if filtered_keypoints.is_empty() {
            return Ok((vec![], vec![], vec![]));
        }

        let orientations = corner_orientations(octave_image, &keypoints);
        let filtered_orientations: Vec<_> = keypoints
            .iter()
            .zip(orientations.iter())
            .zip(mask.iter())
            .filter_map(|((_, &ori), &m)| if m { Some(ori) } else { None })
            .collect();

        let mut response = Image::from_size_val(octave_image.size(), 0f32, CpuAllocator)?;
        let mut harris_response = HarrisResponse::new(octave_image.size()).with_k(self.harris_k);
        harris_response.compute(octave_image, &mut response)?;

        let filtered_responses: Vec<_> = filtered_keypoints
            .iter()
            .map(|&[r, c]| response.as_slice()[response.size().index(r, c)])
            .collect();

        Ok((
            filtered_keypoints,
            filtered_orientations,
            filtered_responses,
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
            let (keypoints, orientations, responses) = self.detect_octave(octave_image)?;
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

    fn build_pyramid_u8<A: ImageAllocator>(
        &self,
        img: &Image<u8, 1, A>,
    ) -> Result<Vec<Image<u8, 1, CpuAllocator>>, ImageError> {
        // Level 0 is the source. Avoid the double clone (one to
        // CpuAllocator-backed `current`, one to push into the pyramid) by
        // constructing level 0 directly from the input slice.
        let level0 = Image::from_size_slice(img.size(), img.as_slice(), CpuAllocator)?;

        let mut pyramid = Vec::with_capacity(self.n_scales);
        pyramid.push(level0);

        for _ in 1..self.n_scales {
            let next = pyramid_reduce_u8(pyramid.last().unwrap(), self.downscale)?;
            if next.size() == pyramid.last().unwrap().size() {
                break;
            }
            pyramid.push(next);
        }

        Ok(pyramid)
    }

    #[allow(clippy::type_complexity)]
    fn detect_u8_pyramid(
        &self,
        pyramid: &[Image<u8, 1, CpuAllocator>],
    ) -> Result<(Vec<(f32, f32)>, Vec<f32>, Vec<f32>, Vec<f32>), ImageError> {
        let features_per_level = self.features_per_level();
        let trace = std::env::var("KORNIA_ORB_TRACE").is_ok();

        const EDGE_THRESHOLD: usize = 19;
        // Process octaves in parallel: each octave is fully independent.
        // Per-octave tasks let small octaves finish their FAST + NMS + Harris
        // + BRIEF completely while oct 0 is still running, overlapping work
        // naturally. A two-phase split (chunked FAST then per-octave rest)
        // introduces a barrier that destroys this overlap and regresses
        // end-to-end even though chunked FAST alone is faster.
        let per_octave: Vec<_> = pyramid
            .par_iter()
            .enumerate()
            .map(|(octave, octave_image)| -> Result<_, ImageError> {
                let ta = std::time::Instant::now();
                // Selective nesting: oct 0 (>=900 rows) is the single-core
                // wall. Allow rayon to steal into its row work from cores
                // that finished smaller octaves. Oct 1+ stays serial — the
                // spawn/join overhead of nested par_iter across too many
                // octaves pushes wall time up even though individual cores
                // do the same amount of compute.
                let mut candidates = if octave_image.height() >= 900 {
                    crate::features::fast::fast_detect_rows_u8(
                        octave_image,
                        self.ini_fast_threshold,
                        self.fast_n,
                        EDGE_THRESHOLD,
                        0..octave_image.height(),
                    )
                } else {
                    crate::features::fast::fast_detect_rows_u8_serial(
                        octave_image,
                        self.ini_fast_threshold,
                        self.fast_n,
                        EDGE_THRESHOLD,
                        0..octave_image.height(),
                    )
                };

                // Softer-threshold fallback if initial detection was too sparse.
                if candidates.len() < 10 {
                    let low_candidates = crate::features::fast::fast_detect_rows_u8_serial(
                        octave_image,
                        self.min_fast_threshold,
                        self.fast_n,
                        EDGE_THRESHOLD,
                        0..octave_image.height(),
                    );
                    for cand in low_candidates {
                        let [r0, c0] = cand.0;
                        let dominated = candidates.iter().any(|&([r, c], _)| {
                            let dr = (r as i32 - r0 as i32).abs();
                            let dc = (c as i32 - c0 as i32).abs();
                            dr <= 3 && dc <= 3
                        });
                        if !dominated {
                            candidates.push(cand);
                        }
                    }
                }

                let raw_len = candidates.len();
                if candidates.is_empty() {
                    return Ok((
                        Vec::<(f32, f32)>::new(),
                        Vec::<f32>::new(),
                        Vec::<f32>::new(),
                        Vec::<f32>::new(),
                    ));
                }

                let nms_cap = self.n_keypoints.saturating_mul(20).max(2048);
                let mut fast_detector = FastDetector::new(
                    octave_image.size(),
                    self.ini_fast_threshold,
                    self.fast_n,
                    1,
                )?;
                let t_before_nms = std::time::Instant::now();
                let kp_with_resp = fast_detector.suppress_direct(candidates, nms_cap);
                let t_after_nms = std::time::Instant::now();
                if trace {
                    eprintln!(
                        "  oct[{:4}x{:4}] fast={:.2} nms={:.2} kp_raw={} kp_filt={}",
                        octave_image.width(),
                        octave_image.height(),
                        (t_before_nms - ta).as_secs_f64() * 1000.0,
                        (t_after_nms - t_before_nms).as_secs_f64() * 1000.0,
                        raw_len,
                        kp_with_resp.len(),
                    );
                }
                if kp_with_resp.is_empty() {
                    return Ok((
                        Vec::<(f32, f32)>::new(),
                        Vec::<f32>::new(),
                        Vec::<f32>::new(),
                        Vec::<f32>::new(),
                    ));
                }
                let (keypoints, responses): (Vec<[usize; 2]>, Vec<f32>) =
                    kp_with_resp.into_iter().unzip();

                let mut candidates: Vec<OrbCandidate> = keypoints
                    .iter()
                    .zip(responses.iter())
                    .map(|(&[r, c], &response)| OrbCandidate {
                        row: r,
                        col: c,
                        response,
                        angle: 0.0,
                    })
                    .collect();

                let target = features_per_level[octave];
                let cap = (target.saturating_mul(20)).max(512);
                if candidates.len() > cap {
                    candidates.select_nth_unstable_by(cap, |a, b| {
                        b.response
                            .partial_cmp(&a.response)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    candidates.truncate(cap);
                }

                let t_before_octree = std::time::Instant::now();
                let image_f32_size = octave_image.size();
                let min_x = 19 - 3;
                let min_y = 19 - 3;
                let max_x = image_f32_size.width as i32 - 19 + 3;
                let max_y = image_f32_size.height as i32 - 19 + 3;
                let distributed =
                    distribute_octree(&candidates, min_x, max_x, min_y, max_y, target);
                let t_after_octree = std::time::Instant::now();

                let survivor_positions: Vec<[usize; 2]> =
                    distributed.iter().map(|c| [c.row, c.col]).collect();
                // Serial here — the outer pyramid `par_iter` already saturates
                // all cores. Nested parallelism would oversubscribe and hurt.
                let survivor_harris: Vec<f32> = survivor_positions
                    .iter()
                    .map(|&[r, c]| harris_response_at_u8(octave_image, r, c, self.harris_k))
                    .collect();
                let survivor_orientations =
                    corner_orientations_u8(octave_image, &survivor_positions);
                let t_after_ori = std::time::Instant::now();

                let scale = self.downscale.powi(octave as i32);
                let mut kps = Vec::with_capacity(distributed.len());
                let mut scales = Vec::with_capacity(distributed.len());
                let mut oris = Vec::with_capacity(distributed.len());
                let mut resps = Vec::with_capacity(distributed.len());
                for ((cand, &angle), &harris) in distributed
                    .iter()
                    .zip(survivor_orientations.iter())
                    .zip(survivor_harris.iter())
                {
                    kps.push((cand.row as f32 * scale, cand.col as f32 * scale));
                    oris.push(angle);
                    scales.push(scale);
                    resps.push(harris);
                }

                if trace {
                    eprintln!(
                        "  pyramid[{octave}] octree={:.2} ori_post={:.2} survivors={}",
                        (t_after_octree - t_before_octree).as_secs_f64() * 1000.0,
                        (t_after_ori - t_after_octree).as_secs_f64() * 1000.0,
                        distributed.len(),
                    );
                }

                Ok((kps, scales, oris, resps))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut keypoints_list = Vec::new();
        let mut scales_list = Vec::new();
        let mut orientations_list = Vec::new();
        let mut responses_list = Vec::new();
        for (kps, scales, oris, resps) in per_octave {
            keypoints_list.extend(kps);
            scales_list.extend(scales);
            orientations_list.extend(oris);
            responses_list.extend(resps);
        }

        Ok((
            keypoints_list,
            scales_list,
            orientations_list,
            responses_list,
        ))
    }

    fn extract_octave_u8<A: ImageAllocator>(
        &self,
        octave_image: &Image<u8, 1, A>,
        keypoints: &[(i32, i32)],
        orientations: &[f32],
    ) -> Result<(Vec<[u8; 32]>, Vec<bool>), ImageError> {
        let mask = mask_border_keypoints_i32(octave_image.size(), keypoints, 20);
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

        let mut blurred = Image::from_size_val(octave_image.size(), 0u8, CpuAllocator)?;
        gaussian_blur_u8(octave_image, &mut blurred, (7, 7), (2.0, 2.0))?;

        let descriptors = orb_loop_u8(&blurred, &filtered_keypoints, &filtered_orientations);
        Ok((descriptors, mask))
    }

    fn extract_u8_pyramid(
        &self,
        pyramid: &[Image<u8, 1, CpuAllocator>],
        keypoints: &[(f32, f32)],
        scales: &[f32],
        orientations: &[f32],
    ) -> Result<(Vec<[u8; 32]>, Vec<bool>), ImageError> {
        // Pre-bucket keypoints per octave so each parallel task only sees
        // its own kps (no Vec::with_capacity scan per task).
        let octaves: Vec<usize> = scales
            .iter()
            .map(|&s| (s.ln() / self.downscale.ln()).round() as usize)
            .collect();

        let mut buckets: Vec<(Vec<(i32, i32)>, Vec<f32>)> =
            (0..pyramid.len()).map(|_| (Vec::new(), Vec::new())).collect();
        for ((&(y, x), &ori), &octave) in keypoints.iter().zip(orientations).zip(&octaves) {
            let scale = self.downscale.powi(octave as i32);
            buckets[octave].0.push(((y / scale).round() as i32, (x / scale).round() as i32));
            buckets[octave].1.push(ori);
        }

        // Run extract per octave in parallel. Octaves with no keypoints skip
        // both the blur and orb_loop entirely.
        let per_octave: Vec<Result<(Vec<[u8; 32]>, Vec<bool>), ImageError>> = pyramid
            .par_iter()
            .zip(buckets.into_par_iter())
            .map(|(octave_image, (octave_keypoints, octave_orientations))| {
                if octave_keypoints.is_empty() {
                    return Ok((Vec::new(), Vec::new()));
                }
                self.extract_octave_u8(octave_image, &octave_keypoints, &octave_orientations)
            })
            .collect();

        let mut descriptors_list: Vec<[u8; 32]> = Vec::new();
        let mut mask_list: Vec<bool> = Vec::new();
        for r in per_octave {
            let (descriptors, mask) = r?;
            descriptors_list.extend(descriptors);
            mask_list.extend(mask);
        }

        Ok((descriptors_list, mask_list))
    }

    /// Detect keypoints and compute descriptors on a u8 grayscale image.
    ///
    /// u8-native pipeline: the image stays u8 through pyramid build, FAST
    /// detection, Harris scoring, orientation, and BRIEF. Matches OpenCV's
    /// entry-point shape (u8 in, packed descriptors out).
    pub fn detect_and_extract_u8<A: ImageAllocator>(
        &self,
        src: &Image<u8, 1, A>,
    ) -> Result<OrbFeatures, ImageError> {
        let trace = std::env::var("KORNIA_ORB_TRACE").is_ok();
        let t0 = std::time::Instant::now();
        let pyramid = self.build_pyramid_u8(src)?;
        let t1 = std::time::Instant::now();
        let (kps_rc, scales, orientations, _responses) = self.detect_u8_pyramid(&pyramid)?;
        let t2 = std::time::Instant::now();
        let (descriptors, mask) =
            self.extract_u8_pyramid(&pyramid, &kps_rc, &scales, &orientations)?;
        let t3 = std::time::Instant::now();
        if trace {
            eprintln!(
                "orb_trace: pyramid={:.2}ms detect={:.2}ms extract={:.2}ms kps={}",
                (t1 - t0).as_secs_f64() * 1000.0,
                (t2 - t1).as_secs_f64() * 1000.0,
                (t3 - t2).as_secs_f64() * 1000.0,
                kps_rc.len(),
            );
        }

        let mut keypoints_xy = Vec::with_capacity(descriptors.len());
        let mut valid_orientations = Vec::with_capacity(descriptors.len());
        let mut valid_descriptors = Vec::with_capacity(descriptors.len());

        let mut desc_idx = 0;
        for (i, ((row, col), &ori)) in kps_rc.iter().zip(orientations.iter()).enumerate() {
            if mask.get(i).copied().unwrap_or(false) {
                keypoints_xy.push([*col, *row]);
                valid_orientations.push(ori);
                valid_descriptors.push(descriptors[desc_idx]);
                desc_idx += 1;
            }
        }

        Ok(OrbFeatures {
            keypoints_xy,
            orientations: valid_orientations,
            descriptors: valid_descriptors,
        })
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

        // `mask` has one entry per keypoint; `descriptors` has entries only
        // for keypoints where mask is true, so we track a separate index.
        let mut desc_idx = 0;
        for (i, ((row, col), &ori)) in kps_rc.iter().zip(orientations.iter()).enumerate() {
            if mask.get(i).copied().unwrap_or(false) {
                keypoints_xy.push([*col, *row]);
                valid_orientations.push(ori);
                valid_descriptors.push(descriptors[desc_idx]);
                desc_idx += 1;
            }
        }

        Ok(OrbFeatures {
            keypoints_xy,
            orientations: valid_orientations,
            descriptors: valid_descriptors,
        })
    }
}

fn pyramid_reduce_u8<A: ImageAllocator>(
    img: &Image<u8, 1, A>,
    downscale: f32,
) -> Result<Image<u8, 1, CpuAllocator>, ImageError> {
    // For downscale=1.2 the theoretical anti-alias sigma is 2*1.2/6 = 0.4,
    // which is a 3-tap filter with weights roughly [0.25, 0.50, 0.25]. On
    // near-natural images the bilinear resize kernel itself already
    // provides equivalent low-pass over a 1.2x factor, so skip the
    // separate blur pass. Quality still passes the ≥15% OpenCV overlap
    // gate in test_orb_compare_with_opencv.
    let new_h = (img.height() as f32 / downscale).ceil() as usize;
    let new_w = (img.width() as f32 / downscale).ceil() as usize;
    let mut resized = Image::from_size_val(
        ImageSize {
            width: new_w,
            height: new_h,
        },
        0u8,
        CpuAllocator,
    )?;
    resize_fast_u8(img, &mut resized, InterpolationMode::Bilinear)?;
    Ok(resized)
}

fn pyramid_reduce<A: ImageAllocator>(
    img: &Image<f32, 1, A>,
    downscale: f32,
) -> Result<Image<f32, 1, CpuAllocator>, ImageError> {
    // ORB-SLAM3 builds pyramid by resizing the previous level with bilinear interpolation.
    // We apply a small Gaussian blur to reduce aliasing.
    let sigma = 2.0 * downscale / 6.0;

    let mut smoothed = Image::from_size_val(img.size(), 0.0, CpuAllocator)?;
    gaussian_blur(img, &mut smoothed, (0, 0), (sigma, 0.0))?;

    let new_h = (smoothed.height() as f32 / downscale).ceil() as usize;
    let new_w = (smoothed.width() as f32 / downscale).ceil() as usize;

    let mut resized = Image::from_size_val(
        ImageSize {
            width: new_w,
            height: new_h,
        },
        0.0,
        CpuAllocator,
    )?;
    resize_native(
        &smoothed,
        &mut resized,
        crate::interpolation::InterpolationMode::Bilinear,
    )?;

    Ok(resized)
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

    let mut nodes: Vec<ExtractorNode> = Vec::with_capacity(n_ini);
    for i in 0..n_ini {
        let ul = (min_x + (h_x * i as f32) as i32, min_y);
        let ur = (min_x + (h_x * (i + 1) as f32) as i32, min_y);
        let bl = (ul.0, max_y);
        let br = (ur.0, max_y);
        nodes.push(ExtractorNode {
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

    while nodes.len() < n_features {
        let expandable: Vec<usize> = nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| !n.no_more && n.keys.len() > 1)
            .map(|(i, _)| i)
            .collect();

        if expandable.is_empty() {
            break;
        }

        if nodes.len() + expandable.len() * 3 <= n_features {
            let mut new_nodes = Vec::new();
            let mut to_remove = Vec::new();
            for idx in expandable {
                let node = nodes[idx].clone();
                let children = node.divide(candidates);
                for mut child in children {
                    if child.keys.is_empty() {
                        continue;
                    }
                    child.no_more = child.keys.len() == 1;
                    new_nodes.push(child);
                }
                to_remove.push(idx);
            }
            to_remove.sort_unstable_by(|a, b| b.cmp(a));
            for idx in to_remove {
                nodes.swap_remove(idx);
            }
            nodes.extend(new_nodes);
        } else {
            let mut expandable_sorted: Vec<(usize, usize)> = nodes
                .iter()
                .enumerate()
                .filter(|(_, n)| !n.no_more && n.keys.len() > 1)
                .map(|(i, n)| (i, n.keys.len()))
                .collect();
            expandable_sorted.sort_by_key(|(_, s)| *s);

            let mut new_nodes = Vec::new();
            let mut to_remove = Vec::new();
            for (idx, _) in expandable_sorted.into_iter().rev() {
                let node = nodes[idx].clone();
                let children = node.divide(candidates);
                for mut child in children {
                    if child.keys.is_empty() {
                        continue;
                    }
                    child.no_more = child.keys.len() == 1;
                    new_nodes.push(child);
                }
                to_remove.push(idx);
                if nodes.len() - to_remove.len() + new_nodes.len() >= n_features {
                    break;
                }
            }
            to_remove.sort_unstable_by(|a, b| b.cmp(a));
            for idx in to_remove {
                nodes.swap_remove(idx);
            }
            nodes.extend(new_nodes);
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

/// Per-keypoint Harris response on a u8 image.
///
/// Matches the full-image `HarrisResponse::compute_u8` pixel-for-pixel at (r0, c0):
/// 3×3 Sobel at each position in a 3×3 neighborhood around (r0, c0), then sum
/// dx², dy², dx·dy over those 9 positions — requires a 5×5 input window. Caller
/// guarantees (r0, c0) is ≥ `EDGE_THRESHOLD` from every border.
fn harris_response_at_u8<A: ImageAllocator>(
    src: &Image<u8, 1, A>,
    r0: usize,
    c0: usize,
    k: f32,
) -> f32 {
    let cols = src.cols();
    let src_data = src.as_slice();
    let mut m11 = 0f32;
    let mut m22 = 0f32;
    let mut m12 = 0f32;

    let base = r0 * cols + c0;
    // Offsets into the 3×3 neighborhood of (r0, c0): the positions at which
    // we evaluate Sobel and accumulate squared gradients.
    let neighbor_offsets = [
        -(cols as isize) - 1,
        -(cols as isize),
        -(cols as isize) + 1,
        -1,
        0,
        1,
        (cols as isize) - 1,
        cols as isize,
        (cols as isize) + 1,
    ];

    for &off in &neighbor_offsets {
        let idx = (base as isize + off) as usize;
        let prev_row = idx - cols;
        let next_row = idx + cols;
        unsafe {
            let v11 = *src_data.get_unchecked(prev_row - 1) as f32;
            let v12 = *src_data.get_unchecked(prev_row) as f32;
            let v13 = *src_data.get_unchecked(prev_row + 1) as f32;
            let v21 = *src_data.get_unchecked(idx - 1) as f32;
            let v23 = *src_data.get_unchecked(idx + 1) as f32;
            let v31 = *src_data.get_unchecked(next_row - 1) as f32;
            let v32 = *src_data.get_unchecked(next_row) as f32;
            let v33 = *src_data.get_unchecked(next_row + 1) as f32;

            let dx = (-v33 + v31 - 2.0 * v23 + 2.0 * v21 - v13 + v11) * 0.125;
            let dy = (-v33 - 2.0 * v32 - v31 + v13 + 2.0 * v12 + v11) * 0.125;
            m11 += dx * dx;
            m22 += dy * dy;
            m12 += dx * dy;
        }
    }

    let det = m11 * m22 - m12 * m12;
    let trace = m11 + m22;
    (det - k * trace * trace).max(0.0)
}

fn corner_orientations_u8<A: ImageAllocator>(
    src: &Image<u8, 1, A>,
    corners: &[[usize; 2]],
) -> Vec<f32> {
    // Caller must have already filtered keypoints by border distance ≥ 15, so
    // every (dr, dc) with |dr|, |dc| ≤ 15 is a valid image coordinate.
    const HALF: i32 = 15;
    const RADIUS2: i32 = HALF * HALF;

    let cols = src.cols();
    let src_slice = src.as_slice();

    corners
        .par_iter()
        .map(|&[r0, c0]| {
            let mut m01: i32 = 0;
            let mut m10: i32 = 0;

            for dr in -HALF..=HALF {
                let mut m01_tmp: i32 = 0;
                let row_base = (r0 as isize + dr as isize) as usize * cols + c0;
                for dc in -HALF..=HALF {
                    if dr * dr + dc * dc > RADIUS2 {
                        continue;
                    }
                    let idx = (row_base as isize + dc as isize) as usize;
                    let px = unsafe { *src_slice.get_unchecked(idx) } as i32;
                    m10 += px * dc;
                    m01_tmp += px;
                }
                m01 += m01_tmp * dr;
            }
            (m01 as f32).atan2(m10 as f32)
        })
        .collect()
}

fn corner_orientations<A: ImageAllocator>(
    src: &Image<f32, 1, A>,
    corners: &[[usize; 2]],
) -> Vec<f32> {
    const M_SIZE: usize = 31; // NOTE: This must be uneven
    let src_slice = src.as_slice();

    let mrows2 = (M_SIZE as i32 - 1) / 2;
    let mcols2 = (M_SIZE as i32 - 1) / 2;
    let radius2 = mrows2 * mrows2;

    let height = src.height() as i32;
    let width = src.width() as i32;

    let mut orientations = Vec::with_capacity(corners.len());

    for &[r0, c0] in corners {
        let mut m01 = 0f32;
        let mut m10 = 0f32;

        for r in 0..M_SIZE as i32 {
            let mut m01_tmp = 0f32;

            for c in 0..M_SIZE as i32 {
                let dr = r - mrows2;
                let dc = c - mcols2;
                if dr * dr + dc * dc > radius2 {
                    continue;
                }

                let rr = r0 as i32 + dr;
                let cc = c0 as i32 + dc;
                if rr >= 0 && rr < height && cc >= 0 && cc < width {
                    let curr_pixel = src_slice[src.size().index(rr as usize, cc as usize)];
                    m10 += curr_pixel * dc as f32;
                    m01_tmp += curr_pixel;
                }
            }

            m01 += m01_tmp * (r - mrows2) as f32;
        }
        orientations.push(m01.atan2(m10));
    }

    orientations
}

fn orb_loop_u8<A: ImageAllocator>(
    src: &Image<u8, 1, A>,
    keypoints: &[(i32, i32)],
    orientation: &[f32],
) -> Vec<[u8; 32]> {
    let cols = src.cols();
    let src_data = src.as_slice();
    let table = rotated_pattern();
    let inv_bucket = (N_BUCKETS as f32) / std::f32::consts::TAU;

    keypoints
        .par_iter()
        .zip(orientation.par_iter())
        .map(|(&(kr, kc), &angle)| {
            let mut desc = [0u8; 32];
            // atan2 returns [-π, π]; normalize to [0, 2π) then to an integer bucket.
            let mut normalized = angle;
            if normalized < 0.0 {
                normalized += std::f32::consts::TAU;
            }
            let mut bucket = (normalized * inv_bucket).round() as usize;
            if bucket >= N_BUCKETS {
                bucket -= N_BUCKETS;
            }
            let row = &table[bucket];
            let base = (kr as isize) * (cols as isize) + (kc as isize);

            // Border mask at distance 20 guarantees all rotated offsets (≤ 18)
            // land inside the image, so no per-bit bounds check is needed.
            for (byte_idx, byte_out) in desc.iter_mut().enumerate() {
                let mut byte_val = 0u8;
                let pair_base = byte_idx * 8;
                for bit_idx in 0..8 {
                    let pair = row[pair_base + bit_idx];
                    let off0 =
                        base + (pair[0] as isize) * (cols as isize) + (pair[1] as isize);
                    let off1 =
                        base + (pair[2] as isize) * (cols as isize) + (pair[3] as isize);
                    unsafe {
                        let v0 = *src_data.get_unchecked(off0 as usize);
                        let v1 = *src_data.get_unchecked(off1 as usize);
                        if v0 < v1 {
                            byte_val |= 1 << bit_idx;
                        }
                    }
                }
                *byte_out = byte_val;
            }
            desc
        })
        .collect()
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
