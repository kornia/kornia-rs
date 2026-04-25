use kornia_image::{allocator::ImageAllocator, Image, ImageError, ImageSize};
use kornia_tensor::CpuAllocator;
use rayon::prelude::*;

use crate::{
    features::{FastDetector, HarrisResponse},
    filter::{gaussian_blur, gaussian_blur_u8},
    interpolation::InterpolationMode,
    resize::{resize_fast_u8, resize_native},
};

use super::pattern::{POS0, POS1};

/// ORB features extracted from a single frame.
#[derive(Debug, Clone)]
pub struct OrbFeatures {
    /// Keypoints as `[col, row]` in full-resolution pixel coordinates.
    pub keypoints_xy: Vec<[f32; 2]>,
    /// Keypoint orientation angles (radians).
    pub orientations: Vec<f32>,
    /// Binary descriptors (256-bit, packed as 32 bytes each).
    pub descriptors: Vec<[u8; 32]>,
    /// Pyramid octave the keypoint was detected at (0 = full resolution,
    /// higher = coarser). BRIEF is scale-variant (the pair pattern is fixed
    /// in pixels) so ORB-SLAM3's matcher requires per-keypoint octave to
    /// reject cross-octave matches and scale the search radius.
    pub octaves: Vec<u8>,
}

impl OrbFeatures {
    /// Number of extracted features.
    pub fn len(&self) -> usize {
        self.keypoints_xy.len()
    }

    /// `true` when the feature set is empty.
    pub fn is_empty(&self) -> bool {
        self.keypoints_xy.is_empty()
    }

    /// Borrow as an [`OrbFeaturesView`] for consumption by the guided
    /// matcher or any other API that takes parallel-slice views.
    pub fn view(&self) -> crate::features::OrbFeaturesView<'_, 32> {
        crate::features::OrbFeaturesView {
            descriptors: &self.descriptors,
            keypoints_xy: &self.keypoints_xy,
            octaves: &self.octaves,
        }
    }
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
        // ORB-SLAM3 `DivideNode`: halfX = ceil((UR.x - UL.x) / 2), mid = UL.x + halfX.
        // Ceil (not floor) of half-width so odd-width nodes split deterministically;
        // keypoints at the centre pixel land in the right-hand child (matches spec).
        let half_x = ((self.ur.0 - self.ul.0) as f32 / 2.0).ceil() as i32;
        let half_y = ((self.bl.1 - self.ul.1) as f32 / 2.0).ceil() as i32;
        let mid_x = self.ul.0 + half_x;
        let mid_y = self.ul.1 + half_y;

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

        let base = img.size();
        let mut pyramid = Vec::with_capacity(self.n_scales);
        let mut current = img.clone();
        pyramid.push(current.clone());

        for level in 1..self.n_scales {
            let target = pyramid_size_at_level(base, self.downscale, level);
            if target == current.size() || target.width == 0 || target.height == 0 {
                break;
            }
            let next = pyramid_reduce(&current, target, self.downscale)?;
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

    #[allow(clippy::type_complexity)]
    fn detect_octave_u8(
        &self,
        octave: usize,
        octave_image: &Image<u8, 1, CpuAllocator>,
        features_per_level: &[usize],
        trace: bool,
    ) -> Result<(Vec<(f32, f32)>, Vec<f32>, Vec<f32>, Vec<f32>), ImageError> {
        const EDGE_THRESHOLD: usize = 19;
        let ta = std::time::Instant::now();
        // Selective nesting: oct 0 (>=1000 rows) is the single-core wall.
        // Let rayon steal into its row work from cores that finished smaller
        // octaves. Oct 1+ stays serial — spawn/join overhead of nested
        // par_iter across many small octaves regresses wall time.
        let mut candidates = if octave_image.height() >= 1000 {
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
            return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new()));
        }

        // Scale NMS cap by the octave's budget share (floor at 512 so
        // low-contrast octaves still feed the distributor a usable pool).
        let nms_cap = features_per_level[octave].saturating_mul(20).max(512);
        let t_before_nms = std::time::Instant::now();
        let kp_with_resp = crate::features::fast::suppress_direct_standalone(
            candidates,
            nms_cap,
            octave_image.width(),
            octave_image.height(),
            1,
        );
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
            return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new()));
        }
        let (keypoints, responses): (Vec<[usize; 2]>, Vec<f32>) = kp_with_resp.into_iter().unzip();

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
        let distributed = distribute_octree(&candidates, min_x, max_x, min_y, max_y, target);
        let t_after_octree = std::time::Instant::now();

        let survivor_positions: Vec<[usize; 2]> =
            distributed.iter().map(|c| [c.row, c.col]).collect();
        // Serial — outer pipeline already saturates cores; nesting oversubscribes.
        let survivor_harris: Vec<f32> = survivor_positions
            .iter()
            .map(|&[r, c]| harris_response_at_u8(octave_image, r, c, self.harris_k))
            .collect();
        let survivor_orientations = corner_orientations_u8(octave_image, &survivor_positions);
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
    }

    /// Build the pyramid while concurrently running detect+extract on each
    /// completed level.
    ///
    /// For level N, `process_octave_u8` is spawned on a rayon worker the
    /// moment level N is built; the main thread continues with
    /// `pyramid_reduce_u8` for level N+1. This overlaps the serial reduce
    /// chain with the full detect+extract work per octave — pyramid images
    /// are consumed by the spawned tasks and aren't returned.
    #[allow(clippy::type_complexity)]
    fn build_pyramid_and_process_u8<A: ImageAllocator>(
        &self,
        img: &Image<u8, 1, A>,
    ) -> Result<Vec<(Vec<[f32; 2]>, Vec<f32>, Vec<[u8; 32]>, Vec<u8>)>, ImageError> {
        use std::sync::{Arc, OnceLock};

        let trace = std::env::var("KORNIA_ORB_TRACE").is_ok();
        let features_per_level = self.features_per_level();

        let level0 = Image::from_size_slice(img.size(), img.as_slice(), CpuAllocator)?;
        let mut pyramid_arcs: Vec<Arc<Image<u8, 1, CpuAllocator>>> =
            Vec::with_capacity(self.n_scales);
        pyramid_arcs.push(Arc::new(level0));

        // OnceLock (not Mutex) — each slot is written by exactly one spawn and
        // read once after scope join, so there's no contention to guard.
        type ProcessedOctave = (Vec<[f32; 2]>, Vec<f32>, Vec<[u8; 32]>, Vec<u8>);
        let slots: Vec<OnceLock<Result<ProcessedOctave, ImageError>>> =
            (0..self.n_scales).map(|_| OnceLock::new()).collect();

        let mut build_err: Option<ImageError> = None;

        rayon::scope(|s| {
            // Kick level-0 process immediately — its detect is the heaviest
            // single task and can run in parallel with reduce(1..n).
            let arc0 = pyramid_arcs[0].clone();
            let slot0 = &slots[0];
            let fpl = &features_per_level;
            s.spawn(move |_| {
                let _ = slot0.set(self.process_octave_u8(0, arc0.as_ref(), fpl, trace));
            });

            let base = pyramid_arcs[0].size();
            for octave in 1..self.n_scales {
                let prev = pyramid_arcs[octave - 1].clone();
                let target = pyramid_size_at_level(base, self.downscale, octave);
                if target == prev.size() || target.width == 0 || target.height == 0 {
                    break;
                }
                let next = match pyramid_reduce_u8(prev.as_ref(), target) {
                    Ok(n) => n,
                    Err(e) => {
                        build_err = Some(e);
                        break;
                    }
                };
                pyramid_arcs.push(Arc::new(next));
                let arc = pyramid_arcs[octave].clone();
                let slot = &slots[octave];
                let fpl = &features_per_level;
                s.spawn(move |_| {
                    let _ = slot.set(self.process_octave_u8(octave, arc.as_ref(), fpl, trace));
                });
            }
        });

        if let Some(e) = build_err {
            return Err(e);
        }

        // Pyramid images fully consumed by the spawned tasks; drop the Arcs.
        let n_built = pyramid_arcs.len();
        drop(pyramid_arcs);

        let mut per_octave = Vec::with_capacity(n_built);
        for (i, slot) in slots.into_iter().take(n_built).enumerate() {
            match slot.into_inner() {
                Some(Ok(r)) => per_octave.push(r),
                Some(Err(e)) => return Err(e),
                None => unreachable!("slot {i} was not written"),
            }
        }

        Ok(per_octave)
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

        // Pre-BRIEF blur via the Q8+Q8 u8 NEON path: ≤1 LSB off a float
        // reference, which is well under the bit-flip threshold that would
        // drift the 256-bit Hamming distance.
        let mut blurred = Image::from_size_val(octave_image.size(), 0u8, CpuAllocator)?;
        gaussian_blur_u8(octave_image, &mut blurred, (7, 7), (2.0, 2.0))?;

        let descriptors = orb_loop_u8(&blurred, &filtered_keypoints, &filtered_orientations);
        Ok((descriptors, mask))
    }

    /// Detect + extract for one pyramid level in a single task.
    ///
    /// Returns the mask-filtered `(keypoints_xy_full_res, orientations,
    /// descriptors)` triple — ready to concatenate across octaves. Fusing
    /// the two phases here lets the pipeline overlap extract with the reduce
    /// chain and with detect on neighboring octaves.
    #[allow(clippy::type_complexity)]
    fn process_octave_u8(
        &self,
        octave: usize,
        octave_image: &Image<u8, 1, CpuAllocator>,
        features_per_level: &[usize],
        trace: bool,
    ) -> Result<(Vec<[f32; 2]>, Vec<f32>, Vec<[u8; 32]>, Vec<u8>), ImageError> {
        let (kps_full, _scales, oris, _resps) =
            self.detect_octave_u8(octave, octave_image, features_per_level, trace)?;
        if kps_full.is_empty() {
            return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new()));
        }

        // extract_octave_u8 wants integer octave-local coords.
        let scale = self.downscale.powi(octave as i32);
        let inv_scale = 1.0 / scale;
        let kps_local: Vec<(i32, i32)> = kps_full
            .iter()
            .map(|&(r, c)| {
                (
                    (r * inv_scale).round() as i32,
                    (c * inv_scale).round() as i32,
                )
            })
            .collect();

        let (descriptors, mask) = self.extract_octave_u8(octave_image, &kps_local, &oris)?;

        // Mask-filter kps+oris to align with the (already-filtered) descriptors.
        let oct_u8 = octave as u8;
        let mut keypoints_xy = Vec::with_capacity(descriptors.len());
        let mut orientations = Vec::with_capacity(descriptors.len());
        let mut octaves = Vec::with_capacity(descriptors.len());
        for (i, &m) in mask.iter().enumerate() {
            if m {
                let (r, c) = kps_full[i];
                keypoints_xy.push([c, r]);
                orientations.push(oris[i]);
                octaves.push(oct_u8);
            }
        }

        Ok((keypoints_xy, orientations, descriptors, octaves))
    }

    /// Detect keypoints and compute descriptors on a u8 grayscale image.
    ///
    /// u8-native pipeline: the image stays u8 through pyramid build, FAST
    /// detection, Harris scoring, orientation, and BRIEF. Takes u8 pixels
    /// in and returns packed 32-byte descriptors out.
    pub fn detect_and_extract_u8<A: ImageAllocator>(
        &self,
        src: &Image<u8, 1, A>,
    ) -> Result<OrbFeatures, ImageError> {
        let trace = std::env::var("KORNIA_ORB_TRACE").is_ok();
        let t0 = std::time::Instant::now();
        let per_octave = self.build_pyramid_and_process_u8(src)?;
        let t1 = std::time::Instant::now();

        let total: usize = per_octave.iter().map(|(k, _, _, _)| k.len()).sum();
        let mut keypoints_xy = Vec::with_capacity(total);
        let mut orientations = Vec::with_capacity(total);
        let mut descriptors = Vec::with_capacity(total);
        let mut octaves = Vec::with_capacity(total);
        for (kps, oris, descs, octs) in per_octave {
            keypoints_xy.extend(kps);
            orientations.extend(oris);
            descriptors.extend(descs);
            octaves.extend(octs);
        }

        if trace {
            eprintln!(
                "orb_trace: build_detect_extract={:.2}ms kps={}",
                (t1 - t0).as_secs_f64() * 1000.0,
                keypoints_xy.len(),
            );
        }

        Ok(OrbFeatures {
            keypoints_xy,
            orientations,
            descriptors,
            octaves,
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
        let mut valid_octaves = Vec::with_capacity(descriptors.len());

        let inv_log_ds = 1.0 / self.downscale.ln();

        // `mask` has one entry per keypoint; `descriptors` has entries only
        // for keypoints where mask is true, so we track a separate index.
        let mut desc_idx = 0;
        for (i, (((row, col), &ori), &s)) in kps_rc
            .iter()
            .zip(orientations.iter())
            .zip(scales.iter())
            .enumerate()
        {
            if mask.get(i).copied().unwrap_or(false) {
                keypoints_xy.push([*col, *row]);
                valid_orientations.push(ori);
                valid_descriptors.push(descriptors[desc_idx]);
                let oct = (s.ln() * inv_log_ds).round().max(0.0) as u8;
                valid_octaves.push(oct);
                desc_idx += 1;
            }
        }

        Ok(OrbFeatures {
            keypoints_xy,
            orientations: valid_orientations,
            descriptors: valid_descriptors,
            octaves: valid_octaves,
        })
    }
}

/// Target size for pyramid level `level` computed from the L0 dimensions.
///
/// Matches ORB-SLAM3 `ComputePyramid`: `sz = round(L0 * inv_scale^level)` using
/// banker's rounding (`round_ties_even`). Computing from L0 at every level (instead
/// of chaining `prev / scale`) avoids rounding error that iterative `ceil` accumulates —
/// at scale=1.2 on 752×480 the chained `ceil` path drifts up to 2 px by L7, which
/// nudges corners across FAST's edge-threshold and causes cross-octave descriptor
/// assignment to shift. Banker's rounding is the same convention used by
/// [`rotate_pattern_for_angle`].
#[inline]
fn pyramid_size_at_level(base: ImageSize, downscale: f32, level: usize) -> ImageSize {
    let inv_scale = (downscale as f64).powi(-(level as i32));
    let w = (base.width as f64 * inv_scale).round_ties_even() as usize;
    let h = (base.height as f64 * inv_scale).round_ties_even() as usize;
    ImageSize {
        width: w,
        height: h,
    }
}

fn pyramid_reduce_u8<A: ImageAllocator>(
    img: &Image<u8, 1, A>,
    target: ImageSize,
) -> Result<Image<u8, 1, CpuAllocator>, ImageError> {
    // For downscale=1.2 the theoretical anti-alias sigma is 2*1.2/6 = 0.4 —
    // a 3-tap filter with weights ≈ [0.25, 0.50, 0.25]. The bilinear resize
    // kernel itself already provides equivalent low-pass over a 1.2× factor
    // on near-natural images, so the separate blur pass is skipped.
    let mut resized = Image::from_size_val(target, 0u8, CpuAllocator)?;
    resize_fast_u8(img, &mut resized, InterpolationMode::Bilinear)?;
    Ok(resized)
}

fn pyramid_reduce<A: ImageAllocator>(
    img: &Image<f32, 1, A>,
    target: ImageSize,
    downscale: f32,
) -> Result<Image<f32, 1, CpuAllocator>, ImageError> {
    // ORB-SLAM3 builds pyramid by resizing the previous level with bilinear interpolation.
    // We apply a small Gaussian blur to reduce aliasing.
    let sigma = 2.0 * downscale / 6.0;

    let mut smoothed = Image::from_size_val(img.size(), 0.0, CpuAllocator)?;
    gaussian_blur(img, &mut smoothed, (0, 0), (sigma, 0.0))?;

    let mut resized = Image::from_size_val(target, 0.0, CpuAllocator)?;
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

    // ORB-SLAM3 `DistributeOctTree` expansion loop:
    //   1. Each iteration, divide EVERY expandable node in one pass (expand-all).
    //   2. Track only the newly-created still-expandable children in `last_expandables`.
    //   3. If size ≥ N or no progress → done.
    //   4. Else if projected size (before the pass) + nToExpand*3 > N, enter a sorted
    //      subloop that splits the largest-by-size of those freshly-created expandables
    //      until we hit N.
    //
    // Ordering and scope of these two phases matter: the expand-all pass runs
    // unconditionally each iteration, and the sorted subloop only sees nodes
    // produced by the immediately-preceding expand-all. A naïve implementation
    // that skips step 1 when it would overshoot, or lets step 4 pool all live
    // expandables, produces a different spatial partition → different best-response
    // pick per node → different final keypoint set.
    loop {
        if nodes.len() >= n_features {
            break;
        }

        let prev_size = nodes.len();

        // Phase 1: expand-all. Divide every non-terminal node with >1 keys.
        let expandable_now: Vec<usize> = nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| !n.no_more && n.keys.len() > 1)
            .map(|(i, _)| i)
            .collect();

        if expandable_now.is_empty() {
            break;
        }

        let n_to_expand = expandable_now.len();
        let mut new_nodes: Vec<ExtractorNode> = Vec::new();
        let mut new_expandable_offsets: Vec<usize> = Vec::new();
        for &idx in &expandable_now {
            let children = nodes[idx].divide(candidates);
            for mut child in children {
                if child.keys.is_empty() {
                    continue;
                }
                child.no_more = child.keys.len() == 1;
                if !child.no_more && child.keys.len() > 1 {
                    new_expandable_offsets.push(new_nodes.len());
                }
                new_nodes.push(child);
            }
        }

        let mut to_remove = expandable_now.clone();
        to_remove.sort_unstable_by(|a, b| b.cmp(a));
        for idx in to_remove {
            nodes.swap_remove(idx);
        }
        let append_base = nodes.len();
        nodes.extend(new_nodes);
        let mut last_expandables: Vec<usize> = new_expandable_offsets
            .into_iter()
            .map(|off| append_base + off)
            .collect();

        if nodes.len() >= n_features || nodes.len() == prev_size {
            break;
        }

        // Phase 2: if expand-all left us still short but a second expand-all
        // would overshoot, split the largest newly-created expandables one-by-one
        // (ORB-SLAM3 sorts ascending then iterates reversed → largest first).
        if prev_size + n_to_expand * 3 > n_features {
            loop {
                let round_prev_size = nodes.len();
                if last_expandables.is_empty() {
                    break;
                }

                // Sort ascending by keys.len(), iterate reversed (biggest first).
                let mut sorted: Vec<(usize, usize)> = last_expandables
                    .iter()
                    .map(|&i| (i, nodes[i].keys.len()))
                    .collect();
                sorted.sort_by_key(|(_, s)| *s);

                let mut to_remove: Vec<usize> = Vec::new();
                let mut new_nodes: Vec<ExtractorNode> = Vec::new();
                let mut next_expandable_offsets: Vec<usize> = Vec::new();
                for (idx, _) in sorted.into_iter().rev() {
                    let children = nodes[idx].divide(candidates);
                    for mut child in children {
                        if child.keys.is_empty() {
                            continue;
                        }
                        child.no_more = child.keys.len() == 1;
                        if !child.no_more && child.keys.len() > 1 {
                            next_expandable_offsets.push(new_nodes.len());
                        }
                        new_nodes.push(child);
                    }
                    to_remove.push(idx);
                    let projected_size = nodes.len() - to_remove.len() + new_nodes.len();
                    if projected_size >= n_features {
                        break;
                    }
                }

                to_remove.sort_unstable_by(|a, b| b.cmp(a));
                for idx in to_remove {
                    nodes.swap_remove(idx);
                }
                let append_base = nodes.len();
                nodes.extend(new_nodes);
                last_expandables = next_expandable_offsets
                    .into_iter()
                    .map(|off| append_base + off)
                    .collect();

                if nodes.len() >= n_features || nodes.len() == round_prev_size {
                    break;
                }
            }
            break;
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
///
/// Loads the shared 5×5 window once (25 bytes, tight in L1D), then runs the 9
/// Sobel evaluations against those values — the prior per-neighbor re-load did
/// 72 u8 reads per call for the same pixels. Bit-exact with the old path because
/// the Sobel expression and accumulation order are unchanged; only the source
/// of the pixel values (register cache vs re-read) differs.
fn harris_response_at_u8<A: ImageAllocator>(
    src: &Image<u8, 1, A>,
    r0: usize,
    c0: usize,
    k: f32,
) -> f32 {
    let cols = src.cols();
    let src_data = src.as_slice();
    let base = (r0 - 2) * cols + (c0 - 2);
    let mut w = [0f32; 25];
    unsafe {
        for dr in 0..5 {
            let row_start = base + dr * cols;
            for dc in 0..5 {
                w[dr * 5 + dc] = *src_data.get_unchecked(row_start + dc) as f32;
            }
        }
    }

    let mut m11 = 0f32;
    let mut m22 = 0f32;
    let mut m12 = 0f32;
    // 9 neighbor positions (ndr, ndc) ∈ {0,1,2}×{0,1,2} inside the 5×5 window
    // map to the original (r0-1..=r0+1, c0-1..=c0+1) locations. Each neighbor's
    // 3×3 Sobel stencil spans window rows (ndr..=ndr+2) × cols (ndc..=ndc+2).
    for ndr in 0..3 {
        for ndc in 0..3 {
            let v11 = w[ndr * 5 + ndc];
            let v12 = w[ndr * 5 + ndc + 1];
            let v13 = w[ndr * 5 + ndc + 2];
            let v21 = w[(ndr + 1) * 5 + ndc];
            let v23 = w[(ndr + 1) * 5 + ndc + 2];
            let v31 = w[(ndr + 2) * 5 + ndc];
            let v32 = w[(ndr + 2) * 5 + ndc + 1];
            let v33 = w[(ndr + 2) * 5 + ndc + 2];

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

/// Per-row disk mask half-widths for the ORB orientation disk (half_k = 15),
/// indexed by |dr|. Built via the two-pass symmetrization from Rublee 2011:
///   Pass 1 (v = 0..=11): `umax[v] = round(sqrt(225 − v²))`
///   Pass 2 (v = 15..=11, reverse): reflect via a `v0` counter that walks past
///     run-equal regions — forces the mask to be symmetric across the 45°
///     diagonal, which a naive Euclidean disk is *not*.
/// Final values: `[15,15,15,15,14,14,14,13,13,12,11,10,9,8,6,3]`. The
/// symmetrized disk is what makes the intensity-centroid orientation (and the
/// rotated BRIEF descriptors that depend on it) rotationally stable.
const UMAX: [i32; 16] = [15, 15, 15, 15, 14, 14, 14, 13, 13, 12, 11, 10, 9, 8, 6, 3];

/// Precomputed byte masks for the NEON orientation path. 31 rows × 32 bytes.
/// Each row covers dc ∈ [-16, 15] split into two 16-lane windows; a lane is
/// 0xFF iff dc ∈ [-UMAX[|dr|], UMAX[|dr|]]. Generated at compile time.
const fn build_disk_masks() -> [[u8; 32]; 31] {
    let mut m = [[0u8; 32]; 31];
    let mut dr = -15i32;
    while dr <= 15 {
        let v = if dr < 0 { -dr } else { dr };
        let d = UMAX[v as usize];
        let mut lane = 0i32;
        while lane < 32 {
            let dc = lane - 16;
            if dc >= -d && dc <= d {
                m[(dr + 15) as usize][lane as usize] = 0xFF;
            }
            lane += 1;
        }
        dr += 1;
    }
    m
}

const DISK_MASKS: [[u8; 32]; 31] = build_disk_masks();

const DC_LOW: [i16; 16] = [
    -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
];
const DC_HIGH: [i16; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn orientation_kp_neon(src_slice: &[u8], r0: usize, c0: usize, cols: usize) -> f32 {
    use std::arch::aarch64::*;

    let dc_low_0 = vld1q_s16(DC_LOW.as_ptr());
    let dc_low_1 = vld1q_s16(DC_LOW.as_ptr().add(8));
    let dc_high_0 = vld1q_s16(DC_HIGH.as_ptr());
    let dc_high_1 = vld1q_s16(DC_HIGH.as_ptr().add(8));

    let mut m01_acc: i32 = 0;
    let mut m10_sum = vdupq_n_s32(0);

    let base = src_slice.as_ptr();
    for dr in -15i32..=15 {
        let row_ptr = base.add(((r0 as isize + dr as isize) * cols as isize) as usize);
        let mask_row = DISK_MASKS[(dr + 15) as usize].as_ptr();
        let mask_low = vld1q_u8(mask_row);
        let mask_high = vld1q_u8(mask_row.add(16));

        // Two 16-byte loads centered at c0. Low: dc ∈ [-16, -1]. High: dc ∈ [0, 15].
        let px_low = vandq_u8(vld1q_u8(row_ptr.offset(c0 as isize - 16)), mask_low);
        let px_high = vandq_u8(vld1q_u8(row_ptr.add(c0)), mask_high);

        // m01 row sum: masked bytes widened-and-added across all lanes.
        let row_sum = (vaddlvq_u8(px_low) + vaddlvq_u8(px_high)) as i32;
        m01_acc += row_sum * dr;

        // m10: Σ px * dc. Widen u8 → u16, reinterpret as s16 (safe: values ≤ 255
        // fit in i16's positive range), multiply by dc using vmull_s16 to
        // produce i32 partial products, accumulate.
        let px_low_u16_0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(px_low)));
        let px_low_u16_1 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(px_low)));
        let px_high_u16_0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(px_high)));
        let px_high_u16_1 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(px_high)));

        let p0 = vmull_s16(vget_low_s16(px_low_u16_0), vget_low_s16(dc_low_0));
        let p1 = vmull_high_s16(px_low_u16_0, dc_low_0);
        let p2 = vmull_s16(vget_low_s16(px_low_u16_1), vget_low_s16(dc_low_1));
        let p3 = vmull_high_s16(px_low_u16_1, dc_low_1);
        let p4 = vmull_s16(vget_low_s16(px_high_u16_0), vget_low_s16(dc_high_0));
        let p5 = vmull_high_s16(px_high_u16_0, dc_high_0);
        let p6 = vmull_s16(vget_low_s16(px_high_u16_1), vget_low_s16(dc_high_1));
        let p7 = vmull_high_s16(px_high_u16_1, dc_high_1);

        m10_sum = vaddq_s32(m10_sum, vaddq_s32(vaddq_s32(p0, p1), vaddq_s32(p2, p3)));
        m10_sum = vaddq_s32(m10_sum, vaddq_s32(vaddq_s32(p4, p5), vaddq_s32(p6, p7)));
    }

    let m10 = vaddvq_s32(m10_sum);
    (m01_acc as f32).atan2(m10 as f32)
}

/// AVX2 mirror of [`orientation_kp_neon`]. Same disk-masked moment scan,
/// translated 1:1 to `__m128i`. NEON's `vmull_s16` lane-by-lane multiply
/// chain is replaced with `_mm_madd_epi16` which pairwise-multiplies 8 i16
/// lanes into 4 i32 sums — exactly equivalent after the final horizontal
/// reduction. Row-sum uses `_mm_sad_epu8(_, zero)` as the SAD-against-zero
/// horizontal-add idiom (NEON's `vaddlvq_u8` has no x86 single-instruction
/// equivalent).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn orientation_kp_avx2(src_slice: &[u8], r0: usize, c0: usize, cols: usize) -> f32 {
    use std::arch::x86_64::*;
    let zero = _mm_setzero_si128();

    let dc_low_0 = _mm_loadu_si128(DC_LOW.as_ptr() as *const __m128i);
    let dc_low_1 = _mm_loadu_si128(DC_LOW.as_ptr().add(8) as *const __m128i);
    let dc_high_0 = _mm_loadu_si128(DC_HIGH.as_ptr() as *const __m128i);
    let dc_high_1 = _mm_loadu_si128(DC_HIGH.as_ptr().add(8) as *const __m128i);

    let mut m01_acc: i32 = 0;
    let mut m10_sum = _mm_setzero_si128();

    let base = src_slice.as_ptr();
    for dr in -15i32..=15 {
        let row_ptr = base.add(((r0 as isize + dr as isize) * cols as isize) as usize);
        let mask_row = DISK_MASKS[(dr + 15) as usize].as_ptr();
        let mask_low = _mm_loadu_si128(mask_row as *const __m128i);
        let mask_high = _mm_loadu_si128(mask_row.add(16) as *const __m128i);

        let px_low = _mm_and_si128(
            _mm_loadu_si128(row_ptr.offset(c0 as isize - 16) as *const __m128i),
            mask_low,
        );
        let px_high = _mm_and_si128(
            _mm_loadu_si128(row_ptr.add(c0) as *const __m128i),
            mask_high,
        );

        // Row sum via SAD-against-zero: each 64-bit lane gets the sum of its
        // 8 source bytes; combine and reduce to a single i32.
        let sad_low = _mm_sad_epu8(px_low, zero);
        let sad_high = _mm_sad_epu8(px_high, zero);
        let r_v = _mm_add_epi64(sad_low, sad_high);
        let r_v = _mm_add_epi64(r_v, _mm_srli_si128::<8>(r_v));
        let row_sum = _mm_cvtsi128_si32(r_v);
        m01_acc += row_sum * dr;

        // m10: widen u8→i16 then `_mm_madd_epi16` pairs i16 lanes into i32.
        let px_low_lo = _mm_unpacklo_epi8(px_low, zero);
        let px_low_hi = _mm_unpackhi_epi8(px_low, zero);
        let px_high_lo = _mm_unpacklo_epi8(px_high, zero);
        let px_high_hi = _mm_unpackhi_epi8(px_high, zero);

        let p0 = _mm_madd_epi16(px_low_lo, dc_low_0);
        let p1 = _mm_madd_epi16(px_low_hi, dc_low_1);
        let p2 = _mm_madd_epi16(px_high_lo, dc_high_0);
        let p3 = _mm_madd_epi16(px_high_hi, dc_high_1);
        m10_sum = _mm_add_epi32(m10_sum, _mm_add_epi32(p0, p1));
        m10_sum = _mm_add_epi32(m10_sum, _mm_add_epi32(p2, p3));
    }

    let s = _mm_add_epi32(m10_sum, _mm_srli_si128::<8>(m10_sum));
    let s = _mm_add_epi32(s, _mm_srli_si128::<4>(s));
    let m10 = _mm_cvtsi128_si32(s);

    (m01_acc as f32).atan2(m10 as f32)
}

fn corner_orientations_u8<A: ImageAllocator>(
    src: &Image<u8, 1, A>,
    corners: &[[usize; 2]],
) -> Vec<f32> {
    // Caller must have already filtered keypoints by border distance ≥ 15.
    const HALF: i32 = 15;

    let cols = src.cols();
    let src_slice = src.as_slice();

    #[cfg(target_arch = "aarch64")]
    let use_neon = std::env::var("KORNIA_ORB_ORI_NEON").map_or(true, |v| v != "0");

    #[cfg(target_arch = "x86_64")]
    let use_avx2 = crate::simd::cpu_features().has_avx2;

    corners
        .par_iter()
        .map(|&[r0, c0]| {
            #[cfg(target_arch = "aarch64")]
            if use_neon {
                return unsafe { orientation_kp_neon(src_slice, r0, c0, cols) };
            }
            #[cfg(target_arch = "x86_64")]
            if use_avx2 {
                return unsafe { orientation_kp_avx2(src_slice, r0, c0, cols) };
            }

            // Standard ORB `ICAngle`: v=0 processed alone, then pairs (+v, -v)
            // together with half-width UMAX[v]. The mask is not a clean Euclidean
            // disk — see UMAX docstring for the symmetrization that enforces
            // 45°-diagonal symmetry.
            let mut m01: i32 = 0;
            let mut m10: i32 = 0;
            let center = r0 as isize * cols as isize + c0 as isize;

            for u in -HALF..=HALF {
                let idx = (center + u as isize) as usize;
                let px = unsafe { *src_slice.get_unchecked(idx) } as i32;
                m10 += u * px;
            }

            for v in 1..=HALF {
                let d = UMAX[v as usize];
                let mut v_sum: i32 = 0;
                let plus_base = center + (v as isize) * cols as isize;
                let minus_base = center - (v as isize) * cols as isize;
                for u in -d..=d {
                    let val_plus =
                        unsafe { *src_slice.get_unchecked((plus_base + u as isize) as usize) }
                            as i32;
                    let val_minus =
                        unsafe { *src_slice.get_unchecked((minus_base + u as isize) as usize) }
                            as i32;
                    v_sum += val_plus - val_minus;
                    m10 += u * (val_plus + val_minus);
                }
                m01 += v * v_sum;
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

    keypoints
        .par_iter()
        .zip(orientation.par_iter())
        .map(|(&(kr, kc), &angle)| {
            let base = (kr as isize) * (cols as isize) + (kc as isize);
            let mut rotated = [[0i8; 4]; 256];
            rotate_pattern_for_angle(angle, &mut rotated);
            compute_brief_descriptor(src_data, cols, &rotated, base)
        })
        .collect()
}

/// Rotate the BRIEF pattern for a single orientation and store as i8 offsets.
///
/// Per-keypoint continuous rotation (Rublee 2011). Cost: 256 pairs × 4 f32 muls
/// plus 4 rounds; shared `cos`/`sin` lifted out of the pair loop. A quantized
/// bucket table is not sufficient — 12°/bucket produces several bits of
/// Hamming distance on otherwise-matched keypoints.
#[inline]
fn rotate_pattern_for_angle(angle: f32, out: &mut [[i8; 4]; 256]) {
    // Banker's rounding: Rust's default ties-away-from-zero `.round()`
    // produces different integer offsets on exact-half coordinates.
    let sin_a = angle.sin();
    let cos_a = angle.cos();
    for j in 0..256 {
        let pr0 = POS0[j][0] as f32;
        let pc0 = POS0[j][1] as f32;
        let pr1 = POS1[j][0] as f32;
        let pc1 = POS1[j][1] as f32;
        out[j][0] = (sin_a * pr0 + cos_a * pc0).round_ties_even() as i8;
        out[j][1] = (cos_a * pr0 - sin_a * pc0).round_ties_even() as i8;
        out[j][2] = (sin_a * pr1 + cos_a * pc1).round_ties_even() as i8;
        out[j][3] = (cos_a * pr1 - sin_a * pc1).round_ties_even() as i8;
    }
}

/// Build one 32-byte BRIEF descriptor from 256 pair comparisons.
///
/// Border mask at distance 20 (applied upstream) guarantees all rotated
/// offsets (≤ 18) land inside the image, so no per-bit bounds check is
/// needed here.
#[inline(always)]
fn compute_brief_descriptor(
    src_data: &[u8],
    cols: usize,
    row: &[[i8; 4]; 256],
    base: isize,
) -> [u8; 32] {
    // NEON path: process 16 pair-comparisons per iteration, producing 2
    // output bytes at once. Replaces 16 `if v0 < v1 { b |= 1<<i }` branch
    // chains with one `vcltq_u8` + bit-mask AND + two `vaddv_u8` reductions.
    // Modest win (~5% at 640p, wash at 1080p) but proven bit-parity via
    // `test_brief_neon_matches_scalar`.
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        static BIT_MASK: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
        let bitmask = vld1q_u8(BIT_MASK.as_ptr());
        let cols_i = cols as isize;
        let base_ptr = src_data.as_ptr().offset(base);
        let mut v0_buf = [0u8; 16];
        let mut v1_buf = [0u8; 16];
        let mut desc = [0u8; 32];
        for chunk in 0..16usize {
            let pair_base = chunk * 16;
            for i in 0..16 {
                let pair = row[pair_base + i];
                let off0 = (pair[0] as isize) * cols_i + (pair[1] as isize);
                let off1 = (pair[2] as isize) * cols_i + (pair[3] as isize);
                *v0_buf.get_unchecked_mut(i) = *base_ptr.offset(off0);
                *v1_buf.get_unchecked_mut(i) = *base_ptr.offset(off1);
            }
            let v0 = vld1q_u8(v0_buf.as_ptr());
            let v1 = vld1q_u8(v1_buf.as_ptr());
            let cmp = vcltq_u8(v0, v1);
            let anded = vandq_u8(cmp, bitmask);
            desc[chunk * 2] = vaddv_u8(vget_low_u8(anded));
            desc[chunk * 2 + 1] = vaddv_u8(vget_high_u8(anded));
        }
        desc
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(target_arch = "x86_64")]
        if crate::simd::cpu_features().has_avx2 {
            return unsafe { compute_brief_descriptor_avx2(src_data, cols, row, base) };
        }
        compute_brief_descriptor_scalar(src_data, cols, row, base)
    }
}

/// AVX2 mirror of the inline NEON BRIEF path. Same 16-pair-per-iteration
/// structure: gather v0/v1, unsigned `<` compare via the
/// `_mm_subs_epu8` + `_mm_cmpeq_epi8` substitute (AVX2 has no native unsigned
/// lane-compare), AND with the per-byte bit-weight pattern, then horizontal
/// sum each 8-byte half via `_mm_sad_epu8(_, zero)` to produce two output
/// descriptor bytes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn compute_brief_descriptor_avx2(
    src_data: &[u8],
    cols: usize,
    row: &[[i8; 4]; 256],
    base: isize,
) -> [u8; 32] {
    use std::arch::x86_64::*;
    static BIT_MASK: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
    let bitmask = _mm_loadu_si128(BIT_MASK.as_ptr() as *const __m128i);
    let zero = _mm_setzero_si128();
    let ones = _mm_set1_epi8(-1i8);
    let cols_i = cols as isize;
    let base_ptr = src_data.as_ptr().offset(base);
    let mut v0_buf = [0u8; 16];
    let mut v1_buf = [0u8; 16];
    let mut desc = [0u8; 32];
    for chunk in 0..16usize {
        let pair_base = chunk * 16;
        for i in 0..16 {
            let pair = row[pair_base + i];
            let off0 = (pair[0] as isize) * cols_i + (pair[1] as isize);
            let off1 = (pair[2] as isize) * cols_i + (pair[3] as isize);
            *v0_buf.get_unchecked_mut(i) = *base_ptr.offset(off0);
            *v1_buf.get_unchecked_mut(i) = *base_ptr.offset(off1);
        }
        let v0 = _mm_loadu_si128(v0_buf.as_ptr() as *const __m128i);
        let v1 = _mm_loadu_si128(v1_buf.as_ptr() as *const __m128i);
        // (v0 < v1) byte-mask: subs_epu8(v1, v0) is 0 iff v1<=v0 (i.e. v0>=v1);
        // andnot with the 0xFF broadcast flips that to 0xFF iff v0<v1.
        let lt = _mm_andnot_si128(_mm_cmpeq_epi8(_mm_subs_epu8(v1, v0), zero), ones);
        let anded = _mm_and_si128(lt, bitmask);
        let sad = _mm_sad_epu8(anded, zero);
        desc[chunk * 2] = _mm_cvtsi128_si32(sad) as u8;
        desc[chunk * 2 + 1] = _mm_extract_epi32::<2>(sad) as u8;
    }
    desc
}

/// Pure-scalar BRIEF reference — kept so bit-parity against the NEON path
/// can be asserted in tests without compile-time dispatch.
#[inline]
#[allow(dead_code)]
fn compute_brief_descriptor_scalar(
    src_data: &[u8],
    cols: usize,
    row: &[[i8; 4]; 256],
    base: isize,
) -> [u8; 32] {
    let mut desc = [0u8; 32];
    for (byte_idx, byte_out) in desc.iter_mut().enumerate() {
        let mut byte_val = 0u8;
        let pair_base = byte_idx * 8;
        for bit_idx in 0..8 {
            let pair = row[pair_base + bit_idx];
            let off0 = base + (pair[0] as isize) * (cols as isize) + (pair[1] as isize);
            let off1 = base + (pair[2] as isize) * (cols as isize) + (pair[3] as isize);
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
}

/// Compute ORB descriptors packed into 32 bytes (256 bits) per keypoint —
/// the standard ORB-SLAM3 descriptor format.
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

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_orientation_neon_matches_scalar() {
        // Random u8 image, random keypoint positions — ensure NEON and scalar
        // produce bit-identical orientations.
        let w = 200usize;
        let h = 200usize;
        let mut data = vec![0u8; w * h];
        let mut s: u64 = 0x1234_5678_9abc_def0;
        for px in data.iter_mut() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *px = (s >> 32) as u8;
        }
        let img = Image::<u8, 1, _>::from_size_slice([w, h].into(), &data, CpuAllocator).unwrap();

        let corners: Vec<[usize; 2]> = (0..50)
            .map(|i| {
                let r = 20 + (i * 7) % (h - 40);
                let c = 20 + (i * 11) % (w - 40);
                [r, c]
            })
            .collect();

        std::env::set_var("KORNIA_ORB_ORI_NEON", "1");
        let neon = corner_orientations_u8(&img, &corners);
        std::env::set_var("KORNIA_ORB_ORI_NEON", "0");
        let scalar = corner_orientations_u8(&img, &corners);
        std::env::remove_var("KORNIA_ORB_ORI_NEON");

        for (i, (n, s)) in neon.iter().zip(scalar.iter()).enumerate() {
            // atan2 of identical integer moments is bit-identical.
            assert_eq!(n.to_bits(), s.to_bits(), "kp {i}: neon={n} scalar={s}");
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_brief_neon_matches_scalar() {
        use super::compute_brief_descriptor;
        use super::compute_brief_descriptor_scalar;
        use super::rotate_pattern_for_angle;

        let w = 300usize;
        let h = 300usize;
        let mut data = vec![0u8; w * h];
        let mut s: u64 = 0xdead_beef_cafe_babe;
        for px in data.iter_mut() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *px = (s >> 32) as u8;
        }

        // Sweep keypoints across the interior and a set of rotation angles.
        let angles: Vec<f32> = (0..32)
            .map(|k| (k as f32) * std::f32::consts::TAU / 32.0)
            .collect();
        for (angle_idx, &angle) in angles.iter().enumerate() {
            let mut rotated = [[0i8; 4]; 256];
            rotate_pattern_for_angle(angle, &mut rotated);
            for kp_idx in 0..20 {
                let kr = 30 + (kp_idx * 13) % (h - 60);
                let kc = 30 + (kp_idx * 17) % (w - 60);
                let base = (kr as isize) * (w as isize) + (kc as isize);
                let neon = compute_brief_descriptor(&data, w, &rotated, base);
                let scalar = compute_brief_descriptor_scalar(&data, w, &rotated, base);
                assert_eq!(
                    neon, scalar,
                    "mismatch at angle_idx={angle_idx} kp=({kr},{kc})"
                );
            }
        }
    }
}
