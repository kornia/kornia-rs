//! End-to-end SIFT: scale-space, detection, orientation and descriptors in one
//! device-resident pass.
//!
//! Scratch is allocated once and reused across frames, so a streaming caller
//! pays no per-call allocation. Everything stays in device memory until
//! [`SiftCudaFeatures::to_host`], which is the only synchronisation point.
//!
//! # Why the octave loop owns the whole pipeline
//!
//! Orientation and descriptors sample the octave's *Gaussian* layers, not the
//! DoG. Running detection for every octave first would mean keeping every
//! octave's Gaussian slab alive at once. Instead each octave is built, mined
//! for keypoints, oriented and described before moving on, so only one octave's
//! layers are ever resident.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};

use super::descriptor::{
    launch_sift_descriptor_cuda_view, launch_sift_pack_descriptor_input_cuda_view, DESCR_LEN,
    DESC_IN_STRIDE,
};
use super::detect::launch_sift_find_extrema_cuda_view;
use super::kernels::gaussian_kernel_f32;
use super::orientation::{launch_sift_orientation_cuda_view, ORI_KP_STRIDE};
use super::pyramid::{
    launch_sift_blur_h_tiled_cuda_view, launch_sift_blur_v_cuda_view,
    launch_sift_blur_v_dog_cuda_view, launch_sift_downsample_nearest_cuda_view,
    launch_sift_upsample2x_cuda_view,
};
use super::{gaussian_ksize, SiftCudaConfig, SiftCudaError, KP_STRIDE};

/// A detected, oriented and described keypoint, in input-image coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SiftKeypoint {
    /// Column, in input-image pixels.
    pub x: f32,
    /// Row, in input-image pixels.
    pub y: f32,
    /// Diameter of the meaningful neighbourhood.
    pub size: f32,
    /// Dominant gradient orientation, in degrees, clockwise from +x.
    pub angle: f32,
    /// Contrast at the refined extremum.
    pub response: f32,
    /// Packed `octave | (layer << 8) | (round((xi + 0.5) * 255) << 16)`.
    pub octave: i32,
}

/// Host-side result of a full detect-and-compute pass.
#[derive(Debug, Clone, Default)]
pub struct SiftFeatures {
    /// One entry per oriented keypoint.
    pub keypoints: Vec<SiftKeypoint>,
    /// Row-major `keypoints.len() * 128` descriptor block.
    pub descriptors: Vec<f32>,
}

impl SiftFeatures {
    /// Number of keypoints.
    pub fn len(&self) -> usize {
        self.keypoints.len()
    }
    /// Whether any keypoint was found.
    pub fn is_empty(&self) -> bool {
        self.keypoints.is_empty()
    }
}

/// Which scale the pyramid starts from.
///
/// `Double` matches OpenCV, COLMAP and VLFeat, which all upsample 2x before
/// building the pyramid; it roughly doubles the keypoint count and costs about
/// 3.6x the time. `Native` starts at the input resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FirstOctave {
    /// `first_octave = -1`: upsample 2x first (the reference default).
    Double,
    /// `first_octave = 0`: start at the input resolution.
    Native,
}

/// Reusable device-resident SIFT pipeline for one image size.
pub struct SiftCuda {
    cfg: SiftCudaConfig,
    first_octave: FirstOctave,
    /// Hard cap on octaves; the geometric tail past ~4 contributes almost
    /// nothing (measured: 0.8% of correct matches, <1% of runtime).
    max_octaves: usize,
    width: usize,
    height: usize,
    /// Ping-pong planes sized for the largest octave.
    buf_a: CudaSlice<f32>,
    buf_b: CudaSlice<f32>,
    /// `n_octave_layers + 3` Gaussian layers of the current octave, one buffer
    /// each: the vertical-blur-plus-DoG pass needs layer `i-1` shared and layer
    /// `i` mutable at the same time, which `split_at_mut` gives only across
    /// separate allocations.
    gauss: Vec<CudaSlice<f32>>,
    /// `n_octave_layers + 2` DoG layers of the current octave.
    dog: CudaSlice<f32>,
    kp: CudaSlice<f32>,
    kp_count: CudaSlice<i32>,
    ori_kp: CudaSlice<f32>,
    ori_count: CudaSlice<i32>,
    desc_in: CudaSlice<f32>,
    desc: CudaSlice<f32>,
    base_kernel: Vec<f32>,
    layer_kernels: Vec<Vec<f32>>,
}

impl SiftCuda {
    /// Allocate the pipeline for a fixed input size.
    pub fn new(
        _ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        width: usize,
        height: usize,
        cfg: SiftCudaConfig,
        first_octave: FirstOctave,
        max_octaves: usize,
    ) -> Result<Self, SiftCudaError> {
        if width == 0 || height == 0 {
            return Err(SiftCudaError::Geometry(
                "image dimensions must be non-zero".into(),
            ));
        }
        if cfg.max_keypoints == 0 {
            return Err(SiftCudaError::Geometry(
                "max_keypoints must be non-zero".into(),
            ));
        }
        let (bw, bh) = match first_octave {
            FirstOctave::Double => (width * 2, height * 2),
            FirstOctave::Native => (width, height),
        };
        let plane = bw * bh;
        let n_layers = cfg.n_octave_layers + 3;
        let n_dog = cfg.n_octave_layers + 2;

        let sigmas = cfg.layer_sigmas();
        let base_sigma = cfg.base_sig_diff() as f64;
        let base_kernel = gaussian_kernel_f32(gaussian_ksize(base_sigma), base_sigma);
        let layer_kernels = (1..n_layers)
            .map(|i| gaussian_kernel_f32(gaussian_ksize(sigmas[i]), sigmas[i]))
            .collect();

        // One oriented keypoint can emit several angles; the reference caps at
        // 4 dominant peaks in practice, so size the oriented buffer accordingly.
        let ori_cap = cfg.max_keypoints * 4;
        Ok(Self {
            cfg,
            first_octave,
            max_octaves,
            width,
            height,
            buf_a: stream.alloc_zeros::<f32>(plane)?,
            buf_b: stream.alloc_zeros::<f32>(plane)?,
            gauss: (0..n_layers)
                .map(|_| stream.alloc_zeros::<f32>(plane))
                .collect::<Result<Vec<_>, _>>()?,
            dog: stream.alloc_zeros::<f32>(plane * n_dog)?,
            kp: stream.alloc_zeros::<f32>(cfg.max_keypoints * KP_STRIDE)?,
            kp_count: stream.alloc_zeros::<i32>(1)?,
            ori_kp: stream.alloc_zeros::<f32>(ori_cap * ORI_KP_STRIDE)?,
            ori_count: stream.alloc_zeros::<i32>(1)?,
            desc_in: stream.alloc_zeros::<f32>(ori_cap * DESC_IN_STRIDE)?,
            desc: stream.alloc_zeros::<f32>(ori_cap * DESCR_LEN)?,
            base_kernel,
            layer_kernels,
        })
    }

    /// Number of octaves this configuration will build.
    fn n_octaves(&self, bw: usize, bh: usize) -> usize {
        self.cfg.n_octaves(bw.min(bh)).min(self.max_octaves)
    }

    /// Detect, orient and describe, returning host-side results.
    ///
    /// `src` is a `width * height` f32 grayscale image in 0..255, matching the
    /// reference's internal representation.
    pub fn detect_and_compute(
        &mut self,
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        src: &CudaSlice<f32>,
    ) -> Result<SiftFeatures, SiftCudaError> {
        let need = self.width * self.height;
        if src.len() < need {
            return Err(SiftCudaError::SliceTooSmall {
                got: src.len(),
                need,
            });
        }
        let n_layers = self.cfg.n_octave_layers + 3;
        let n_dog = self.cfg.n_octave_layers + 2;
        // KORNIA_SIFT_STAGES=1 breaks the pass down; each probe synchronises,
        // so the total is inflated -- read the ratios, not the absolutes.
        let probe = std::env::var("KORNIA_SIFT_STAGES").is_ok();
        let mut t_blur = 0.0f64;
        let mut t_det = 0.0f64;
        let mut t_ori = 0.0f64;
        let mut t_desc = 0.0f64;
        let mut t_copy = 0.0f64;
        let mark = |on: bool| -> Option<std::time::Instant> { on.then(std::time::Instant::now) };
        let since = |t: Option<std::time::Instant>, stream: &Arc<CudaStream>, acc: &mut f64| {
            if let Some(t) = t {
                stream.synchronize().ok();
                *acc += t.elapsed().as_secs_f64() * 1e3;
            }
        };

        // ── Base image of the first octave ──────────────────────────────────
        let (mut cw, mut ch) = match self.first_octave {
            FirstOctave::Double => {
                launch_sift_upsample2x_cuda_view(
                    ctx,
                    stream,
                    &src.slice(0..need),
                    &mut self.buf_a.slice_mut(0..self.width * 2 * self.height * 2),
                    self.width as u32,
                    self.height as u32,
                )?;
                (self.width * 2, self.height * 2)
            }
            FirstOctave::Native => {
                stream.memcpy_dtod(src, &mut self.buf_a)?;
                (self.width, self.height)
            }
        };
        launch_sift_blur_h_tiled_cuda_view(
            ctx,
            stream,
            &self.buf_a.slice(0..cw * ch),
            &mut self.buf_b.slice_mut(0..cw * ch),
            cw as u32,
            ch as u32,
            &self.base_kernel,
        )?;
        launch_sift_blur_v_cuda_view(
            ctx,
            stream,
            &self.buf_b.slice(0..cw * ch),
            &mut self.gauss[0].slice_mut(0..cw * ch),
            cw as u32,
            ch as u32,
            &self.base_kernel,
        )?;

        let n_oct = self.n_octaves(cw, ch);
        let mut all_kps: Vec<SiftKeypoint> = Vec::new();
        let mut all_desc: Vec<f32> = Vec::new();

        for octv in 0..n_oct {
            if cw < 16 || ch < 16 {
                break;
            }
            let plane = cw * ch;

            // ── Gaussian layers and their DoGs ──────────────────────────────
            let tb = mark(probe);
            for i in 1..n_layers {
                let gk = &self.layer_kernels[i - 1];
                launch_sift_blur_h_tiled_cuda_view(
                    ctx,
                    stream,
                    &self.gauss[i - 1].slice(0..plane),
                    &mut self.buf_b.slice_mut(0..plane),
                    cw as u32,
                    ch as u32,
                    gk,
                )?;
                // blur-V writes layer `i` and the DoG against layer `i-1` from
                // the same registers, so the difference costs no extra pass.
                let (lo_half, hi_half) = self.gauss.split_at_mut(i);
                launch_sift_blur_v_dog_cuda_view(
                    ctx,
                    stream,
                    &self.buf_b.slice(0..plane),
                    &mut hi_half[0].slice_mut(0..plane),
                    &lo_half[i - 1].slice(0..plane),
                    &mut self.dog.slice_mut((i - 1) * plane..i * plane),
                    cw as u32,
                    ch as u32,
                    gk,
                )?;
            }

            since(tb, stream, &mut t_blur);

            // ── Detect, orient, describe for this octave ────────────────────
            let td = mark(probe);
            stream.memset_zeros(&mut self.kp_count)?;
            for layer in 1..=self.cfg.n_octave_layers {
                launch_sift_find_extrema_cuda_view(
                    ctx,
                    stream,
                    &self.cfg,
                    &self.dog.slice(0..plane * n_dog),
                    &mut self.kp.as_view_mut(),
                    &mut self.kp_count.as_view_mut(),
                    cw as u32,
                    ch as u32,
                    n_dog as u32,
                    layer as u32,
                    octv as u32,
                )?;
            }
            since(td, stream, &mut t_det);
            let n_kp = stream.clone_dtoh(&self.kp_count)?[0].max(0) as usize;
            let n_kp = n_kp.min(self.cfg.max_keypoints);
            if n_kp > 0 {
                // Orientation and descriptors read the Gaussian layer the
                // keypoint was found in, so group by layer.
                let raw = stream.clone_dtoh(&self.kp.slice(0..n_kp * KP_STRIDE))?;
                for layer in 1..=self.cfg.n_octave_layers {
                    let idx: Vec<usize> = (0..n_kp)
                        .filter(|&i| raw[i * KP_STRIDE + 5].to_bits() as i32 == layer as i32)
                        .collect();
                    if idx.is_empty() {
                        continue;
                    }
                    let mut packed = Vec::with_capacity(idx.len() * KP_STRIDE);
                    for &i in &idx {
                        packed.extend_from_slice(&raw[i * KP_STRIDE..(i + 1) * KP_STRIDE]);
                    }
                    let d_in = stream.clone_htod(&packed)?;
                    let img = self.gauss[layer].slice(0..plane);

                    let to = mark(probe);
                    stream.memset_zeros(&mut self.ori_count)?;
                    launch_sift_orientation_cuda_view(
                        ctx,
                        stream,
                        &self.cfg,
                        &img,
                        cw as u32,
                        ch as u32,
                        &d_in.as_view(),
                        idx.len() as u32,
                        KP_STRIDE as u32,
                        &mut self.ori_kp.as_view_mut(),
                        &mut self.ori_count.as_view_mut(),
                    )?;
                    since(to, stream, &mut t_ori);
                    let n_ori = stream.clone_dtoh(&self.ori_count)?[0].max(0) as usize;
                    let cap = self.ori_kp.len() / ORI_KP_STRIDE;
                    let n_ori = n_ori.min(cap);
                    if n_ori == 0 {
                        continue;
                    }
                    let tds = mark(probe);
                    // The orientation record is in the pyramid base's frame; the
                    // descriptor works in this octave's. Every octave rescales
                    // by 1 / (1 << octv) -- for `Double` the loop index and the
                    // reference's signed octave differ by one, but so do the
                    // stored position and size, and the two offsets cancel.
                    launch_sift_pack_descriptor_input_cuda_view(
                        ctx,
                        stream,
                        &self.ori_kp.as_view(),
                        n_ori as u32,
                        ORI_KP_STRIDE as u32,
                        5,
                        1.0 / ((1u32 << octv) as f32),
                        &mut self.desc_in.as_view_mut(),
                    )?;
                    launch_sift_descriptor_cuda_view(
                        ctx,
                        stream,
                        &img,
                        cw as u32,
                        ch as u32,
                        &self.desc_in.as_view(),
                        n_ori as u32,
                        DESC_IN_STRIDE as u32,
                        &mut self.desc.as_view_mut(),
                    )?;

                    since(tds, stream, &mut t_desc);
                    let tc = mark(probe);
                    // Copy back only the rows this launch actually wrote.
                    // The buffers are sized for `max_keypoints * 4` oriented
                    // keypoints, so downloading them whole moves ~17 MB per
                    // layer per octave -- ~400 MB an image, which dominated
                    // the end-to-end time by an order of magnitude.
                    let ok = stream.clone_dtoh(&self.ori_kp.slice(0..n_ori * ORI_KP_STRIDE))?;
                    let od = stream.clone_dtoh(&self.desc.slice(0..n_ori * DESCR_LEN))?;
                    // first_octave = -1 post-processing: halve position and
                    // size, and rewrite the packed octave byte.
                    let scale = match self.first_octave {
                        FirstOctave::Double => 0.5f32,
                        FirstOctave::Native => 1.0f32,
                    };
                    since(tc, stream, &mut t_copy);
                    for r in 0..n_ori {
                        let o = &ok[r * ORI_KP_STRIDE..(r + 1) * ORI_KP_STRIDE];
                        let packed_oct = o[4].to_bits() as i32;
                        let oct = match self.first_octave {
                            FirstOctave::Double => (packed_oct & !255) | ((packed_oct - 1) & 255),
                            FirstOctave::Native => packed_oct,
                        };
                        all_kps.push(SiftKeypoint {
                            x: o[0] * scale,
                            y: o[1] * scale,
                            size: o[2] * scale,
                            angle: o[5],
                            response: o[3],
                            octave: oct,
                        });
                        all_desc.extend_from_slice(&od[r * DESCR_LEN..(r + 1) * DESCR_LEN]);
                    }
                }
            }

            // ── Next octave base: stride-2 subsample of layer n_octave_layers ─
            let (nw, nh) = (cw / 2, ch / 2);
            if nw == 0 || nh == 0 || octv + 1 >= n_oct {
                break;
            }
            launch_sift_downsample_nearest_cuda_view(
                ctx,
                stream,
                &self.gauss[self.cfg.n_octave_layers].slice(0..plane),
                &mut self.buf_a.slice_mut(0..nw * nh),
                cw as u32,
                ch as u32,
                nw as u32,
                nh as u32,
            )?;
            {
                let (src_base, l0) = (self.buf_a.slice(0..nw * nh), &mut self.gauss[0]);
                let mut dst = l0.slice_mut(0..nw * nh);
                stream.memcpy_dtod(&src_base, &mut dst)?;
            }
            cw = nw;
            ch = nh;
        }

        if probe {
            eprintln!(
                "  stages: blur={t_blur:.1} detect={t_det:.1} orient={t_ori:.1} \
                 descriptor={t_desc:.1} copyback={t_copy:.1} (ms)"
            );
        }
        Ok(SiftFeatures {
            keypoints: all_kps,
            descriptors: all_desc,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::color::test_utils::default_stream;

    fn load_dump(path: &str) -> Option<(usize, usize, Vec<f32>)> {
        let b = std::fs::read(path).ok()?;
        let rows = i32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
        let cols = i32::from_le_bytes(b[4..8].try_into().unwrap()) as usize;
        let data = b[8..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .take(rows * cols)
            .collect();
        Some((rows, cols, data))
    }

    /// Reference `(x, y)` bit patterns from the oracle's own `detectAndCompute`.
    fn ref_positions(dir: &str) -> Vec<(u32, u32)> {
        let b = std::fs::read(format!("{dir}/keypoints.bin")).expect("keypoints");
        let n = i32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
        (0..n)
            .map(|i| {
                let o = 4 + i * 24;
                let f = |k: usize| {
                    f32::from_le_bytes(b[o + k * 4..o + k * 4 + 4].try_into().unwrap()).to_bits()
                };
                (f(0), f(1))
            })
            .collect()
    }

    #[test]
    fn end_to_end_matches_reference_keypoints() {
        let Some(dir) = std::env::var("KORNIA_SIFT_ORACLE")
            .ok()
            .and_then(|v| v.split(':').next().map(String::from))
        else {
            eprintln!("KORNIA_SIFT_ORACLE unset; skipping");
            return;
        };
        let (h, w, img) = load_dump(&format!("{dir}/gray_fpt.f32")).expect("gray_fpt");
        let stream = default_stream();
        let ctx = &stream.context();
        let d_src = stream.clone_htod(&img).unwrap();

        let mut plan = SiftCuda::new(
            ctx,
            &stream,
            w,
            h,
            SiftCudaConfig::default(),
            FirstOctave::Double,
            usize::MAX,
        )
        .unwrap();
        let feats = plan.detect_and_compute(ctx, &stream, &d_src).unwrap();

        let want: std::collections::HashSet<(u32, u32)> = ref_positions(&dir).into_iter().collect();
        let got: std::collections::HashSet<(u32, u32)> = feats
            .keypoints
            .iter()
            .map(|k| (k.x.to_bits(), k.y.to_bits()))
            .collect();
        let hit = got.intersection(&want).count();
        eprintln!(
            "  end-to-end: produced={} unique_pos={} reference_pos={} exact_pos_match={}",
            feats.len(),
            got.len(),
            want.len(),
            hit
        );
        assert_eq!(
            feats.descriptors.len(),
            feats.len() * DESCR_LEN,
            "descriptor block must be one 128-vector per keypoint"
        );
        assert!(!feats.is_empty(), "pipeline produced no keypoints");
        // The reference additionally de-duplicates and retains the best N; this
        // pipeline does not, so compare coverage of the reference set rather
        // than requiring equal counts.
        let cover = hit as f64 / want.len() as f64;
        assert!(
            cover > 0.9,
            "only {:.1}% of reference positions found",
            cover * 100.0
        );
    }
}
