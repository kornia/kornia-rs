//! End-to-end SIFT: scale-space, detection, orientation and descriptors in one
//! device-resident pass.
//!
//! Scratch is allocated once and reused across frames, so a streaming caller
//! pays no per-call allocation. Everything stays in device memory until
//! [`SiftCudaFeatures::to_host`], which is the only synchronisation point.
//!
//! # Why the whole pyramid stays resident
//!
//! Orientation and descriptors sample the octave's *Gaussian* layers, not the
//! DoG, so an earlier revision built, mined and described one octave at a time
//! to keep only one octave's layers alive.
//!
//! That ordering is wrong, and it cost more than it saved. The reference applies
//! `retainBest` **before** computing descriptors (`sift.dispatch.cpp:568-600`);
//! describing inside the octave loop means the keypoint budget cannot be applied
//! until every descriptor has already been computed. Measured: `n_features` of
//! 200 or 2515 both took 18.3 ms, i.e. 2315 descriptors were computed and thrown
//! away.
//!
//! Keeping every octave resident costs far less than it appears: the octave
//! sizes are a geometric series, so the whole pyramid is only ~1.33x the octave-0
//! slab that was already allocated — about +12 MB at `fo=-1` on 752x480.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};

use super::descriptor::{
    launch_sift_descriptor_cuda_view, launch_sift_gather_descriptors_cuda_view, DESCR_LEN,
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
    /// Gaussian layers for every octave: `pyr[octave][layer]`.
    ///
    /// Resident for the whole frame so descriptors can run after the keypoint
    /// budget has been applied. See the module docs.
    pyr: Vec<Vec<CudaSlice<f32>>>,
    /// `n_octave_layers + 2` DoG layers of the current octave.
    dog: CudaSlice<f32>,
    kp: CudaSlice<f32>,
    kp_count: CudaSlice<i32>,
    ori_kp: CudaSlice<f32>,
    ori_count: CudaSlice<i32>,
    desc_in: CudaSlice<f32>,
    /// Descriptors for the whole frame, in detection order. Each launch writes
    /// straight into its own row range, so there is no per-layer staging copy.
    desc_all: CudaSlice<f32>,
    /// The same descriptors in final keypoint order — what callers see.
    desc_out: CudaSlice<f32>,
    perm: CudaSlice<i32>,
    /// Survivor count for the deferred descriptor pass.
    desc_live: CudaSlice<i32>,
    /// Row where each (octave, layer) group's oriented keypoints start. Written
    /// device-to-device so no count ever has to come back to the host mid-frame.
    ranges: CudaSlice<i32>,
    n_desc: usize,
    /// Opt-in rotated-frame descriptor: faster, not bit-exact. See
    /// [`super::descriptor`].
    fast_descriptor: bool,
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
        // Same selector the CPU path uses, so the two agree on what is invalid.
        cfg.shared_config()
            .validate(max_octaves)
            .map_err(|e| SiftCudaError::Geometry(e.to_string()))?;
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
        if cfg.n_octave_layers == 0 {
            return Err(SiftCudaError::Geometry(
                "n_octave_layers must be non-zero".into(),
            ));
        }
        if max_octaves == 0 {
            return Err(SiftCudaError::Geometry(
                "max_octaves must be non-zero".into(),
            ));
        }
        // `gaussian_kernel_f32` asserts on a non-positive sigma, and these values
        // come straight from a public constructor (including the Python one), so
        // reject them here rather than panicking across the FFI boundary.
        if !(cfg.sigma.is_finite() && cfg.sigma > 0.0) {
            return Err(SiftCudaError::Geometry(format!(
                "sigma must be finite and positive, got {}",
                cfg.sigma
            )));
        }
        let (bw, bh) = match first_octave {
            FirstOctave::Double => (width * 2, height * 2),
            FirstOctave::Native => (width, height),
        };
        let plane = bw * bh;
        let n_layers = cfg.n_octave_layers + 3;
        // Every octave stays resident (see the module docs), so size the pyramid
        // for the octave count this geometry can actually reach. The sizes are a
        // geometric series, so this is ~1.33x one octave's slab, not Nx it.
        let n_oct_max = cfg
            .n_octaves_for(
                bw.min(bh),
                if first_octave == FirstOctave::Double {
                    -1
                } else {
                    0
                },
            )
            .min(max_octaves)
            .max(1);
        let n_dog = cfg.n_octave_layers + 2;

        let sigmas = cfg.layer_sigmas();
        // The doubled and native branches of the reference's `createInitialImage`
        // remove different multiples of the assumed input blur; using the doubled
        // constant on the native path under-blurs every pyramid layer.
        let base_sigma = cfg.base_sig_diff_for(first_octave == FirstOctave::Double) as f64;
        let base_kernel = gaussian_kernel_f32(gaussian_ksize(base_sigma), base_sigma);
        let layer_kernels = (1..n_layers)
            .map(|i| gaussian_kernel_f32(gaussian_ksize(sigmas[i]), sigmas[i]))
            .collect();

        // One detected keypoint can emit several angles. The reference imposes
        // no hard cap — every bin that is a strict local maximum of the
        // 36-bin smoothed histogram AND at least `ORI_PEAK_RATIO` of its peak
        // is emitted, so up to 18 in the pathological (near-flat histogram)
        // case. Four is the empirical average, not a bound: this is a capacity,
        // and a frame that exceeds it drops the surplus (the orientation
        // kernel's `slot < max_out` guard) rather than overrunning anything.
        let ori_cap = cfg.max_keypoints * 4;
        Ok(Self {
            cfg,
            first_octave,
            max_octaves,
            width,
            height,
            buf_a: stream.alloc_zeros::<f32>(plane)?,
            buf_b: stream.alloc_zeros::<f32>(plane)?,
            pyr: (0..n_oct_max)
                .map(|o| {
                    let op = (bw >> o).max(1) * (bh >> o).max(1);
                    (0..n_layers)
                        .map(|_| stream.alloc_zeros::<f32>(op))
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<Vec<_>, _>>()?,
            dog: stream.alloc_zeros::<f32>(plane * n_dog)?,
            kp: stream.alloc_zeros::<f32>(cfg.max_keypoints * KP_STRIDE)?,
            kp_count: stream.alloc_zeros::<i32>(1)?,
            ori_kp: stream.alloc_zeros::<f32>(ori_cap * ORI_KP_STRIDE)?,
            ori_count: stream.alloc_zeros::<i32>(1)?,
            desc_in: stream.alloc_zeros::<f32>(ori_cap * DESC_IN_STRIDE)?,
            desc_all: stream.alloc_zeros::<f32>(ori_cap * DESCR_LEN)?,
            desc_out: stream.alloc_zeros::<f32>(ori_cap * DESCR_LEN)?,
            perm: stream.alloc_zeros::<i32>(ori_cap)?,
            desc_live: stream.alloc_zeros::<i32>(1)?,
            ranges: stream.alloc_zeros::<i32>(64 * (cfg.n_octave_layers + 1))?,
            n_desc: 0,
            fast_descriptor: false,
            base_kernel,
            layer_kernels,
        })
    }

    /// Select the rotated-frame descriptor kernel.
    ///
    /// Faster, and its cost does not grow with keypoint scale, but the
    /// descriptors are a sampling approximation of `cv::SIFT`'s rather than a
    /// reproduction. Detection, orientation and the scale space are unaffected.
    pub fn set_fast_descriptor(&mut self, on: bool) {
        self.fast_descriptor = on;
    }

    /// Number of octaves this configuration will build.
    ///
    /// The reference's count carries a `- first_octave` term, so the doubled and
    /// native paths differ by one octave even before the base image is resized.
    fn n_octaves(&self, bw: usize, bh: usize) -> usize {
        let first_octave = match self.first_octave {
            FirstOctave::Double => -1,
            FirstOctave::Native => 0,
        };
        self.cfg
            .n_octaves_for(bw.min(bh), first_octave)
            .min(self.max_octaves)
    }

    /// Detect, orient and describe, leaving the descriptors on device.
    ///
    /// `src` is a `width * height` f32 grayscale image in 0..255, matching the
    /// reference's internal representation. Returns the keypoints; the matching
    /// descriptor rows are reachable through
    /// [`SiftCuda::descriptors_device`], so a caller that goes straight on to
    /// matching never moves them across the bus.
    pub fn detect_and_compute_device(
        &mut self,
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        src: &CudaSlice<f32>,
    ) -> Result<Vec<SiftKeypoint>, SiftCudaError> {
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
        // Read once: this is a per-frame path and `env::var` allocates.
        static PROBE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let probe = *PROBE.get_or_init(|| std::env::var("KORNIA_SIFT_STAGES").is_ok());
        let mut t_blur = 0.0f64;
        let mut t_det = 0.0f64;
        let mut t_ori = 0.0f64;
        let mut t_desc = 0.0f64;
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
                // Slice both sides: `src` is only required to be *at least*
                // `need` long, and `memcpy_dtod` asserts `dst.len() >= src.len()`
                // — an over-long input would panic instead of copying.
                stream.memcpy_dtod(&src.slice(0..need), &mut self.buf_a.slice_mut(0..need))?;
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
            &mut self.pyr[0][0].slice_mut(0..cw * ch),
            cw as u32,
            ch as u32,
            &self.base_kernel,
        )?;

        let n_oct = self.n_octaves(cw, ch);
        let mut range_i = 0usize;
        // The deferred descriptor pass needs each octave's dimensions; the loop
        // halves with integer division, which is not `>>` for odd sizes.
        let mut oct_dims: Vec<(usize, usize)> = Vec::with_capacity(n_oct);
        stream.memset_zeros(&mut self.ori_count)?;

        for octv in 0..n_oct {
            // Deliberate divergence from the reference, which has no lower bound
            // and keeps halving. Below 16 px the interior left by the 5-px
            // detection border is a handful of pixels wide and the octave
            // contributes essentially nothing, while the orientation kernel
            // needs `h >= 3` to have any samples at all.
            if cw < 16 || ch < 16 {
                break;
            }
            let plane = cw * ch;
            oct_dims.push((cw, ch));

            // ── Gaussian layers and their DoGs ──────────────────────────────
            let tb = mark(probe);
            for i in 1..n_layers {
                let gk = &self.layer_kernels[i - 1];
                launch_sift_blur_h_tiled_cuda_view(
                    ctx,
                    stream,
                    &self.pyr[octv][i - 1].slice(0..plane),
                    &mut self.buf_b.slice_mut(0..plane),
                    cw as u32,
                    ch as u32,
                    gk,
                )?;
                // blur-V writes layer `i` and the DoG against layer `i-1` from
                // the same registers, so the difference costs no extra pass.
                let (lo_half, hi_half) = self.pyr[octv].split_at_mut(i);
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
                // keypoint was found in, so each launch is given one layer and
                // skips the keypoints that do not belong to it.
                //
                // Oriented keypoints accumulate across the WHOLE frame rather
                // than being reset per layer, and each layer's row range is
                // recorded on device (`ranges`) with a 4-byte device-to-device
                // copy. The descriptor launches size their grid from an upper
                // bound and retire the blocks past the live count. That is what
                // removes the per-layer count read: every one of those was a
                // blocking D2H that drained the stream, 54 of them a frame, and
                // they left no octave able to overlap the next.
                for layer in 1..=self.cfg.n_octave_layers {
                    let img = self.pyr[octv][layer].slice(0..plane);
                    // Snapshot where this layer's rows will start.
                    {
                        let src = self.ori_count.slice(0..1);
                        let mut dst = self.ranges.slice_mut(range_i..range_i + 1);
                        stream.memcpy_dtod(&src, &mut dst)?;
                    }

                    let to = mark(probe);
                    launch_sift_orientation_cuda_view(
                        ctx,
                        stream,
                        &self.cfg,
                        &img,
                        cw as u32,
                        ch as u32,
                        &self.kp.slice(0..n_kp * KP_STRIDE),
                        n_kp as u32,
                        KP_STRIDE as u32,
                        &mut self.ori_kp.as_view_mut(),
                        &mut self.ori_count.as_view_mut(),
                        layer as i32,
                        self.fast_descriptor,
                    )?;
                    since(to, stream, &mut t_ori);

                    range_i += 1;
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
                &self.pyr[octv][self.cfg.n_octave_layers].slice(0..plane),
                &mut self.buf_a.slice_mut(0..nw * nh),
                cw as u32,
                ch as u32,
                nw as u32,
                nh as u32,
            )?;
            {
                let (src_base, l0) = (self.buf_a.slice(0..nw * nh), &mut self.pyr[octv + 1][0]);
                let mut dst = l0.slice_mut(0..nw * nh);
                stream.memcpy_dtod(&src_base, &mut dst)?;
            }
            cw = nw;
            ch = nh;
        }

        // One download for the whole frame. The packed octave field already
        // carries the octave and layer, so the host can reconstruct everything
        // from this single copy instead of one per layer per octave.
        let n_ori = stream.clone_dtoh(&self.ori_count)?[0].max(0) as usize;
        let n_ori = n_ori
            .min(self.ori_kp.len() / ORI_KP_STRIDE)
            .min(self.desc_all.len() / DESCR_LEN);
        let ok = stream.clone_dtoh(&self.ori_kp.slice(0..n_ori * ORI_KP_STRIDE))?;

        // first_octave = -1 post-processing: halve position and size, and
        // rewrite the packed octave byte.
        let scale = match self.first_octave {
            FirstOctave::Double => 0.5f32,
            FirstOctave::Native => 1.0f32,
        };
        let all_kps: Vec<SiftKeypoint> = (0..n_ori)
            .map(|r| {
                let o = &ok[r * ORI_KP_STRIDE..(r + 1) * ORI_KP_STRIDE];
                let packed_oct = o[4].to_bits() as i32;
                SiftKeypoint {
                    x: o[0] * scale,
                    y: o[1] * scale,
                    size: o[2] * scale,
                    angle: o[5],
                    response: o[3],
                    octave: match self.first_octave {
                        FirstOctave::Double => (packed_oct & !255) | ((packed_oct - 1) & 255),
                        FirstOctave::Native => packed_oct,
                    },
                }
            })
            .collect();

        // Decide the final order on the host, then apply it to the descriptors
        // on device -- they are never downloaded just to be shuffled.
        let order = final_order(&all_kps, self.cfg.n_features);
        let n = order.len().min(n_ori);
        let keypoints: Vec<SiftKeypoint> = order[..n].iter().map(|&i| all_kps[i]).collect();

        // ── Descriptors, for the survivors only ─────────────────────────────
        //
        // This is why the pyramid stays resident. The reference applies
        // `retainBest` before `calcDescriptors`; describing inside the octave
        // loop meant computing every descriptor and discarding the ones the
        // budget cut. Grouped by (octave, layer) because each launch reads one
        // Gaussian layer, then permuted back into the caller's order by the
        // gather that was already here.
        let tds = mark(probe);
        if n > 0 {
            // Stable sort keeps each group in the order the octave loop
            // produced, so the descriptor sees the same rows it always did.
            let mut describe: Vec<usize> = (0..n).collect();
            describe.sort_by_key(|&i| {
                let packed = ok[order[i] * ORI_KP_STRIDE + 4].to_bits() as i32;
                ((packed & 255) as u8, ((packed >> 8) & 255) as u8)
            });

            // Pack on the host: four f32 ops per keypoint, the same expressions
            // and the same order as `sift_pack_desc_input`, so the values are
            // identical. Doing it here removes a kernel launch per layer and
            // the device-side range bookkeeping it needed.
            let mut din = vec![0.0f32; n * DESC_IN_STRIDE];
            for (pos, &i) in describe.iter().enumerate() {
                let k = &ok[order[i] * ORI_KP_STRIDE..(order[i] + 1) * ORI_KP_STRIDE];
                let packed = k[4].to_bits() as i32;
                let scale = 1.0f32 / ((1u32 << (packed & 255)) as f32);
                let mut ang = 360.0f32 - k[5];
                if (ang - 360.0).abs() < f32::EPSILON {
                    ang = 0.0;
                }
                let o = &mut din[pos * DESC_IN_STRIDE..pos * DESC_IN_STRIDE + 4];
                o[0] = k[0] * scale;
                o[1] = k[1] * scale;
                o[2] = (k[2] * scale) * 0.5;
                o[3] = ang;
            }
            stream.memcpy_htod(&din, &mut self.desc_in.slice_mut(0..n * DESC_IN_STRIDE))?;

            // One launch per contiguous (octave, layer) run.
            let key = |i: usize| {
                let packed = ok[order[i] * ORI_KP_STRIDE + 4].to_bits() as i32;
                ((packed & 255) as usize, ((packed >> 8) & 255) as usize)
            };
            let mut starts: Vec<i32> = Vec::new();
            let mut groups: Vec<(usize, usize, usize, usize)> = Vec::new();
            let mut p0 = 0usize;
            while p0 < n {
                let (oct, layer) = key(describe[p0]);
                let mut p1 = p0 + 1;
                while p1 < n && key(describe[p1]) == (oct, layer) {
                    p1 += 1;
                }
                if oct < oct_dims.len() && layer < self.pyr[oct].len() {
                    starts.push(p0 as i32);
                    groups.push((oct, layer, p0, p1 - p0));
                }
                p0 = p1;
            }
            let n_i = [n as i32];
            stream.memcpy_htod(&n_i, &mut self.desc_live.slice_mut(0..1))?;
            if !starts.is_empty() {
                let cap = self.ranges.len().min(starts.len());
                stream.memcpy_htod(&starts[..cap], &mut self.ranges.slice_mut(0..cap))?;
                for (g, &(oct, layer, _, len)) in groups.iter().enumerate().take(cap) {
                    let (gw, gh) = oct_dims[oct];
                    launch_sift_descriptor_cuda_view(
                        ctx,
                        stream,
                        &self.pyr[oct][layer].slice(0..gw * gh),
                        gw as u32,
                        gh as u32,
                        &self.desc_in.as_view(),
                        len as u32,
                        DESC_IN_STRIDE as u32,
                        &mut self.desc_all.as_view_mut(),
                        self.fast_descriptor,
                        &self.ranges.slice(g..g + 1),
                        &self.desc_live.slice(0..1),
                    )?;
                }
            }
            since(tds, stream, &mut t_desc);

            // `describe` reordered the rows; put them back in caller order.
            let mut perm = vec![0i32; n];
            for (pos, &i) in describe.iter().enumerate() {
                perm[i] = pos as i32;
            }
            stream.memcpy_htod(&perm, &mut self.perm.slice_mut(0..n))?;
            let src = self.desc_all.slice(0..n * DESCR_LEN);
            let p = self.perm.slice(0..n);
            let mut out = self.desc_out.slice_mut(0..n * DESCR_LEN);
            launch_sift_gather_descriptors_cuda_view(ctx, stream, &src, &p, n as u32, &mut out)?;
        }
        if probe {
            eprintln!(
                "  stages: blur={t_blur:.1} detect={t_det:.1} orient={t_ori:.1} \
                 descriptor={t_desc:.1} (ms)"
            );
        }
        self.n_desc = n;
        Ok(keypoints)
    }

    /// Detect, orient and describe, returning host-side results.
    ///
    /// This is [`SiftCuda::detect_and_compute_device`] plus a single download of
    /// the ordered descriptor block. A caller that goes straight on to matching
    /// should use the device form and skip the round trip entirely.
    pub fn detect_and_compute(
        &mut self,
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        src: &CudaSlice<f32>,
    ) -> Result<SiftFeatures, SiftCudaError> {
        let keypoints = self.detect_and_compute_device(ctx, stream, src)?;
        let descriptors = if self.n_desc == 0 {
            Vec::new()
        } else {
            stream.clone_dtoh(&self.desc_out.slice(0..self.n_desc * DESCR_LEN))?
        };
        Ok(SiftFeatures {
            keypoints,
            descriptors,
        })
    }

    /// The ordered descriptor block from the last call, on device.
    ///
    /// Row `i` belongs to keypoint `i` of the returned keypoint list. Valid
    /// until the next call, which overwrites it.
    pub fn descriptors_device(&self) -> &CudaSlice<f32> {
        &self.desc_out
    }

    /// Number of descriptor rows written by the last call.
    pub fn descriptor_count(&self) -> usize {
        self.n_desc
    }
}

/// The reference's final keypoint order: `removeDuplicatedSorted`, then
/// `retainBest`. Returns indices into the appended order.
fn final_order(kps: &[SiftKeypoint], n_features: usize) -> Vec<usize> {
    let deduped = sorted_dedup_order(kps);
    retain_best_order(kps, deduped, n_features)
}

/// Keep the `n` highest-response keypoints, matching `KeyPointsFilter::retainBest`.
///
/// `n == 0` means unlimited, as it does in `cv::SIFT::create`.
///
/// The subtlety worth preserving is the boundary: the reference partitions on
/// `response >= keypoints[n-1].response` **after** selecting, so every keypoint
/// tied with the cut-off survives and the result can be longer than `n`.
/// Truncating at exactly `n` would drop an arbitrary member of a tie group,
/// which is precisely the case the reference goes out of its way to avoid.
///
/// This is the faithful response-rank cut. It clusters keypoints on
/// high-contrast texture and thins the periphery — for pose estimation a
/// spatially-binned variant (as `orb::ExtractorNode::divide` does here) spreads
/// correspondences better, but it would not be what `cv2` returns.
fn retain_best_order(kps: &[SiftKeypoint], order: Vec<usize>, n: usize) -> Vec<usize> {
    if n == 0 || order.len() <= n {
        return order;
    }
    let mut rank = order.clone();
    // Descending response, index breaking ties so the choice is reproducible.
    rank.sort_by(|&a, &b| kps[b].response.total_cmp(&kps[a].response).then(a.cmp(&b)));
    let cutoff = kps[rank[n - 1]].response;
    order
        .into_iter()
        .filter(|&i| kps[i].response >= cutoff)
        .collect()
}

/// Order keypoints the way `KeyPointsFilter::removeDuplicatedSorted` does, and
/// drop the exact duplicates it drops.
///
/// Two reasons this is not optional. The detector appends through an atomic
/// counter, so without it the row order varies run to run even though the set
/// does not — surprising for anything that caches indices or diffs two runs.
/// And the reference genuinely removes duplicates: one extremum can be reached
/// from neighbouring start pixels and land on the same refined point.
///
/// The comparator is the reference's, including its descending fields: `size`,
/// `response` and `octave` sort the opposite way to `x`, `y` and `angle`.
fn sorted_dedup_order(kps: &[SiftKeypoint]) -> Vec<usize> {
    if kps.is_empty() {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..kps.len()).collect();
    order.sort_by(|&a, &b| {
        let (p, q) = (&kps[a], &kps[b]);
        p.x.total_cmp(&q.x)
            .then(p.y.total_cmp(&q.y))
            .then(q.size.total_cmp(&p.size))
            .then(p.angle.total_cmp(&q.angle))
            .then(q.response.total_cmp(&p.response))
            .then(q.octave.cmp(&p.octave))
            .then(a.cmp(&b))
    });

    // The reference's duplicate test is NOT the full record: it compares only
    // `pt.x`, `pt.y`, `size` and `angle`, so two keypoints that agree on those
    // but differ in `response` or `octave` are still duplicates to it. Deriving
    // this from `PartialEq` on the whole struct would keep such a pair.
    let same = |a: &SiftKeypoint, b: &SiftKeypoint| {
        a.x == b.x && a.y == b.y && a.size == b.size && a.angle == b.angle
    };
    let mut out: Vec<usize> = Vec::with_capacity(order.len());
    for &i in &order {
        // Adjacent-equal only: the sort has already grouped duplicates.
        if out.last().is_some_and(|&p| same(&kps[p], &kps[i])) {
            continue;
        }
        out.push(i);
    }
    out
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

    /// Unconditional device smoke test for the assembled pipeline. The bitwise
    /// end-to-end test below needs an oracle dump, so without this nothing
    /// exercised the descriptor launches at all — which is how a kernel that
    /// ignored `range_start` (every layer overwriting row 0) survived.
    ///
    /// Both descriptor kernels are covered: `fast` is a parameter, and the
    /// `KORNIA_SIFT_DESC=exact` kernel shares the row-range contract asserted
    /// here.
    /// The mirror of `features::sift::pipeline::rejects_the_same_configurations_as_cuda`:
    /// both backends route through `SiftConfig::validate`, so the set of
    /// rejected configurations is residency-independent rather than
    /// hand-mirrored.
    #[test]
    fn rejects_the_same_configurations_as_cpu() {
        let stream = default_stream();
        let ctx = &stream.context();
        let base = SiftCudaConfig::default();
        let mk = |c: SiftCudaConfig, m: usize| {
            SiftCuda::new(ctx, &stream, 64, 64, c, FirstOctave::Native, m).is_ok()
        };
        assert!(mk(base, usize::MAX));
        assert!(!mk(base, 0), "max_octaves = 0 must be rejected");
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let c = SiftCudaConfig { sigma: bad, ..base };
            assert!(!mk(c, usize::MAX), "sigma {bad} accepted");
        }
        let c = SiftCudaConfig {
            n_octave_layers: 0,
            ..base
        };
        assert!(!mk(c, usize::MAX));
    }

    #[test]
    fn assembled_pipeline_fills_every_descriptor_row() {
        let stream = default_stream();
        let ctx = stream.context();
        let (w, h) = (256usize, 192usize);
        let host: Vec<f32> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32, (i / w) as f32);
                128.0 + 100.0 * ((x * 0.37).sin() * (y * 0.29).cos())
            })
            .collect();
        let d_src = stream.clone_htod(&host).expect("upload");
        let mut plan = SiftCuda::new(
            ctx,
            &stream,
            w,
            h,
            SiftCudaConfig::default(),
            FirstOctave::Double,
            8,
        )
        .expect("plan");
        for fast in [false, true] {
            plan.set_fast_descriptor(fast);
            let f = plan
                .detect_and_compute(ctx, &stream, &d_src)
                .expect("detect");
            assert!(f.len() > 10, "expected keypoints, got {}", f.len());
            assert_eq!(f.descriptors.len(), f.len() * DESCR_LEN);
            // Every row must have been written. An all-zero row means its
            // launch retired the block or wrote somewhere else.
            for (i, row) in f.descriptors.chunks_exact(DESCR_LEN).enumerate() {
                assert!(
                    row.iter().any(|v| *v != 0.0),
                    "descriptor row {i} is all zero (fast={fast})"
                );
            }
        }
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
        // Coverage, not equality: the default descriptor kernel accumulates its
        // histogram with shared-memory float atomics, so the last bits of a few
        // descriptors — and with them the odd `retainBest` tie — are not
        // reproducible. Positions are; that is what is compared here.
        let cover = hit as f64 / want.len() as f64;
        assert!(
            cover > 0.9,
            "only {:.1}% of reference positions found",
            cover * 100.0
        );
    }
}
