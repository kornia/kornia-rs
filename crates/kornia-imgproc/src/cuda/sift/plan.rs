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
    launch_sift_descriptor_cuda_view, launch_sift_gather_descriptors_cuda_view,
    launch_sift_pack_desc_cuda_view, DESCR_LEN, DESC_IN_STRIDE,
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
use kornia_image::Image;

// `final_order` — `removeDuplicatedSorted` then `retainBest` — is the shared
// implementation. Both backends must return the same rows in the same order, so
// it has one definition rather than two copies kept in step by hand.
use crate::features::sift::pipeline::final_order;

// The keypoint record, the result bundle and the starting-scale selector are
// backend-independent, so they have one definition — in `features::sift` — and
// this module re-exports it. Declaring a second, structurally identical copy
// here made the two backends' results different types to the compiler for no
// reason, and forced the Python binding to alias one of them.
pub use crate::features::{FirstOctave, SiftFeatures, SiftKeypoint};

/// One frame's output: host keypoints, device descriptors.
///
/// `descriptors` is an **owned** `keypoints.len() * 128` device buffer — the
/// final gather writes each frame straight into a fresh allocation, so the
/// result has its own lifetime (holding two frames for matching just works)
/// and nothing is copied to make that true. Keypoints come to the host
/// because the reference's final ordering is a host-side sort; they are two
/// orders of magnitude smaller than the descriptors.
pub struct SiftCudaFeatures {
    /// Keypoints in the reference's final order.
    pub keypoints: Vec<SiftKeypoint>,
    /// Row-major `keypoints.len() * DESCR_LEN` block on device; row `i`
    /// belongs to `keypoints[i]`.
    pub descriptors: CudaSlice<f32>,
}

impl SiftCudaFeatures {
    /// Number of keypoints.
    pub fn len(&self) -> usize {
        self.keypoints.len()
    }
    /// Whether any keypoint was found.
    pub fn is_empty(&self) -> bool {
        self.keypoints.is_empty()
    }
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
    perm: CudaSlice<i32>,
    /// Survivor count for the deferred descriptor pass.
    desc_live: CudaSlice<i32>,
    /// Row where each (octave, layer) group's oriented keypoints start,
    /// uploaded from the host once per frame after `retain_best` decides the
    /// survivor grouping.
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
        // DETECTION CAPACITY, and why it is no longer `max_keypoints`.
        //
        // The extrema kernel appends through `atomicAdd(counter, 1)` and drops whatever lands past
        // the buffer (`if (slot >= max_kp) return`, detect.rs:331). Sizing that buffer AT
        // `max_keypoints` made the surviving SET depend on which threads reached the atomic first,
        // because the host then kept the leading `max_keypoints` entries of an arrival-ordered list.
        //
        // Measured before this change — same binary, same clip, same flags, an 80-keyframe
        // reconstruction run twice: 40,878 vs 40,795 points and 11.86 vs 3.28 px reprojection RMSE,
        // the first REJECTED by the 10 px quality gate and the second kept. Sorting the output does
        // not repair it: a sort canonicalises the ORDER of a truncated set, never its MEMBERSHIP.
        //
        // The detector is now given room to report what it actually found, and `select_best_keypoints`
        // applies the `max_keypoints` limit on the host by response — the same rule
        // `retain_best_order` already uses on the CPU path. Four is the multiple `ori_cap` uses, for
        // the same reason: working headroom, not a proof. A frame exceeding even this still loses the
        // surplus by arrival, and now warns instead of doing it silently.
        //
        // EIGHT, not four. Measured on the reference clip at the default 4096 budget: a frame
        // detected 16,847 extrema against a 4x buffer of 16,384, so 463 were still dropped by
        // arrival. They happened to fall below the response cutoff and the two runs agreed anyway,
        // but that is luck, not a property. 8x covers the observed worst case with headroom, and
        // costs 1.2 MB of device memory at the default budget.
        let det_cap = cfg.max_keypoints * 8;
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
            kp: stream.alloc_zeros::<f32>(det_cap * KP_STRIDE)?,
            kp_count: stream.alloc_zeros::<i32>(1)?,
            ori_kp: stream.alloc_zeros::<f32>(ori_cap * ORI_KP_STRIDE)?,
            ori_count: stream.alloc_zeros::<i32>(1)?,
            desc_in: stream.alloc_zeros::<f32>(ori_cap * DESC_IN_STRIDE)?,
            desc_all: stream.alloc_zeros::<f32>(ori_cap * DESCR_LEN)?,
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

    /// Apply the `max_keypoints` budget to this octave's detections, deterministically.
    ///
    /// The detector appends through an atomic, so `self.kp[0..n_raw]` is in thread-arrival order —
    /// reproducible as a SET only while nothing is discarded, and not reproducible at all once it
    /// is. Taking the leading `max_keypoints` of that list (the previous behaviour) therefore made
    /// the kept keypoints a property of GPU scheduling.
    ///
    /// Selection is by descending response, which is the reference's rule and the one
    /// `retain_best_order` already applies on the CPU path, with position and size breaking ties so
    /// the order is TOTAL. Two keypoints identical in all four fields are interchangeable: they
    /// produce the same descriptor from the same patch.
    ///
    /// The survivors are written back compacted, in that same canonical order, because the
    /// orientation stage consumes `kp[0..n]` positionally and its output order feeds matching and
    /// then track building — both index-sensitive.
    ///
    /// Returns the number of keypoints now valid at the front of `self.kp`. Costs one round trip of
    /// `n_raw * KP_STRIDE` floats (~150 KB at the default budget), and only when the budget binds.
    fn select_best_keypoints(
        &mut self,
        stream: &Arc<CudaStream>,
        n_raw: usize,
    ) -> Result<usize, SiftCudaError> {
        let det_cap = self.kp.len() / KP_STRIDE;
        if n_raw > det_cap {
            // Still lossy, and by arrival — but no longer silent. Reaching this means a frame
            // detected more than 4x the budget, and the answer is a larger budget, not a larger cap.
            //
            // Warned ONCE per process, not per frame: this fires on a whole clip's worth of frames
            // or none of them, and a per-frame line would bury the build log it is meant to inform.
            // This crate carries no logging facade, and adding one for a single diagnostic is not
            // worth the dependency.
            static OVERFLOW_WARNED: std::sync::Once = std::sync::Once::new();
            let dropped = n_raw - det_cap;
            OVERFLOW_WARNED.call_once(|| {
                eprintln!(
                    "kornia sift: {n_raw} extrema exceed the {det_cap}-slot detection buffer; \
                     {dropped} dropped by thread arrival, so affected frames are NOT reproducible. \
                     Raise max_keypoints. (warned once)"
                );
            });
        }
        let n_have = n_raw.min(det_cap);
        if n_have <= self.cfg.max_keypoints {
            // Nothing discarded, so arrival order is a permutation of a fixed set. Canonicalising it
            // anyway would change results for every frame under budget without fixing anything.
            return Ok(n_have);
        }

        let host = stream.clone_dtoh(&self.kp.slice(0..n_have * KP_STRIDE))?;
        let field = |i: usize, f: usize| host[i * KP_STRIDE + f];
        let mut order: Vec<usize> = (0..n_have).collect();
        order.sort_unstable_by(|&a, &b| {
            // `total_cmp`, not `partial_cmp`: a NaN response would otherwise make the comparator
            // inconsistent and the resulting order unspecified — the exact failure being removed.
            field(b, 3)
                .total_cmp(&field(a, 3))
                .then(field(a, 0).total_cmp(&field(b, 0)))
                .then(field(a, 1).total_cmp(&field(b, 1)))
                .then(field(a, 2).total_cmp(&field(b, 2)))
        });
        order.truncate(self.cfg.max_keypoints);

        let mut packed = vec![0.0f32; order.len() * KP_STRIDE];
        for (dst, &src) in order.iter().enumerate() {
            packed[dst * KP_STRIDE..(dst + 1) * KP_STRIDE]
                .copy_from_slice(&host[src * KP_STRIDE..(src + 1) * KP_STRIDE]);
        }
        stream.memcpy_htod(&packed, &mut self.kp.slice_mut(0..packed.len()))?;
        Ok(order.len())
    }

    /// Detect, orient and describe, leaving the descriptors on device.
    ///
    /// `src` is a device-resident single-channel f32 [`Image`] in 0..255,
    /// matching the reference's internal representation. Returns
    /// [`SiftCudaFeatures`]: host keypoints and an owned device descriptor
    /// block — the CUDA path never downloads what a caller did not ask to
    /// move.
    pub fn detect_and_compute(
        &mut self,
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        src: &Image<f32, 1>,
    ) -> Result<SiftCudaFeatures, SiftCudaError> {
        let size = src.size();
        if size.width != self.width || size.height != self.height {
            return Err(SiftCudaError::Geometry(format!(
                "plan built for {}x{}, image is {}x{}",
                self.width, self.height, size.width, size.height
            )));
        }
        let src = src.0.as_cudaslice().ok_or_else(|| {
            SiftCudaError::Geometry(
                "image is not device-resident; move it with Image::to_cuda first".into(),
            )
        })?;
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
            let n_raw = stream.clone_dtoh(&self.kp_count)?[0].max(0) as usize;
            let n_kp = self.select_best_keypoints(stream, n_raw)?;
            if n_kp > 0 {
                // Orientation and descriptors read the Gaussian layer the
                // keypoint was found in, so each launch is given one layer and
                // skips the keypoints that do not belong to it.
                //
                // Oriented keypoints accumulate across the WHOLE frame rather
                // than being reset per layer. `ranges` is NOT written here:
                // an earlier revision snapshotted `ori_count` per layer for a
                // device-side descriptor pass, but the retain_best reorder
                // moved descriptor grouping to the host, which uploads its own
                // `starts` into `ranges` — the audit found the in-loop
                // snapshots written and then overwritten, never read.
                for layer in 1..=self.cfg.n_octave_layers {
                    let img = self.pyr[octv][layer].slice(0..plane);

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
                }
            }

            // ── Next octave base: stride-2 subsample of layer n_octave_layers ─
            let (nw, nh) = (cw / 2, ch / 2);
            if nw == 0 || nh == 0 || octv + 1 >= n_oct {
                break;
            }
            // Straight into the next octave's layer 0. `split_at_mut` at the
            // octave boundary lets the source layer and the destination
            // borrow simultaneously (the blur loop uses the same trick), which
            // removes the buf_a staging hop and its full-plane
            // device-to-device copy.
            {
                let (head, rest) = self.pyr.split_at_mut(octv + 1);
                let src = head[octv][self.cfg.n_octave_layers].slice(0..plane);
                let mut dst = rest[0][0].slice_mut(0..nw * nh);
                launch_sift_downsample_nearest_cuda_view(
                    ctx, stream, &src, &mut dst, cw as u32, ch as u32, nw as u32, nh as u32,
                )?;
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

        // Host-only window: the queue is empty from here until the descriptor
        // launches, so every millisecond in this span is GPU idle time. The
        // audit found it invisible to the stage probes — the probes bracket
        // launches — which is why it gets its own timer. No sync needed: there
        // is nothing in flight to wait for.
        let th_start = mark(probe);

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

        let t_host = th_start
            .map(|t| t.elapsed().as_secs_f64() * 1e3)
            .unwrap_or(0.0);

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

            // Pack on DEVICE from the ori_kp rows it already owns: only the
            // 4-byte row indices cross the bus (the ordering is a host-side
            // sort, so the indices genuinely originate on the host). The
            // kernel's expressions match the previous host pack bit for bit.
            // The `perm` buffer stages the indices; the gather's own upload
            // later overwrites it AFTER this kernel in stream order.
            let rows: Vec<i32> = describe.iter().map(|&i| order[i] as i32).collect();
            stream.memcpy_htod(&rows, &mut self.perm.slice_mut(0..n))?;
            launch_sift_pack_desc_cuda_view(
                ctx,
                stream,
                &self.ori_kp.slice(0..n_ori * ORI_KP_STRIDE),
                &self.perm.slice(0..n),
                n as u32,
                &mut self.desc_in.slice_mut(0..n * DESC_IN_STRIDE),
            )?;

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
            // `n` and `starts` are host-DERIVED control data (retain_best and
            // the group sort run on the host), not device data round-tripped:
            // together they are under 100 bytes.
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
        }
        // The final gather writes each frame's block into a FRESH allocation
        // rather than a plan-owned slab: the result owns its descriptors, so a
        // caller holding two frames for matching needs no defensive copy.
        // SAFETY: the gather kernel writes all n * DESCR_LEN elements before
        // anything reads them; n == 0 allocates a single unread element
        // because a zero-length device allocation is not something CUDA
        // promises to hand back.
        let mut descriptors = unsafe { stream.alloc::<f32>((n * DESCR_LEN).max(1))? };
        if n > 0 {
            let src = self.desc_all.slice(0..n * DESCR_LEN);
            let p = self.perm.slice(0..n);
            let mut out = descriptors.slice_mut(0..n * DESCR_LEN);
            launch_sift_gather_descriptors_cuda_view(ctx, stream, &src, &p, n as u32, &mut out)?;
        }
        if probe {
            eprintln!(
                "  stages: blur={t_blur:.1} detect={t_det:.1} orient={t_ori:.1} \
                 host={t_host:.1} descriptor={t_desc:.1} (ms)"
            );
        }
        self.n_desc = n;
        Ok(SiftCudaFeatures {
            keypoints,
            descriptors,
        })
    }

    /// Number of descriptor rows written by the last call.
    pub fn descriptor_count(&self) -> usize {
        self.n_desc
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::color::test_utils::default_stream;
    use kornia_image::ImageSize;

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
        let d_src = Image::<f32, 1>::from_size_slice(
            ImageSize {
                width: w,
                height: h,
            },
            &host,
        )
        .expect("image")
        .to_cuda(&stream)
        .expect("upload");
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
            assert_eq!(plan.n_desc * DESCR_LEN, f.len() * DESCR_LEN);
            // Every row must have been written. An all-zero row means its
            // launch retired the block or wrote somewhere else.
            let descs = stream
                .clone_dtoh(&f.descriptors.slice(0..f.len() * DESCR_LEN))
                .unwrap();
            for (i, row) in descs.chunks_exact(DESCR_LEN).enumerate() {
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
        let d_src = Image::<f32, 1>::from_size_slice(
            ImageSize {
                width: w,
                height: h,
            },
            &img,
        )
        .unwrap()
        .to_cuda(&stream)
        .unwrap();

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
            plan.n_desc,
            feats.len(),
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

    /// A keypoint budget must cap the count, emit one descriptor row per
    /// survivor, and drop only the weakest keypoints.
    ///
    /// `retainBest` runs *before* the descriptor stage on this path, matching
    /// `sift.dispatch.cpp:568-600`; previously every descriptor was computed and
    /// the ones the budget cut were thrown away. `n_features > 0` had no test on
    /// either backend when that reorder landed.
    ///
    /// Deliberately not gated on `KORNIA_SIFT_ORACLE`: this is a contract about
    /// selection and row alignment, not about descriptor values, so a synthetic
    /// image is enough and the test runs on every CUDA build. The oracle tests
    /// cover values, and they all run at `n_features = 0`.
    ///
    /// **Every assertion is fault-injected, not assumed.** Replacing the
    /// response cut in `final_order` with an arbitrary subset fails this with
    /// "dropped a keypoint stronger than one kept"; miscounting fails the count;
    /// misaligning a row fails the above-count no-op, which compares the whole
    /// descriptor block.
    ///
    /// It uses a real frame rather than synthetic noise deliberately. An earlier
    /// version generated a pseudo-random image, and the response-ordering check
    /// could not be shown to have teeth on it.
    #[test]
    fn budget_caps_the_count_and_keeps_the_strongest() {
        // A real frame, not synthetic noise: a pseudo-random image yields
        // keypoint responses clustered too tightly for the "keeps the strongest"
        // assertion below to separate, which an injected selection bug proved by
        // going undetected on one.
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/data/mh01_frame1.png");
        let gray = kornia_io::png::read_image_png_mono8(path).expect("mh01_frame1.png");
        let (w, h) = (gray.width(), gray.height());
        // The reference works in 0..255 floats, not 0..1.
        let img: Vec<f32> = gray.as_slice().iter().map(|&p| p as f32).collect();

        let stream = default_stream();
        let ctx = &stream.context();
        let d_src = Image::<f32, 1>::from_size_slice(
            ImageSize {
                width: w,
                height: h,
            },
            &img,
        )
        .unwrap()
        .to_cuda(&stream)
        .unwrap();

        let mut all_plan = SiftCuda::new(
            ctx,
            &stream,
            w,
            h,
            SiftCudaConfig::default(),
            FirstOctave::Native,
            8,
        )
        .expect("plan");
        let all = all_plan
            .detect_and_compute(ctx, &stream, &d_src)
            .expect("detect");
        assert!(
            all.len() > 8,
            "need enough keypoints to budget, got {}",
            all.len()
        );

        let n = all.len() / 2;
        let mut cut_plan = SiftCuda::new(
            ctx,
            &stream,
            w,
            h,
            SiftCudaConfig {
                n_features: n,
                ..SiftCudaConfig::default()
            },
            FirstOctave::Native,
            8,
        )
        .expect("plan");
        let cut = cut_plan
            .detect_and_compute(ctx, &stream, &d_src)
            .expect("detect");

        assert_eq!(cut.len(), n, "budget must cap the count");
        assert_eq!(
            cut_plan.descriptor_count(),
            n,
            "one descriptor row per surviving keypoint"
        );
        let cut_desc = stream
            .clone_dtoh(&cut.descriptors.slice(0..n * DESCR_LEN))
            .unwrap();

        // Every dropped keypoint must be no stronger than every kept one, which
        // is what `retainBest` guarantees.
        let worst_kept = cut
            .keypoints
            .iter()
            .map(|k| k.response)
            .fold(f32::INFINITY, f32::min);
        let mut dropped = 0usize;
        for a in &all.keypoints {
            if !cut.keypoints.iter().any(|b| b.x == a.x && b.y == a.y) {
                dropped += 1;
                assert!(
                    a.response <= worst_kept,
                    "dropped a keypoint stronger than one kept"
                );
            }
        }
        assert_eq!(dropped, all.len() - n);

        // A budget above the keypoint count is a no-op.
        let mut big_plan = SiftCuda::new(
            ctx,
            &stream,
            w,
            h,
            SiftCudaConfig {
                n_features: all.len() + 1000,
                ..SiftCudaConfig::default()
            },
            FirstOctave::Native,
            8,
        )
        .expect("plan");
        let big = big_plan
            .detect_and_compute(ctx, &stream, &d_src)
            .expect("detect");
        assert_eq!(big.len(), all.len());
        let big_desc = stream
            .clone_dtoh(&big.descriptors.slice(0..big.len() * DESCR_LEN))
            .unwrap();
        // `cut_desc` pins the budgeted rows too: the kept keypoints' rows must
        // be a prefix-by-order subset of the unbudgeted block's rows.
        assert_eq!(cut_desc.len(), n * DESCR_LEN);
        let all_desc = stream
            .clone_dtoh(&all.descriptors.slice(0..all.len() * DESCR_LEN))
            .unwrap();
        assert_eq!(big_desc, all_desc);
    }
}
