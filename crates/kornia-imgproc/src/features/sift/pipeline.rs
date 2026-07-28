//! End-to-end CPU SIFT: base image, octave loop, and the reference's final
//! ordering.
//!
//! # Measured position
//!
//! On mh01 (752x480), median of 5, against `cv::SIFT` on one thread:
//!
//! ```text
//! OpenCV 5.0.0   225.4 ms      OpenCV 4.13.0   223.1 ms     (both 2515 kp)
//!
//! ours, fo=-1            ~138 ms exact    ~108 ms fast      1.6x / 2.1x
//! ours, fo=0, 4 octaves   ~40 ms exact     ~32 ms fast      5.7x / 7.1x
//! ```
//!
//! The two OpenCV versions are within 1% of each other — 5.0 dropped the Tegra
//! HAL that accelerates `exp`/`atan2`/`magnitude` on aarch64, and it makes no
//! measurable difference to SIFT.
//!
//! There is deliberately **no whole-layer gradient precompute** on this path.
//! Deriving magnitude and angle inside the orientation and descriptor loops is
//! what the reference does, and it is what the CUDA twin does too
//! (`cuda/sift/orientation.rs` and `cuda/sift/descriptor.rs` both take the
//! `dx`/`dy` stencil per sample) — so the two backends agree here.
//!
//! The precompute this replaced evaluated `magnitude` and `atan2` for every
//! pixel of every searched layer — 5.63M of them on a 752x480 frame — when the
//! keypoint patches only ever read about 3.1M. Measured at 30.9 ms saved
//! against 20.9 ms added, and it drops ~46 MB of workspace.
//!
//! The ceiling is set by one comparison: OpenCV's own `GaussianBlur` runs the
//! scale space's five layers in 42.7 ms, ours in 45.0 single-threaded. We are at
//! 0.95x of OpenCV *per core*, so the speedup has to come from threading, and
//! six cores bound that near 6x — which is where `fo=0` lands. Every lever has
//! been implemented and measured: threading, four-accumulator latency hiding,
//! a 4-lane NEON HAL, and the symmetric row filter. A rotated-frame descriptor
//! that samples a fixed 576 points regardless of scale was implemented too and
//! **regressed on CPU** despite sampling a third as many points — the rotated
//! walk loses the sequential row access the reference's stencil gets. It is
//! worth 2.9x on the GPU, where the access pattern does not matter the same
//! way, and lives there only.

use rayon::prelude::*;

use super::descriptor::{compute_descriptor, descriptor_inputs, DescriptorScratch, DESCR_LEN};
use super::detect::{find_extrema, RawKeypoint};
use super::orient::{assign_orientations, OrientScratch, OrientedKeypoint};
use super::params::{
    gaussian_kernel_f32, gaussian_ksize, validate_source, SiftConfig, SiftConfigError,
};
use super::scalespace::{blur_h_f32_mode, blur_v_f32};

// These three types are the backend-independent vocabulary of the pipeline, so
// like `params` they have one definition and `cuda::sift` re-exports it. They
// were declared twice — identical fields, identical derives — which meant the
// two backends' results were nominally different types and the Python binding
// had to import one of them under an alias.

/// Which scale the pyramid starts from.
///
/// `Double` matches OpenCV, COLMAP and VLFeat, which all upsample 2x before
/// building the pyramid; it roughly doubles the keypoint count and costs about
/// 3.6x the time. `Native` starts at the input resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FirstOctave {
    /// `first_octave = -1`: upsample 2x first (the reference's default).
    Double,
    /// `first_octave = 0`: start at the input resolution.
    Native,
}

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

/// Separable bilinear 2x upsample forming the base of octave 0.
///
/// This is the reference's ordinary resize, **not** its precise-upscale affine
/// warp — the option defaults to off, and the two differ by a quarter pixel,
/// which shifts every downstream keypoint.
fn upsample2x(src: &[f32], sw: usize, sh: usize, dst: &mut [f32], hbuf: &mut [f32]) {
    let (dw, dh) = (sw * 2, sh * 2);
    // For a 2x upscale the source tap reduces to a parity test with weights
    // {0.75, 0.25}: `d/2 - 0.25` is exact in f32 for any integer `d`.
    let tap = |d: usize, n: usize| -> (usize, f32) {
        let odd = d & 1;
        let mut s = (d >> 1) as isize - ((odd ^ 1) as isize);
        let mut f = if odd == 1 { 0.25f32 } else { 0.75f32 };
        if s < 0 {
            f = 0.0;
            s = 0;
        }
        if s >= n as isize - 1 {
            f = 0.0;
            s = n as isize - 1;
        }
        (s as usize, f)
    };

    // Separate the two axes instead of evaluating a full bilinear per output
    // pixel. The row lerp of a given source row is what the fused form calls
    // `r0` for one output row and `r1` for the next, so it was computed twice
    // for every source row; here it is computed once. Bit-exact by
    // construction: the horizontal expression and its operands are unchanged,
    // and the vertical lerp still combines exactly those two values.
    hbuf[..sh * dw]
        .par_chunks_mut(dw)
        .enumerate()
        .for_each(|(sy, hrow)| {
            let base = sy * sw;
            // `x` advances (and `row` is read) only inside the NEON block, so
            // the x86 fallback build sees them unused — allow, don't gate the
            // declaration, since the tail loops below consume `x` on both.
            #[cfg_attr(not(target_arch = "aarch64"), allow(unused_mut))]
            let mut x = 0usize;
            #[cfg(target_arch = "aarch64")]
            let row = &src[base..base + sw];

            // Vector body: away from the two clamped outputs (x = 0 and
            // x = dw-1) the taps are a fixed two-phase pattern —
            //   even x = 2m: src[m-1]*0.25 + src[m]*0.75
            //   odd  x = 2m+1: src[m]*0.75 + src[m+1]*0.25
            // so four source pixels yield eight outputs: two mul+mul+add
            // vectors (the scalar's exact expression shape — no FMA, Rust
            // does not contract) interleaved with vzip. The scalar loop
            // below remains the head (x < 2), the tail, and the whole row
            // for the non-NEON build.
            #[cfg(target_arch = "aarch64")]
            unsafe {
                use std::arch::aarch64::*;
                if sw >= 6 {
                    let mut m = 1usize;
                    x = 2;
                    while m + 5 <= sw {
                        let am1 = vld1q_f32(row.as_ptr().add(m - 1));
                        let a = vld1q_f32(row.as_ptr().add(m));
                        let ap1 = vld1q_f32(row.as_ptr().add(m + 1));
                        // even: src[m-1]*(1-0.75) + src[m]*0.75
                        let e = vaddq_f32(vmulq_n_f32(am1, 0.25), vmulq_n_f32(a, 0.75));
                        // odd: src[m]*(1-0.25) + src[m+1]*0.25
                        let o = vaddq_f32(vmulq_n_f32(a, 0.75), vmulq_n_f32(ap1, 0.25));
                        vst1q_f32(hrow.as_mut_ptr().add(x), vzip1q_f32(e, o));
                        vst1q_f32(hrow.as_mut_ptr().add(x + 4), vzip2q_f32(e, o));
                        m += 4;
                        x += 8;
                    }
                }
            }

            for (xx, o) in hrow.iter_mut().enumerate().skip(x) {
                let (sx, fx) = tap(xx, sw);
                let sx1 = (sx + 1).min(sw - 1);
                *o = src[base + sx] * (1.0 - fx) + src[base + sx1] * fx;
            }
            // The head (x = 0..2) is not covered by the vector loop.
            for (xx, o) in hrow.iter_mut().enumerate().take(x.min(2)) {
                let (sx, fx) = tap(xx, sw);
                let sx1 = (sx + 1).min(sw - 1);
                *o = src[base + sx] * (1.0 - fx) + src[base + sx1] * fx;
            }
        });

    dst[..dw * dh]
        .par_chunks_mut(dw)
        .enumerate()
        .for_each(|(y, row)| {
            let (sy, fy) = tap(y, sh);
            let sy1 = (sy + 1).min(sh - 1);
            let (r0, r1) = (&hbuf[sy * dw..sy * dw + dw], &hbuf[sy1 * dw..sy1 * dw + dw]);
            for ((o, a), b) in row.iter_mut().zip(r0.iter()).zip(r1.iter()) {
                *o = a * (1.0 - fy) + b * fy;
            }
        });
}

/// Nearest stride-2 subsample — **not** a blur-and-decimate, which would break
/// parity with the reference.
fn downsample(src: &[f32], sw: usize, sh: usize, dst: &mut [f32], dw: usize, dh: usize) {
    // The only caller passes `dw = sw / 2`, `dh = sh / 2`, and under that the
    // general rescale index reduces to the even pixel: `sw ∈ {2dw, 2dw+1}`, so
    // `(x * sw) / dw = 2x + x/dw = 2x` for every `x < dw` (identically for y).
    // The general form cost a non-pipelined 64-bit `udiv` per output pixel —
    // ~15 cycles in a loop that is otherwise one load and one store — and this
    // function sits between two stage probes, so none of it ever showed up in
    // a breakdown. Keep the general path for any future caller that halves
    // differently.
    let halving = dw > 0 && dh > 0 && sw / 2 == dw && sh / 2 == dh;
    dst[..dw * dh]
        .par_chunks_mut(dw)
        .enumerate()
        .for_each(|(y, row)| {
            if halving {
                let r = &src[(2 * y) * sw..(2 * y) * sw + sw];
                for (x, o) in row.iter_mut().enumerate() {
                    *o = r[2 * x];
                }
            } else {
                let sy = (y * sh) / dh;
                for (x, o) in row.iter_mut().enumerate() {
                    *o = src[sy * sw + (x * sw) / dw];
                }
            }
        });
}

/// Reusable scratch for [`detect_and_compute_with`].
///
/// The pipeline touches a dozen full-resolution planes — six Gaussian layers,
/// five DoG layers, plus the base and a transpose buffer. Allocating and
/// zeroing those per call is tens of megabytes of pure overhead at `fo=-1`,
/// which is why a streaming caller should hold one of these and reuse it. This
/// is the same reason the CUDA path owns a plan object.
#[derive(Default)]
pub struct SiftWorkspace {
    plane: usize,
    base: Vec<f32>,
    tmp: Vec<f32>,
    /// Gaussian layers for every octave: `pyr[octave][layer]`.
    ///
    /// Resident for the whole frame so descriptors can run after the keypoint
    /// budget has been applied — the reference applies `retainBest` before
    /// `calcDescriptors`. Octave sizes are a geometric series, so holding all of
    /// them is ~1.33x the single-octave slab this used to keep.
    pyr: Vec<Vec<Vec<f32>>>,
    dog: Vec<f32>,
}

impl SiftWorkspace {
    /// Empty scratch; sized on first use.
    pub fn new() -> Self {
        Self::default()
    }

    fn ensure(&mut self, plane: usize, n_layers: usize, n_dog: usize, n_oct: usize) {
        if self.plane >= plane && self.pyr.first().is_some_and(|o| o.len() >= n_layers) {
            return;
        }
        self.plane = plane;
        self.base = vec![0.0; plane];
        self.tmp = vec![0.0; plane];
        self.pyr = (0..n_oct)
            .map(|o| {
                let op = (plane >> (2 * o)).max(16);
                (0..n_layers).map(|_| vec![0.0; op]).collect()
            })
            .collect();
        self.dog = vec![0.0; plane * n_dog];
    }
}

/// Detect, orient and describe.
///
/// Allocates its own scratch. For more than one frame use
/// [`detect_and_compute_with`], which takes a reusable [`SiftWorkspace`].
///
/// # Arguments
///
/// * `src` - Row-major `w * h` f32 grayscale image in **0..255** — the
///   reference's own internal representation. Normalising to 0..1 changes what
///   `cfg.contrast_threshold` means and silently returns far fewer keypoints.
/// * `w`, `h` - Image dimensions; `src.len()` must equal `w * h`.
/// * `cfg` - Detector parameters, mirroring `cv::SIFT::create`.
/// * `first_octave` - [`FirstOctave::Double`] upsamples 2x first, as OpenCV does
///   by default; [`FirstOctave::Native`] starts at the input resolution.
/// * `max_octaves` - Ceiling on the pyramid depth; `usize::MAX` for unlimited.
/// * `fast` - Trade the reference's exact tap order for speed in the row blur.
///   Keypoints and descriptors are then no longer bit-exact against `cv2`.
///
/// # Returns
///
/// [`SiftFeatures`]: the keypoints in the reference's final order, and a
/// row-major `keypoints.len() * 128` descriptor block.
///
/// # Errors
///
/// Returns [`SiftConfigError::SourceLen`] if `src.len() != w * h`, and
/// [`SiftConfigError::Invalid`] if the dimensions overflow or a parameter in
/// `cfg` is out of range (for example `n_octave_layers == 0`).
///
/// # Example
///
/// ```rust
/// use kornia_imgproc::features::{sift_detect_and_compute, FirstOctave, SiftConfig};
///
/// let (w, h) = (64, 64);
/// // A single bright square: enough structure to exercise the pipeline.
/// let mut img = vec![0.0f32; w * h];
/// for y in 24..40 {
///     for x in 24..40 {
///         img[y * w + x] = 255.0;
///     }
/// }
///
/// let cfg = SiftConfig::default();
/// let feats = sift_detect_and_compute(
///     &img, w, h, &cfg, FirstOctave::Native, usize::MAX, false,
/// )?;
/// assert_eq!(feats.descriptors.len(), feats.keypoints.len() * 128);
/// # Ok::<(), kornia_imgproc::features::SiftConfigError>(())
/// ```
pub fn detect_and_compute(
    src: &[f32],
    w: usize,
    h: usize,
    cfg: &SiftConfig,
    first_octave: FirstOctave,
    max_octaves: usize,
    fast: bool,
) -> Result<SiftFeatures, SiftConfigError> {
    let mut ws = SiftWorkspace::new();
    detect_and_compute_with(&mut ws, src, w, h, cfg, first_octave, max_octaves, fast)
}

/// Detect, orient and describe against caller-owned scratch.
///
/// Prefer this when processing more than one frame: it is the same work without
/// the per-call allocation of ~20 full-resolution planes.
///
/// # Arguments
///
/// * `ws` - Reusable scratch, sized on first use and regrown only when a later
///   frame is larger. Keep one per thread and pass it every call.
/// * All remaining arguments are as in [`detect_and_compute`].
///
/// # Returns
///
/// [`SiftFeatures`], identical to what [`detect_and_compute`] returns for the
/// same inputs — `ws` affects allocation, never the result.
///
/// # Errors
///
/// The same [`SiftConfigError`] cases as [`detect_and_compute`].
///
/// # Example
///
/// ```rust
/// use kornia_imgproc::features::{sift_detect_with, FirstOctave, SiftConfig, SiftWorkspace};
///
/// let (w, h) = (64, 64);
/// let img = vec![0.0f32; w * h];
/// let cfg = SiftConfig::default();
/// let mut ws = SiftWorkspace::new();
/// for _frame in 0..2 {
///     let feats = sift_detect_with(
///         &mut ws, &img, w, h, &cfg, FirstOctave::Native, usize::MAX, false,
///     )?;
///     assert!(feats.is_empty()); // a flat image has no extrema
/// }
/// # Ok::<(), kornia_imgproc::features::SiftConfigError>(())
/// ```
#[allow(clippy::too_many_arguments)]
pub fn detect_and_compute_with(
    ws: &mut SiftWorkspace,
    src: &[f32],
    w: usize,
    h: usize,
    cfg: &SiftConfig,
    first_octave: FirstOctave,
    max_octaves: usize,
    fast: bool,
) -> Result<SiftFeatures, SiftConfigError> {
    // Validated through the same selector the CUDA path uses, so which inputs
    // are rejected does not depend on where the image lives.
    let t_start = std::time::Instant::now();
    validate_source(src.len(), w, h)?;
    cfg.validate(max_octaves)?;
    let doubled = first_octave == FirstOctave::Double;
    let (mut cw, mut ch) = if doubled { (w * 2, h * 2) } else { (w, h) };

    let n_layers = cfg.n_octave_layers + 3;
    let n_dog = cfg.n_octave_layers + 2;
    let sigmas = cfg.layer_sigmas();
    let base_sigma = cfg.base_sig_diff(doubled) as f64;
    let base_kernel = gaussian_kernel_f32(gaussian_ksize(base_sigma), base_sigma);
    let layer_kernels: Vec<Vec<f32>> = (1..n_layers)
        .map(|i| gaussian_kernel_f32(gaussian_ksize(sigmas[i]), sigmas[i]))
        .collect();

    let plane = cw * ch;
    // Needed before `ensure`: the workspace now sizes one Gaussian slab per
    // octave rather than one shared slab.
    let n_oct = cfg
        .n_octaves(cw.min(ch), if doubled { -1 } else { 0 })
        .min(max_octaves);
    ws.ensure(plane, n_layers, n_dog, n_oct);
    let SiftWorkspace {
        base,
        tmp,
        pyr,
        dog,
        ..
    } = ws;
    if doubled {
        upsample2x(src, w, h, base, tmp);
    } else {
        // Slice both sides: `validate_source` only requires `src` to be *at
        // least* `w * h` long, so an over-long input would trip
        // `copy_from_slice`'s equal-length requirement and panic. The CUDA twin
        // slices for exactly this reason.
        base[..plane].copy_from_slice(&src[..plane]);
    }
    blur_h_f32_mode(
        &base[..plane],
        &mut tmp[..plane],
        cw,
        ch,
        &base_kernel,
        fast,
    );
    blur_v_f32(
        &tmp[..plane],
        &mut pyr[0][0][..plane],
        cw,
        ch,
        &base_kernel,
        None,
    );

    // KORNIA_SIFT_STAGES=1 breaks the pass down. Each probe is a plain elapsed
    // measurement -- no synchronisation needed on CPU -- so the total is not
    // inflated the way the CUDA equivalent is.
    // Read once: this is a per-frame path and `env::var` locks and allocates.
    // Same treatment as the CUDA path in `cuda::sift::plan`.
    static PROBE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let probe = *PROBE.get_or_init(|| std::env::var("KORNIA_SIFT_STAGES").is_ok());
    let (mut t_blur, mut t_det, mut t_ori, mut t_desc) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mark = || std::time::Instant::now();

    // The base image build is a stage in its own right: it upsamples and blurs
    // a full plane, and it sat outside every probe until it was measured at
    // 12 ms — the same way the gradient pass hid between two timers.
    let t_base = t_start.elapsed().as_secs_f64() * 1e3;

    let mut all: Vec<(OrientedKeypoint, usize, usize)> = Vec::new();
    // The deferred descriptor pass needs each octave's dimensions; the loop
    // halves with integer division, which is not `>>` for odd sizes.
    let mut oct_dims: Vec<(usize, usize)> = Vec::with_capacity(n_oct);

    for octv in 0..n_oct {
        if cw < 16 || ch < 16 {
            break;
        }
        let p = cw * ch;
        oct_dims.push((cw, ch));
        let tb = mark();
        for i in 1..n_layers {
            let k = &layer_kernels[i - 1];
            // Slice to this octave's plane: the workspace buffers stay sized for
            // the FIRST octave, so from octave 1 on the full `gauss[i - 1]` is
            // longer than `cw * ch` and trips `blur_h_f32_mode`'s length assert.
            blur_h_f32_mode(&pyr[octv][i - 1][..p], &mut tmp[..p], cw, ch, k, fast);
            let (lo, hi) = pyr[octv].split_at_mut(i);
            blur_v_f32(
                &tmp[..p],
                &mut hi[0][..p],
                cw,
                ch,
                k,
                Some((&lo[i - 1][..p], &mut dog[(i - 1) * p..i * p])),
            );
        }

        if probe {
            t_blur += tb.elapsed().as_secs_f64() * 1e3;
        }

        let td = mark();
        let mut kps: Vec<RawKeypoint> = Vec::new();
        for layer in 1..=cfg.n_octave_layers {
            find_extrema(&dog[..p * n_dog], cw, ch, n_dog, layer, octv, cfg, &mut kps);
        }
        if probe {
            t_det += td.elapsed().as_secs_f64() * 1e3;
        }
        let to = mark();
        // Indexing by layer is the point here: `gauss[layer]` is the Gaussian
        // the keypoints of that layer were found in.
        #[allow(clippy::needless_range_loop)]
        for layer in 1..=cfg.n_octave_layers {
            let gl = &pyr[octv][layer][..p];
            // One task per keypoint group: each histogram is independent, and
            // the reference's order within a keypoint is preserved because a
            // single task owns it.
            let group: Vec<&RawKeypoint> = kps.iter().filter(|k| k.layer == layer as i32).collect();
            let oriented: Vec<OrientedKeypoint> = group
                .par_iter()
                // One scratch per worker, not per keypoint.
                .map_init(OrientScratch::new, |sc, kp| {
                    let mut o = Vec::new();
                    assign_orientations(gl, cw, ch, kp, cfg, &mut o, sc);
                    o
                })
                .flatten_iter()
                .collect();
            for k in oriented {
                all.push((k, octv, layer));
            }
        }
        if probe {
            t_ori += to.elapsed().as_secs_f64() * 1e3;
        }
        let (nw, nh) = (cw / 2, ch / 2);
        if nw == 0 || nh == 0 || octv + 1 >= n_oct {
            break;
        }
        let (cur, next) = pyr.split_at_mut(octv + 1);
        downsample(
            &cur[octv][cfg.n_octave_layers][..p],
            cw,
            ch,
            &mut next[0][0],
            nw,
            nh,
        );
        cw = nw;
        ch = nh;
    }

    // first_octave = -1 post-processing, then the reference's ordering.
    let scale = if doubled { 0.5f32 } else { 1.0f32 };
    let mut kps: Vec<SiftKeypoint> = all
        .iter()
        .map(|(k, _, _)| {
            let packed = k.octave;
            SiftKeypoint {
                x: k.x * scale,
                y: k.y * scale,
                size: k.size * scale,
                angle: k.angle,
                response: k.response,
                octave: if doubled {
                    (packed & !255) | ((packed - 1) & 255)
                } else {
                    packed
                },
            }
        })
        .collect();

    let order = final_order(&kps, cfg.n_features);
    let n = order.len();

    // ── Descriptors, for the survivors only ─────────────────────────────────
    //
    // The reference applies `retainBest` before `calcDescriptors`; describing
    // inside the octave loop meant computing every descriptor and discarding
    // the ones the budget cut. That is why the pyramid stays resident. Writing
    // straight into `order` also removes the permutation the old code needed.
    let tds = mark();
    let mut out_desc = vec![0.0f32; n * DESCR_LEN];
    out_desc
        .par_chunks_mut(DESCR_LEN)
        .zip(order.par_iter())
        // One scratch per worker, not per keypoint: a patch is a few thousand
        // floats and allocating it per descriptor would dominate.
        .for_each_init(DescriptorScratch::new, |sc, (out, &i)| {
            let (k, octv, layer) = all[i];
            let (gw, gh) = oct_dims[octv];
            let gl = &pyr[octv][layer][..gw * gh];
            let (x, y, s, a) = descriptor_inputs(&k, octv as i32);
            // `fast` deliberately does NOT select a rotated-frame descriptor
            // here: that variant is measured slower on CPU despite sampling a
            // third as many points, so it exists only on the CUDA path. See
            // this module's docs.
            compute_descriptor(gl, gw, gh, x, y, s, a, out, sc);
        });
    if probe {
        t_desc += tds.elapsed().as_secs_f64() * 1e3;
    }
    kps = order.iter().map(|&i| kps[i]).collect();
    if probe {
        eprintln!(
            "    stages: base={t_base:.1} blur={t_blur:.1} detect={t_det:.1} \
orient={t_ori:.1} desc={t_desc:.1} (ms)"
        );
    }
    Ok(SiftFeatures {
        keypoints: kps,
        descriptors: out_desc,
    })
}

/// The reference's final keypoint order: `removeDuplicatedSorted`, then
/// `retainBest`. Returns indices into the input order.
///
/// Shared with the CUDA pipeline: both backends must return the same rows in the
/// same order, and this ordering is subtle enough in two places (the comparator's
/// descending fields, the tie behaviour at the budget cut-off) that keeping two
/// copies in step was never going to hold.
pub(crate) fn final_order(kps: &[SiftKeypoint], n_features: usize) -> Vec<usize> {
    retain_best_order(kps, sorted_dedup_order(kps), n_features)
}

/// Order keypoints the way `KeyPointsFilter::removeDuplicatedSorted` does, and
/// drop the exact duplicates it drops.
///
/// Two reasons this is not optional. The CUDA detector appends through an atomic
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Every octave past the first runs on a plane smaller than the workspace
    /// buffers, so the scale-space calls must slice down to it. Passing a whole
    /// workspace buffer tripped `blur_h_f32_mode`'s length assert on octave 1,
    /// and nothing covered it: the only other test here is opt-in via env vars.
    #[test]
    fn runs_every_octave_on_reused_workspace() {
        let (w, h) = (128usize, 128usize);
        let img: Vec<f32> = (0..w * h).map(|i| ((i * 37) % 251) as f32).collect();
        let cfg = SiftConfig::default();
        let mut ws = SiftWorkspace::new();
        for fo in [FirstOctave::Native, FirstOctave::Double] {
            // Two passes: the second reuses scratch sized by the first.
            for _ in 0..2 {
                let f = detect_and_compute_with(&mut ws, &img, w, h, &cfg, fo, usize::MAX, false)
                    .expect("valid configuration");
                assert_eq!(f.descriptors.len(), f.keypoints.len() * DESCR_LEN);
            }
        }
    }

    /// The rejections are residency-independent: both backends route through
    /// `SiftConfig::validate` and `validate_source`, so this pins what the CPU
    /// side refuses. The CUDA side asserts the same set against the same
    /// selector in `cuda::sift::tests`.
    #[test]
    fn rejects_the_same_configurations_as_cuda() {
        let img = vec![0.0f32; 64 * 64];
        let cfg = SiftConfig::default();
        let ok = |c: &SiftConfig, m: usize, len: usize, w: usize, h: usize| {
            detect_and_compute_with(
                &mut SiftWorkspace::new(),
                &img[..len],
                w,
                h,
                c,
                FirstOctave::Native,
                m,
                false,
            )
            .is_ok()
        };
        assert!(ok(&cfg, usize::MAX, 64 * 64, 64, 64));
        // max_octaves = 0 used to clamp silently here and error on the GPU.
        assert!(!ok(&cfg, 0, 64 * 64, 64, 64));
        // A short source used to panic here and error on the GPU.
        assert!(!ok(&cfg, usize::MAX, 64 * 63, 64, 64));
        assert!(!ok(&cfg, usize::MAX, 64 * 64, 0, 64));
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let c = SiftConfig { sigma: bad, ..cfg };
            assert!(!ok(&c, usize::MAX, 64 * 64, 64, 64), "sigma {bad} accepted");
        }
        let c = SiftConfig {
            n_octave_layers: 0,
            ..cfg
        };
        assert!(!ok(&c, usize::MAX, 64 * 64, 64, 64));
    }

    /// A keypoint budget selects the highest-response keypoints, and a survivor
    /// keeps the descriptor row it would have had unbudgeted.
    ///
    /// `n_features > 0` had no test on either backend when `retainBest` was
    /// moved ahead of the descriptor stage, which is what this covers.
    ///
    /// **Scope, established by injecting faults rather than assumed.** The
    /// descriptor comparison is against this same pipeline with no budget, so it
    /// cannot catch anything that corrupts both runs identically — a wrong
    /// Gaussian layer and a uniform index shift were both injected and both went
    /// undetected, because the unbudgeted run computes them equally wrongly.
    /// What it does catch is budget-*specific* breakage: a survivor paired with
    /// another's row, a permutation that only misbehaves on a sparse set, or a
    /// grouping that depends on which keypoints survive — the code paths the
    /// reorder actually added. Descriptor *values* are the bitwise oracle's job
    /// (`descriptor_matches_reference_bitwise`), which runs at `n_features = 0`.
    #[test]
    fn budget_keeps_the_best_and_their_own_descriptors() {
        let (w, h) = (192usize, 144usize);
        let mut seed = 0x9E3779B9u32;
        let img: Vec<f32> = (0..w * h)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                ((seed >> 16) & 255) as f32
            })
            .collect();
        let cfg = SiftConfig::default();
        let all = detect_and_compute(&img, w, h, &cfg, FirstOctave::Native, 3, false).unwrap();
        assert!(all.keypoints.len() > 8, "need enough keypoints to budget");

        let n = all.keypoints.len() / 2;
        let budget = SiftConfig {
            n_features: n,
            ..cfg
        };
        let cut = detect_and_compute(&img, w, h, &budget, FirstOctave::Native, 3, false).unwrap();

        assert_eq!(cut.keypoints.len(), n, "budget must cap the count");
        assert_eq!(cut.descriptors.len(), n * DESCR_LEN, "one row per keypoint");

        // Every survivor must be at least as strong as every keypoint dropped,
        // which is what `retainBest` guarantees.
        let kept: Vec<f32> = cut.keypoints.iter().map(|k| k.response).collect();
        let worst_kept = kept.iter().cloned().fold(f32::INFINITY, f32::min);
        let dropped = all
            .keypoints
            .iter()
            .filter(|a| !cut.keypoints.iter().any(|b| b.x == a.x && b.y == a.y))
            .count();
        assert_eq!(dropped, all.keypoints.len() - n);
        for a in &all.keypoints {
            if !cut.keypoints.iter().any(|b| b.x == a.x && b.y == a.y) {
                assert!(
                    a.response <= worst_kept,
                    "dropped a keypoint stronger than one kept"
                );
            }
        }

        // ...and each survivor carries its own descriptor, bit for bit.
        for (i, k) in cut.keypoints.iter().enumerate() {
            let j = all
                .keypoints
                .iter()
                .position(|a| a.x == k.x && a.y == k.y && a.angle == k.angle)
                .expect("survivor missing from the unbudgeted run");
            let got = &cut.descriptors[i * DESCR_LEN..(i + 1) * DESCR_LEN];
            let want = &all.descriptors[j * DESCR_LEN..(j + 1) * DESCR_LEN];
            assert!(
                got.iter()
                    .zip(want)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "keypoint {i} got a different descriptor under the budget"
            );
        }
    }
}
