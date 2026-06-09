//! The XFeat model graph — `XFeat::extract(image) -> XFeatOutput`.
//!
//! All allocations happen once at construction time. The hot path (per frame)
//! is zero-alloc except for the sparse output, which is bounded by `top_k`.
//!
//! ## Architecture (upstream `modules/model.py`, CVPR 2024)
//!
//! ```text
//! gray (H,W,1)
//!   └─ InstanceNorm2d → norm_gray (H,W)
//!       ├─ Block 1: 3×3 conv(1→4) → (4→8,s2) → (8→8) → (8→24,s2)   [BN-fold]
//!       ├─ Skip1:   AvgPool(4×4) → conv1x1(1→24, real bias)
//!       └─ x1 = block1_out + skip1_out                                (H/4,W/4,24)
//!           └─ Block 2: 2× 3×3 conv(24→24)                           (H/4,W/4,24)
//!               └─ Block 3: 3×3 (24→64,s2) → (64→64) → 1×1 (64→64)  (H/8,W/8,64)
//!                   └─ Block 4: 3×3 (64→64,s2) → (64→64) → (64→64)  (H/16,W/16,64)
//!                       └─ Block 5: (64→128,s2) → (128→128)×2
//!                                   → 1×1 (128→64)                    (H/32,W/32,64)
//!   feats = block_fusion(x3 + up(x4) + up(x5))                        (H/8,W/8,64)
//!   ├─ heatmap_head: 1×1 (64→64)×2 → conv1x1(64→1)+sigmoid           reliability h1 (H/8,W/8,1)
//!   └─ keypoint_head [input = unfold_8x8(norm_gray)]:
//!          1×1 (64→64)×3 → conv1x1(64→65)                            logits K1 (H/8,W/8,65)
//!          → softmax/65 → drop dustbin → pixel_shuffle(8)             heatmap K1h (H,W)
//! ```
//!
//! ## Key: `BasicLayer` = `Conv2d(bias=False) + BN(affine=False) + ReLU`.
//! BN fold (γ=1, β=0): `W_eff = W/sqrt(var+ε)`, `b_eff = -mean/sqrt(var+ε)`.
//!
//! Layers block3.2, block5.3, and all head layers are 1×1 convs (not 3×3).

use rayon::prelude::*;
use crate::ops::{
    add3_inplace, add_inplace, avgpool_4x4_s4, bilinear_upsample, channel_softmax,
    drop_last_channel_nhwc, instance_norm_2d_singlech, l2_normalize_channel, pixel_shuffle_8,
    repack_weights_co4_3x3, repack_weights_co8_3x3_f16, unfold_8x8, winograd,
    Activation, Conv1x1Args, Conv3x3Args, OpsVtable,
};
#[cfg(target_arch = "aarch64")]
use crate::ops::neon;
use crate::postproc::{bicubic_sample_descriptors, nms_topk, KeyPoint};
use crate::preproc::InputScale;
use crate::weights::{PackedWeights, WinogradCache};
use crate::XFeatError;

// ─── Winograd-eligible stride-1 3×3 layers ────────────────────────────────────
// (layer_name, c_out, c_in) in model execution order.
// Only stride-1 3×3 layers qualify. Stride-2 layers (block1.1, block1.3,
// block3.0, block4.0, block5.0) remain on the NEON/AVX2 vtable path.
const WINOGRAD_LAYERS: &[(&str, usize, usize)] = &[
    ("block1.2",         8,   8),
    ("block2.0",        24,  24),
    ("block2.1",        24,  24),
    ("block3.1",        64,  64),
    ("block4.1",        64,  64),
    ("block4.2",        64,  64),
    ("block5.1",       128, 128),
    ("block5.2",       128, 128),
    ("block_fusion.0",  64,  64),
    ("block_fusion.1",  64,  64),
];

/// Static configuration of an XFeat instance.
#[derive(Debug, Clone)]
pub struct XFeatConfig {
    /// Height (pixels), must be a multiple of 32.
    pub height: usize,
    /// Width (pixels), must be a multiple of 32.
    pub width: usize,
    /// NMS score threshold. Upstream default: 0.05.
    pub score_threshold: f32,
    /// Max keypoints to return. Upstream default: 4096.
    pub top_k: usize,
}

impl Default for XFeatConfig {
    fn default() -> Self {
        Self {
            height: 480,
            width: 640,
            score_threshold: 0.05,
            top_k: 4096,
        }
    }
}

/// The XFeat model. Reuse one instance across frames; all intermediates are
/// pre-allocated. The `&mut self` borrow on `extract` prevents concurrent use.
pub struct XFeat {
    config: XFeatConfig,
    weights: PackedWeights,
    vtable: OpsVtable,

    // ── Sequential ping-pong buffers ──────────────────────────────────────
    // Sized to the largest sequential intermediate: Block-1 step-1 → (H, W, 4).
    buf_a: Vec<f32>,
    buf_b: Vec<f32>,

    // ── Named intermediates that survive across block boundaries ──────────
    norm_gray: Vec<f32>,  // (H, W)    InstanceNorm output; also unfold input
    skip1_pool: Vec<f32>, // (H/4, W/4)  AvgPool output
    x3: Vec<f32>,         // (H/8, W/8, 64)  Block-3 output; FPN accumulator
    x4: Vec<f32>,         // (H/16, W/16, 64)  Block-4 output
    x4_up: Vec<f32>,      // (H/8, W/8, 64)   x4 bilinear-upsampled
    x5_up: Vec<f32>,      // (H/8, W/8, 64)   x5 bilinear-upsampled
    feats: Vec<f32>,      // (H/8, W/8, 64)   block_fusion output; descriptor map

    // ── Head outputs ──────────────────────────────────────────────────────
    h1_rel: Vec<f32>, // (H/8, W/8)     heatmap_head 1-ch sigmoid reliability
    k1_raw: Vec<f32>, // (H/8, W/8, 65) keypoint_head logits (before softmax)
    k1h: Vec<f32>,    // (H, W)         full-res keypoint heatmap after pixel_shuffle

    // ── Sparse output (pre-allocated to top_k) ────────────────────────────
    keypoints: Vec<KeyPoint>,
    descriptors: Vec<f32>,
    reliability_per_kp: Vec<f32>,

    last_scale: InputScale,

    // ── Winograd pre-computation (Problem 1 fix) ──────────────────────────
    // Pre-transformed weights [16 * c_out * c_in] for the 10 eligible layers.
    // Built once in new(); looked up per frame instead of recomputing G·g·G^T.
    winograd_cache: WinogradCache,

    // ── Winograd input-tile scratch reserve ───────────────────────────────
    // Retained for API stability. The F(4,3) drivers manage their own scratch
    // internally (stack accumulator for f32; per-worker heap m_acc for fp16),
    // so this buffer is no longer threaded through the hot path.
    #[allow(dead_code)]
    winograd_v_buf: Vec<f32>,

    // When false (set by with_scalar_backend()), winograd_conv() skips the
    // cache and falls through to the vtable path, preserving scalar-oracle
    // behaviour for parity tests.
    use_winograd_cache: bool,

    // When true (aarch64 + has_fp16 at runtime), winograd_conv() uses the
    // ARMv8.2 fp16 GEMM path instead of the f32 Winograd kernel.
    use_fp16_winograd: bool,

    // ── Pre-packed conv3x3 weights for stride-2 NEON layers ───────────────
    // Eliminates a per-frame Vec<f32> alloc in conv3x3_v2. Layout for each:
    // [c_out/4, 9, c_in, 4] — matches the NEON v2 inner-loop read pattern.
    packed_b1_1: Vec<f32>,  // block1.1  c_in=4,  c_out=8   → 288 f32
    packed_b1_3: Vec<f32>,  // block1.3  c_in=8,  c_out=24  → 1728 f32
    packed_b3_0: Vec<f32>,  // block3.0  c_in=24, c_out=64  → 13824 f32
    packed_b4_0: Vec<f32>,  // block4.0  c_in=64, c_out=64  → 36864 f32
    packed_b5_0: Vec<f32>,  // block5.0  c_in=64, c_out=128 → 73728 f32

    // ── fp16 pre-packed weights for stride-2 direct conv3x3 ───────────────
    // Layout [c_out/8, 9, c_in, 8] fp16-as-u16. Only populated when
    // has_fp16 == true at construction; empty vec → fall back to f32 v2.
    packed_b1_1_f16: Vec<u16>, // block1.1 c_in=4,  c_out=8   → 576 u16
    packed_b1_3_f16: Vec<u16>, // block1.3 c_in=8,  c_out=24  → 3456 u16
    packed_b3_0_f16: Vec<u16>, // block3.0 c_in=24, c_out=64  → 27648 u16
    packed_b4_0_f16: Vec<u16>, // block4.0 c_in=64, c_out=64  → 73728 u16
    packed_b5_0_f16: Vec<u16>, // block5.0 c_in=64, c_out=128 → 147456 u16

    // ── fp16 conv1x1 scratch buffers (Problem 3 fix) ─────────────────────
    // Pre-allocated to eliminate 33+ heap allocs per frame from the 11
    // conv1x1_nhwc_f16 calls.  Sized to the maximum across all 1×1 layers:
    //   scratch_a / scratch_c: max(H×W × c_in/c_out) = (H/8)×(W/8)×128 (block5.3 input)
    //     = 60×80×128 = 614,400 u16 for 480×640
    //   scratch_b: max(c_in × c_out) = 128×128 = 16,384 u16
    //   f16_pack_a: max(c_in × 4) = 128×4 = 512 u16
    //   f16_pack_b: max(c_in × c_out) = 128×128 = 16,384 u16 (full B pre-pack)
    f16_scratch_a: Vec<u16>, // f16 input:   h*w*c_in  (max across all conv1x1 layers)
    f16_scratch_b: Vec<u16>, // f16 weights: c_in*c_out
    f16_scratch_c: Vec<u16>, // f16 output:  h*w*c_out
    f16_pack_a:    Vec<u16>, // GEMM A-panel pack: c_in * 4
    f16_pack_b:    Vec<u16>, // GEMM full B pre-pack: c_in * c_out
}

impl std::fmt::Debug for XFeat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("XFeat")
            .field("config", &self.config)
            .field("backend", &self.vtable)
            .finish()
    }
}

impl XFeat {
    /// Construct an XFeat instance for a fixed input resolution.
    /// Allocates all intermediate and output buffers. Weight verification
    /// is lazy (happens on the first `extract` call) so placeholder weights
    /// can be loaded without failing here.
    pub fn new(config: XFeatConfig, weights: PackedWeights) -> Result<Self, XFeatError> {
        if config.height % 32 != 0 || config.width % 32 != 0 {
            return Err(XFeatError::InputNotAlignedTo32(config.height, config.width));
        }
        let (h, w) = (config.height, config.width);
        let (h4, w4) = (h / 4, w / 4);
        let (h8, w8) = (h / 8, w / 8);
        let (h16, w16) = (h / 16, w / 16);

        // Largest sequential intermediate: Block-1 step-1 output (H, W, 4).
        let seq_max = h * w * 4;

        // ── Build WinogradCache (Problem 1 fix) ──────────────────────────
        // Pre-transform weights for all 10 eligible stride-1 3×3 layers.
        // We do this even if the weights blob is a placeholder (the tensors
        // may be missing). Missing tensors gracefully produce an empty cache
        // and fall through to the vtable path with an Err on extract().
        let mut winograd_cache = WinogradCache {
            transformed: std::collections::HashMap::new(),
            transformed_f16: std::collections::HashMap::new(),
            b_panels_f16: std::collections::HashMap::new(),
            b_panels_packed: std::collections::HashMap::new(),
            fp16_direct_conv: std::collections::HashMap::new(),
            c_out: std::collections::HashMap::new(),
            c_in: std::collections::HashMap::new(),
        };
        // Probe fp16 support once here so we only pay conversion cost when
        // the CPU will actually use the f16 path at run-time.
        #[cfg(target_arch = "aarch64")]
        let build_f16 = crate::cpu_features::cpu_features().has_fp16;
        #[cfg(not(target_arch = "aarch64"))]
        let build_f16 = false;

        for &(layer_name, c_out, c_in) in WINOGRAD_LAYERS {
            let wt_key = format!("{layer_name}.weight");
            if let Ok(wt) = weights.get(&wt_key) {
                let transformed =
                    winograd::winograd_transform_weights_f32_f43(wt, c_out, c_in);

                // Build the f16 copy if the CPU supports fp16 arithmetic.
                if build_f16 {
                    #[cfg(target_arch = "aarch64")]
                    {
                        let mut f16_buf: Vec<u16> = Vec::with_capacity(transformed.len());
                        // SAFETY: transformed is a valid, aligned Vec<f32>.
                        unsafe {
                            crate::ops::neon_asm_f16::f32_to_f16_slice(
                                &transformed,
                                &mut f16_buf,
                            );
                        }
                        // Pre-transpose B panels for the fp16 F(4,3) GEMM driver.
                        // transformed_f16 layout: [36 * c_out * c_in] (position p
                        // at offset p*c_out*c_in, row-major c_out × c_in).
                        // B-panel for GEMM needs [c_in, c_out] per position.
                        // Flat layout: b_panels_f16[p*c_in*c_out + ci*c_out + co].
                        let mut b_panels = vec![0u16; 36 * c_in * c_out];
                        for p in 0..36usize {
                            let pos_offset = p * c_out * c_in;
                            for co in 0..c_out {
                                for ci in 0..c_in {
                                    b_panels[p * c_in * c_out + ci * c_out + co] =
                                        f16_buf[pos_offset + co * c_in + ci];
                                }
                            }
                        }

                        // Pre-pack B panels for zero-copy GEMM: [36, n_blocks, K, NR=8].
                        const NR: usize = 8;
                        let n_blocks = c_out / NR;
                        let n_rem    = c_out % NR;
                        // Each position p: aligned [n_blocks, K, NR] + tail [K, n_rem].
                        let slot_sz = n_blocks * c_in * NR + c_in * n_rem;
                        let mut b_packed = vec![0u16; 36 * slot_sz];
                        for p in 0..36usize {
                            let dst = &mut b_packed[p * slot_sz..];
                            for nb in 0..n_blocks {
                                let nr_start = nb * NR;
                                for ki in 0..c_in {
                                    for ni in 0..NR {
                                        dst[nb * c_in * NR + ki * NR + ni] =
                                            b_panels[p * c_in * c_out + ki * c_out + nr_start + ni];
                                    }
                                }
                            }
                            if n_rem > 0 {
                                let rem_off = n_blocks * c_in * NR;
                                let nr_start = n_blocks * NR;
                                for ki in 0..c_in {
                                    for ni in 0..n_rem {
                                        dst[rem_off + ki * n_rem + ni] =
                                            b_panels[p * c_in * c_out + ki * c_out + nr_start + ni];
                                    }
                                }
                            }
                        }

                        winograd_cache
                            .transformed_f16
                            .insert(layer_name.to_string(), f16_buf);
                        winograd_cache
                            .b_panels_f16
                            .insert(layer_name.to_string(), b_panels);
                        winograd_cache
                            .b_panels_packed
                            .insert(layer_name.to_string(), b_packed);

                        // For small-K layers (c_in ≤ 8 && c_out ≤ 8), also
                        // pre-pack weights in [c_out/8, 9, c_in, 8] fp16 layout
                        // for the direct fp16 conv3x3 bypass path (faster than
                        // Winograd when K is too small to amortise transforms).
                        if c_in <= 8 && c_out <= 8 {
                            let packed_fp16 = repack_weights_co8_3x3_f16(wt, c_out, c_in);
                            winograd_cache
                                .fp16_direct_conv
                                .insert(layer_name.to_string(), packed_fp16);
                        }
                    }
                }

                winograd_cache.transformed.insert(layer_name.to_string(), transformed);
                winograd_cache.c_out.insert(layer_name.to_string(), c_out);
                winograd_cache.c_in.insert(layer_name.to_string(), c_in);
            }
        }

        // ── Pre-pack stride-2 conv3x3 weights (Problem 4 fix) ─────────────
        // Build once per model instance to eliminate per-frame Vec allocs in
        // conv3x3_v2. Falls back to an empty vec if the weights are missing
        // (placeholder weights): conv3x3_v2 will repack on the fly in that case.
        let pack = |key: &str, c_out: usize, c_in: usize| -> Vec<f32> {
            weights
                .get(key)
                .map(|w| repack_weights_co4_3x3(w, c_out, c_in))
                .unwrap_or_default()
        };
        let packed_b1_1 = pack("block1.1.weight", 8,   4);
        let packed_b1_3 = pack("block1.3.weight", 24,  8);
        let packed_b3_0 = pack("block3.0.weight", 64,  24);
        let packed_b4_0 = pack("block4.0.weight", 64,  64);
        let packed_b5_0 = pack("block5.0.weight", 128, 64);

        // fp16 packing — only when the CPU has ARMv8.2 fp16 (build_f16 true).
        let pack_f16 = |key: &str, c_out: usize, c_in: usize| -> Vec<u16> {
            if !build_f16 {
                return Vec::new();
            }
            weights
                .get(key)
                .map(|w| repack_weights_co8_3x3_f16(w, c_out, c_in))
                .unwrap_or_default()
        };
        let packed_b1_1_f16 = pack_f16("block1.1.weight", 8,   4);
        let packed_b1_3_f16 = pack_f16("block1.3.weight", 24,  8);
        let packed_b3_0_f16 = pack_f16("block3.0.weight", 64,  24);
        let packed_b4_0_f16 = pack_f16("block4.0.weight", 64,  64);
        let packed_b5_0_f16 = pack_f16("block5.0.weight", 128, 64);

        // ── Winograd input-tile scratch reserve ──────────────────────────
        // The F(4,3) drivers manage their own scratch internally, so we no
        // longer pre-allocate a large external v_buf. Keep a tiny placeholder.
        let winograd_v_buf_size = 1usize;

        // ── fp16 conv1x1 scratch sizing ──────────────────────────────────
        // Maximum sizes across all 11 conv1x1 call sites in extract():
        //   Inputs into conv1x1:
        //     skip1   → (h4, w4, c_in=1)   → h4*w4*1
        //     block3.2 → (h8, w8, c_in=64) → h8*w8*64
        //     block5.3 → (h32, w32, c_in=128) → h32*w32*128  ← NOT a/c max
        //     block_fusion.2 → (h8, w8, c_in=64)
        //     heatmap_head.0/1 → (h8, w8, c_in=64)
        //     heatmap_head.2 → (h8, w8, c_in=64) → c_out=1
        //     keypoint_head.0/1/2 → (h8, w8, c_in=64)
        //     keypoint_head.3 → (h8, w8, c_in=64) → c_out=65
        //   max a/c = max(h8*w8*128, h32*w32*128)
        //           = max(h8*w8, h32*w32) * 128 — h8*w8 dominates at 480×640
        //   max b   = max c_in * max c_out = 128*128 = 16384
        let f16_scratch_ac = {
            let h32 = h / 32;
            let w32 = w / 32;
            let candidates = [
                h4 * w4 * 1,          // skip1 (tiny)
                h8 * w8 * 128,        // block5.3 output side OR keypoint_head input: 128ch
                h8 * w8 * 65,         // keypoint_head.3 output: 65ch
                h32 * w32 * 128,      // block5.3 input: 128ch at h/32×w/32
            ];
            *candidates.iter().max().unwrap_or(&0)
        };
        let f16_scratch_b  = 128 * 128; // max c_in * max c_out
        let f16_pack_a_sz  = 128 * 4;   // max c_in * MR
        let f16_pack_b_sz  = 128 * 128; // max c_in * c_out — full B pre-pack buffer

        Ok(Self {
            config: config.clone(),
            weights,
            vtable: OpsVtable::select(),

            buf_a: vec![0.0f32; seq_max],
            buf_b: vec![0.0f32; seq_max],

            norm_gray: vec![0.0f32; h * w],
            skip1_pool: vec![0.0f32; h4 * w4],
            x3: vec![0.0f32; h8 * w8 * 64],
            x4: vec![0.0f32; h16 * w16 * 64],
            x4_up: vec![0.0f32; h8 * w8 * 64],
            x5_up: vec![0.0f32; h8 * w8 * 64],
            feats: vec![0.0f32; h8 * w8 * 64],

            h1_rel: vec![0.0f32; h8 * w8],
            k1_raw: vec![0.0f32; h8 * w8 * 65],
            k1h: vec![0.0f32; h * w],

            keypoints: Vec::with_capacity(config.top_k),
            descriptors: Vec::with_capacity(config.top_k * 64),
            reliability_per_kp: Vec::with_capacity(config.top_k),

            last_scale: InputScale {
                rw: 1.0,
                rh: 1.0,
                w_norm: w,
                h_norm: h,
            },

            winograd_cache,
            winograd_v_buf: vec![0.0f32; winograd_v_buf_size],
            use_winograd_cache: true,
            use_fp16_winograd: build_f16,

            packed_b1_1,
            packed_b1_3,
            packed_b3_0,
            packed_b4_0,
            packed_b5_0,

            packed_b1_1_f16,
            packed_b1_3_f16,
            packed_b3_0_f16,
            packed_b4_0_f16,
            packed_b5_0_f16,

            f16_scratch_a: vec![0u16; f16_scratch_ac],
            f16_scratch_b: vec![0u16; f16_scratch_b],
            f16_scratch_c: vec![0u16; f16_scratch_ac],
            f16_pack_a:    vec![0u16; f16_pack_a_sz],
            f16_pack_b:    vec![0u16; f16_pack_b_sz],
        })
    }

    /// Force the scalar backend regardless of CPU support. For parity tests.
    pub fn with_scalar_backend(mut self) -> Self {
        self.vtable = OpsVtable::scalar();
        // Disable Winograd and all fp16 paths so the scalar vtable handles
        // every 3×3 conv call — preserving the scalar parity oracle invariant.
        self.use_winograd_cache = false;
        self.use_fp16_winograd = false;
        // Clear fp16 direct-conv weights so the dispatch falls through to vtable.
        self.packed_b1_1_f16.clear();
        self.packed_b1_3_f16.clear();
        self.packed_b3_0_f16.clear();
        self.packed_b4_0_f16.clear();
        self.packed_b5_0_f16.clear();
        self
    }

    /// Returns the model's configured input shape `(height, width)`.
    pub fn input_shape(&self) -> (usize, usize) {
        (self.config.height, self.config.width)
    }

    /// Run extraction on a single grayscale f32 image of the configured shape.
    ///
    /// Values should be in `[0, 1]`; the model normalises per-image internally.
    /// Returns `Err(XFeatError::Weights)` if the packed weights are missing or
    /// have wrong shapes (placeholder weights gracefully produce this error).
    pub fn extract<'a>(
        &'a mut self,
        gray_f32: &[f32],
    ) -> Result<crate::XFeatOutput<'a>, XFeatError> {
        let h = self.config.height;
        let w = self.config.width;
        if gray_f32.len() != h * w {
            return Err(XFeatError::InputShapeMismatch {
                expected: (h, w),
                got: (gray_f32.len(), 1),
            });
        }

        let (h4, w4) = (h / 4, w / 4);
        let (h8, w8) = (h / 8, w / 8);
        let (h16, w16) = (h / 16, w / 16);
        let (h32, w32) = (h / 32, w / 32);

        let vt = self.vtable; // Copy — no ongoing borrow of self

        // ── Winograd cache lookup helpers ────────────────────────────────
        // The F(4,3) drivers manage their own per-tile-row scratch internally
        // (stack accumulator for f32, heap m_acc for fp16), so no external
        // v_buf is threaded through any more. winograd_v_buf is retained as a
        // pre-allocated reserve but is currently unused by the hot path.
        let use_winograd = self.use_winograd_cache;
        let use_fp16 = self.use_fp16_winograd;

        // ── fp16 conv1x1 scratch pointers ────────────────────────────────
        // Same trick as vbuf_ptr: extract raw pointers so the scratch vecs
        // can be used independently of other &mut self borrows in this fn.
        // SAFETY: each scratch buffer is only accessed at one call site at a
        // time (extract() is single-threaded), never aliasing other borrows.
        let f16_sa_ptr: *mut u16 = self.f16_scratch_a.as_mut_ptr();
        let f16_sa_len: usize    = self.f16_scratch_a.len();
        let f16_sb_ptr: *mut u16 = self.f16_scratch_b.as_mut_ptr();
        let f16_sb_len: usize    = self.f16_scratch_b.len();
        let f16_sc_ptr: *mut u16 = self.f16_scratch_c.as_mut_ptr();
        let f16_sc_len: usize    = self.f16_scratch_c.len();
        let f16_pa_ptr: *mut u16 = self.f16_pack_a.as_mut_ptr();
        let f16_pa_len: usize    = self.f16_pack_a.len();
        let f16_pb_ptr: *mut u16 = self.f16_pack_b.as_mut_ptr();
        let f16_pb_len: usize    = self.f16_pack_b.len();

        // ── fp16 direct-conv3x3 weight pointers ──────────────────────────
        // Store as (ptr-as-usize, len) to avoid borrow-checker conflicts with
        // simultaneous &mut buf_* borrows at the stride-2 conv call sites.
        // SAFETY: packed_b*_f16 are never mutated in extract(); pointers are
        // valid for the lifetime of self.
        let b1_1_f16 = (self.packed_b1_1_f16.as_ptr() as usize, self.packed_b1_1_f16.len());
        let b1_3_f16 = (self.packed_b1_3_f16.as_ptr() as usize, self.packed_b1_3_f16.len());
        let b3_0_f16 = (self.packed_b3_0_f16.as_ptr() as usize, self.packed_b3_0_f16.len());
        let b4_0_f16 = (self.packed_b4_0_f16.as_ptr() as usize, self.packed_b4_0_f16.len());
        let b5_0_f16 = (self.packed_b5_0_f16.as_ptr() as usize, self.packed_b5_0_f16.len());

        /// Dispatch a stride-2 3×3 conv: fp16 direct path when weights are present,
        /// otherwise the NEON v2 f32 vtable entry.
        #[allow(unused_variables)]
        #[inline(always)]
        fn s2_conv<'x>(
            use_fp16: bool,
            vt_s2: fn(&Conv3x3Args<'_>, &mut [f32]),
            args: &Conv3x3Args<'x>,
            out: &mut [f32],
            f16_packed: (usize, usize), // (ptr as usize, len)
        ) {
            #[cfg(target_arch = "aarch64")]
            if use_fp16 && f16_packed.1 > 0 {
                let w = unsafe {
                    core::slice::from_raw_parts(f16_packed.0 as *const u16, f16_packed.1)
                };
                neon::conv3x3_nhwc_fp16(args, out, w, 2);
                return;
            }
            vt_s2(args, out);
        }

        /// Helper: call conv3x3_winograd_nhwc (or the fp16 variant) with a cached weight transform.
        /// Returns true if the cache hit and the conv ran; false → caller uses vtable.
        /// `enabled` is false when `with_scalar_backend()` was called (parity tests).
        /// When `fp16` is true and f16 weights are present, delegates to the fp16 Winograd path.
        #[allow(clippy::too_many_arguments)]
        #[inline(always)]
        fn winograd_conv<'x>(
            enabled: bool,
            fp16: bool,
            cache: &WinogradCache,
            spatial: &'x [f32], // [c_out, 9, c_in] — original weights for epilogue
            bias: &'x [f32],
            layer: &str,
            input: &'x [f32],
            h_in: usize,
            w_in: usize,
            activation: Activation,
            output: &'x mut [f32],
        ) -> bool {
            if !enabled { return false; }
            let Some(wt) = cache.transformed.get(layer) else { return false; };
            let c_out = cache.c_out[layer];
            let c_in = cache.c_in[layer];

            // Small-K fast path: for c_in=c_out=8, direct fp16 conv3x3 beats
            // Winograd F(4,3) because transform overhead dominates for K=8.
            // Pre-packed fp16 weights are stored at model init in fp16_direct_conv.
            #[cfg(target_arch = "aarch64")]
            if fp16 {
                if let Some(packed_f16) = cache.fp16_direct_conv.get(layer) {
                    let args = Conv3x3Args {
                        input, residual: None, weights: spatial, bias,
                        h_in, w_in, c_in, c_out, activation, packed_weights: None,
                    };
                    neon::conv3x3_nhwc_fp16(&args, output, packed_f16, 1);
                    return true;
                }
            }

            // F(4,3): drivers tile in 4×4 and handle row/col epilogues via the
            // scalar fallback using the original spatial weights. h_in/w_in are
            // not always multiples of 4 (e.g. block4.* at 30, block5.* at 15),
            // so always route through the scalar-fallback variant for safety.
            // (For multiples of 4 the fallback adds no extra work.)

            // fp16 fast path: use the ARMv8.2 FMLA.8H GEMM kernel.
            #[cfg(target_arch = "aarch64")]
            if fp16 {
                if let (Some(wt_f16), Some(bp_f16)) =
                    (cache.transformed_f16.get(layer), cache.b_panels_f16.get(layer))
                {
                    let bp_packed = cache.b_panels_packed.get(layer).map_or(&[][..], |v| v);
                    winograd::conv3x3_winograd_nhwc_f43_f16_with_scalar_fallback(
                        input, h_in, w_in, c_in,
                        wt_f16, bp_f16, bp_packed, spatial, Some(bias), c_out,
                        activation, output, h_in, w_in,
                    );
                    return true;
                }
            }
            let _ = fp16; // suppress unused warning on non-aarch64

            winograd::conv3x3_winograd_nhwc_f43_with_scalar_fallback(
                input, h_in, w_in, c_in,
                wt, spatial, Some(bias), c_out,
                activation, output, h_in, w_in,
            );
            true
        }

        /// Zero-alloc conv1x1 dispatch for the fp16 hot path.
        ///
        /// On aarch64 with `use_fp16=true`: calls `conv1x1_nhwc_f16_scratch` with the
        /// caller-provided pre-allocated scratch buffers — zero heap allocs.
        /// Otherwise: falls back to the vtable entry (which allocates its own scratch).
        ///
        /// All ptr/len args are the raw-pointer equivalents of the XFeat scratch fields
        /// extracted at the top of extract() to work around the borrow checker.
        #[allow(clippy::too_many_arguments)]
        #[inline(always)]
        fn conv1x1_f16_direct<'x>(
            use_fp16: bool,
            vt: OpsVtable,
            args: &Conv1x1Args<'x>,
            output: &'x mut [f32],
            sa_ptr: *mut u16, sa_len: usize,
            sb_ptr: *mut u16, sb_len: usize,
            sc_ptr: *mut u16, sc_len: usize,
            pa_ptr: *mut u16, pa_len: usize,
            pb_ptr: *mut u16, pb_len: usize,
        ) {
            #[cfg(target_arch = "aarch64")]
            if use_fp16 {
                // SAFETY: ptrs are valid for their respective lengths; no aliasing
                // since extract() is single-threaded and only one call is active.
                let sa = unsafe { std::slice::from_raw_parts_mut(sa_ptr, sa_len) };
                let sb = unsafe { std::slice::from_raw_parts_mut(sb_ptr, sb_len) };
                let sc = unsafe { std::slice::from_raw_parts_mut(sc_ptr, sc_len) };
                let pa = unsafe { std::slice::from_raw_parts_mut(pa_ptr, pa_len) };
                let pb = unsafe { std::slice::from_raw_parts_mut(pb_ptr, pb_len) };
                let m  = args.h * args.w;
                // Parallel path: Rayon strips with pre-packed B, thread-local scratch.
                // Serial path: pre-allocated sa/sc/pa (zero-alloc) for small M or
                // c_out not a multiple of NR=8 (e.g. keypoint_head.3 c_out=65).
                if m >= 2 * crate::ops::neon_asm_f16::CONV1X1_STRIP_SIZE {
                    crate::ops::neon_asm_f16::conv1x1_nhwc_f16_parallel(
                        args, output, sb, pb,
                    );
                } else {
                    crate::ops::neon_asm_f16::conv1x1_nhwc_f16_scratch(
                        args, output, sa, sb, sc, pa, pb,
                    );
                }
                return;
            }
            // Suppress unused-variable warnings on non-aarch64 or when use_fp16=false.
            let _ = (use_fp16, sa_ptr, sa_len, sb_ptr, sb_len,
                     sc_ptr, sc_len, pa_ptr, pa_len, pb_ptr, pb_len);
            (vt.conv1x1_f16)(args, output);
        }

        // ── Timing scaffold (temporary, remove after profiling) ──────────
        let _timing = std::env::var("XFEAT_TIMING").is_ok();
        macro_rules! t_lap {
            ($label:expr, $t0:ident) => {
                if _timing {
                    let _elapsed = $t0.elapsed();
                    eprintln!("  {:30} {:>8.2}ms", $label, _elapsed.as_secs_f64() * 1000.0);
                    $t0 = std::time::Instant::now();
                }
            };
        }
        let mut _t = std::time::Instant::now();

        // ── 1. InstanceNorm2d ────────────────────────────────────────────
        instance_norm_2d_singlech(gray_f32, &mut self.norm_gray);
        t_lap!("1.norm", _t);

        // ── 2. Block 1 (four 3×3 BasicLayers) ───────────────────────────
        // step 1: conv3x3(1→4, relu)
        {
            let wt = self.weights.get("block1.0.weight")?;
            let bt = self.weights.get("block1.0.bias")?;
            (vt.conv3x3)(
                &Conv3x3Args {
                    input: &self.norm_gray,
                    residual: None,
                    weights: wt,
                    bias: bt,
                    h_in: h,
                    w_in: w,
                    c_in: 1,
                    c_out: 4,
                    activation: Activation::Relu,
                    packed_weights: None,
                },
                &mut self.buf_a[..h * w * 4],
            );
        }
        t_lap!("2.block1.0 conv(1->4)", _t);
        // step 2: conv3x3(4→8, s=2, relu)
        {
            let wt = self.weights.get("block1.1.weight")?;
            let bt = self.weights.get("block1.1.bias")?;
            s2_conv(use_fp16, vt.conv3x3_s2,
                &Conv3x3Args {
                    input: &self.buf_a[..h * w * 4],
                    residual: None,
                    weights: wt,
                    bias: bt,
                    h_in: h, w_in: w, c_in: 4, c_out: 8,
                    activation: Activation::Relu,
                    packed_weights: (!self.packed_b1_1.is_empty()).then_some(&self.packed_b1_1),
                },
                &mut self.buf_b[..h / 2 * (w / 2) * 8],
                b1_1_f16,
            );
        }
        t_lap!("3.block1.1 s2(4->8)", _t);
        // step 3: conv3x3(8→8, relu)  [Winograd-eligible]
        {
            let bt = self.weights.get("block1.2.bias")?;
            let wt_sp = self.weights.get("block1.2.weight")?;
            let hit = winograd_conv(
                use_winograd, use_fp16, &self.winograd_cache, wt_sp,
                bt, "block1.2",
                &self.buf_b[..h / 2 * (w / 2) * 8],
                h / 2, w / 2, Activation::Relu,
                &mut self.buf_a[..h / 2 * (w / 2) * 8],
            );
            if !hit {
                let wt = self.weights.get("block1.2.weight")?;
                (vt.conv3x3)(
                    &Conv3x3Args {
                        input: &self.buf_b[..h / 2 * (w / 2) * 8],
                        residual: None, weights: wt, bias: bt, packed_weights: None,
                        h_in: h / 2, w_in: w / 2, c_in: 8, c_out: 8,
                        activation: Activation::Relu,
                    },
                    &mut self.buf_a[..h / 2 * (w / 2) * 8],
                );
            }
        }
        t_lap!("4.block1.2 wino(8->8)", _t);
        // step 4: conv3x3(8→24, s=2, relu) → buf_b = x1_raw (H/4,W/4,24)
        {
            let wt = self.weights.get("block1.3.weight")?;
            let bt = self.weights.get("block1.3.bias")?;
            s2_conv(use_fp16, vt.conv3x3_s2,
                &Conv3x3Args {
                    input: &self.buf_a[..h / 2 * (w / 2) * 8],
                    residual: None,
                    weights: wt,
                    bias: bt,
                    h_in: h / 2, w_in: w / 2, c_in: 8, c_out: 24,
                    activation: Activation::Relu,
                    packed_weights: (!self.packed_b1_3.is_empty()).then_some(&self.packed_b1_3),
                },
                &mut self.buf_b[..h4 * w4 * 24],
                b1_3_f16,
            );
        }

        t_lap!("5.block1.3 s2(8->24)", _t);
        // ── 3. Skip1 (AvgPool + conv1x1 with real bias) ──────────────────
        avgpool_4x4_s4(&self.norm_gray, &mut self.skip1_pool, h, w, 1);
        {
            let wt = self.weights.get("skip1.weight")?;
            let bt = self.weights.get("skip1.bias")?;
            conv1x1_f16_direct(use_fp16, vt,
                &Conv1x1Args {
                    input: &self.skip1_pool,
                    weights: wt,
                    bias: bt,
                    h: h4,
                    w: w4,
                    c_in: 1,
                    c_out: 24,
                    activation: Activation::Identity,
                },
                &mut self.buf_a[..h4 * w4 * 24],
                f16_sa_ptr, f16_sa_len,
                f16_sb_ptr, f16_sb_len,
                f16_sc_ptr, f16_sc_len,
                f16_pa_ptr, f16_pa_len,
                f16_pb_ptr, f16_pb_len,
            );
        }

        t_lap!("6.skip1 avgpool+1x1", _t);
        // ── 4. x1 = x1_raw + skip1 ───────────────────────────────────────
        // buf_b holds x1_raw, buf_a holds skip1; add buf_a into buf_b.
        add_inplace(&mut self.buf_b[..h4 * w4 * 24], &self.buf_a[..h4 * w4 * 24]);

        t_lap!("7.x1=add", _t);
        // ── 5. Block 2 (two 3×3 BasicLayers) [Winograd-eligible] ────────
        {
            let bt = self.weights.get("block2.0.bias")?;
            let wt_sp = self.weights.get("block2.0.weight")?;
            let hit = winograd_conv(
                use_winograd, use_fp16, &self.winograd_cache, wt_sp,
                bt, "block2.0",
                &self.buf_b[..h4 * w4 * 24],
                h4, w4, Activation::Relu,
                &mut self.buf_a[..h4 * w4 * 24],
            );
            if !hit {
                let wt = self.weights.get("block2.0.weight")?;
                (vt.conv3x3)(
                    &Conv3x3Args {
                        input: &self.buf_b[..h4 * w4 * 24],
                        residual: None, weights: wt, bias: bt, packed_weights: None,
                        h_in: h4, w_in: w4, c_in: 24, c_out: 24,
                        activation: Activation::Relu,
                    },
                    &mut self.buf_a[..h4 * w4 * 24],
                );
            }
        }
        {
            let bt = self.weights.get("block2.1.bias")?;
            let wt_sp = self.weights.get("block2.1.weight")?;
            let hit = winograd_conv(
                use_winograd, use_fp16, &self.winograd_cache, wt_sp,
                bt, "block2.1",
                &self.buf_a[..h4 * w4 * 24],
                h4, w4, Activation::Relu,
                &mut self.buf_b[..h4 * w4 * 24],
            );
            if !hit {
                let wt = self.weights.get("block2.1.weight")?;
                (vt.conv3x3)(
                    &Conv3x3Args {
                        input: &self.buf_a[..h4 * w4 * 24],
                        residual: None, weights: wt, bias: bt, packed_weights: None,
                        h_in: h4, w_in: w4, c_in: 24, c_out: 24,
                        activation: Activation::Relu,
                    },
                    &mut self.buf_b[..h4 * w4 * 24],
                );
            }
        }

        t_lap!("8.block2 2x wino(24->24)", _t);
        // ── 6. Block 3 (two 3×3 + one 1×1 BasicLayer) ───────────────────
        {
            let wt = self.weights.get("block3.0.weight")?;
            let bt = self.weights.get("block3.0.bias")?;
            s2_conv(use_fp16, vt.conv3x3_s2,
                &Conv3x3Args {
                    input: &self.buf_b[..h4 * w4 * 24],
                    residual: None,
                    weights: wt,
                    bias: bt,
                    h_in: h4, w_in: w4, c_in: 24, c_out: 64,
                    activation: Activation::Relu,
                    packed_weights: (!self.packed_b3_0.is_empty()).then_some(&self.packed_b3_0),
                },
                &mut self.buf_a[..h8 * w8 * 64],
                b3_0_f16,
            );
        }
        {
            // block3.1: conv3x3(64→64, relu)  [Winograd-eligible]
            let bt = self.weights.get("block3.1.bias")?;
            let wt_sp = self.weights.get("block3.1.weight")?;
            let hit = winograd_conv(
                use_winograd, use_fp16, &self.winograd_cache, wt_sp,
                bt, "block3.1",
                &self.buf_a[..h8 * w8 * 64],
                h8, w8, Activation::Relu,
                &mut self.buf_b[..h8 * w8 * 64],
            );
            if !hit {
                let wt = self.weights.get("block3.1.weight")?;
                (vt.conv3x3)(
                    &Conv3x3Args {
                        input: &self.buf_a[..h8 * w8 * 64],
                        residual: None, weights: wt, bias: bt, packed_weights: None,
                        h_in: h8, w_in: w8, c_in: 64, c_out: 64,
                        activation: Activation::Relu,
                    },
                    &mut self.buf_b[..h8 * w8 * 64],
                );
            }
        }
        // block3.2: 1×1 BasicLayer (BN-folded, relu) → x3
        {
            let wt = self.weights.get("block3.2.weight")?;
            let bt = self.weights.get("block3.2.bias")?;
            conv1x1_f16_direct(use_fp16, vt,
                &Conv1x1Args {
                    input: &self.buf_b[..h8 * w8 * 64],
                    weights: wt,
                    bias: bt,
                    h: h8,
                    w: w8,
                    c_in: 64,
                    c_out: 64,
                    activation: Activation::Relu,
                },
                &mut self.x3,
                f16_sa_ptr, f16_sa_len,
                f16_sb_ptr, f16_sb_len,
                f16_sc_ptr, f16_sc_len,
                f16_pa_ptr, f16_pa_len,
                f16_pb_ptr, f16_pb_len,
            );
        }

        t_lap!("9.block3 s2+wino+1x1", _t);
        // ── 7. Block 4 (three 3×3 BasicLayers) ───────────────────────────
        {
            let wt = self.weights.get("block4.0.weight")?;
            let bt = self.weights.get("block4.0.bias")?;
            s2_conv(use_fp16, vt.conv3x3_s2,
                &Conv3x3Args {
                    input: &self.x3,
                    residual: None,
                    weights: wt,
                    bias: bt,
                    h_in: h8, w_in: w8, c_in: 64, c_out: 64,
                    activation: Activation::Relu,
                    packed_weights: (!self.packed_b4_0.is_empty()).then_some(&self.packed_b4_0),
                },
                &mut self.buf_a[..h16 * w16 * 64],
                b4_0_f16,
            );
        }
        {
            // block4.1: conv3x3(64→64, relu)  [Winograd-eligible]
            let bt = self.weights.get("block4.1.bias")?;
            let wt_sp = self.weights.get("block4.1.weight")?;
            let hit = winograd_conv(
                use_winograd, use_fp16, &self.winograd_cache, wt_sp,
                bt, "block4.1",
                &self.buf_a[..h16 * w16 * 64],
                h16, w16, Activation::Relu,
                &mut self.buf_b[..h16 * w16 * 64],
            );
            if !hit {
                let wt = self.weights.get("block4.1.weight")?;
                (vt.conv3x3)(
                    &Conv3x3Args {
                        input: &self.buf_a[..h16 * w16 * 64],
                        residual: None, weights: wt, bias: bt, packed_weights: None,
                        h_in: h16, w_in: w16, c_in: 64, c_out: 64,
                        activation: Activation::Relu,
                    },
                    &mut self.buf_b[..h16 * w16 * 64],
                );
            }
        }
        {
            // block4.2: conv3x3(64→64, relu)  [Winograd-eligible]
            let bt = self.weights.get("block4.2.bias")?;
            let wt_sp = self.weights.get("block4.2.weight")?;
            let hit = winograd_conv(
                use_winograd, use_fp16, &self.winograd_cache, wt_sp,
                bt, "block4.2",
                &self.buf_b[..h16 * w16 * 64],
                h16, w16, Activation::Relu,
                &mut self.x4,
            );
            if !hit {
                let wt = self.weights.get("block4.2.weight")?;
                (vt.conv3x3)(
                    &Conv3x3Args {
                        input: &self.buf_b[..h16 * w16 * 64],
                        residual: None, weights: wt, bias: bt, packed_weights: None,
                        h_in: h16, w_in: w16, c_in: 64, c_out: 64,
                        activation: Activation::Relu,
                    },
                    &mut self.x4,
                );
            }
        }

        t_lap!("10.block4 s2+2xwino", _t);
        // ── 8. Block 5 (three 3×3 + one 1×1 BasicLayer) ─────────────────
        {
            let wt = self.weights.get("block5.0.weight")?;
            let bt = self.weights.get("block5.0.bias")?;
            s2_conv(use_fp16, vt.conv3x3_s2,
                &Conv3x3Args {
                    input: &self.x4,
                    residual: None,
                    weights: wt,
                    bias: bt,
                    h_in: h16, w_in: w16, c_in: 64, c_out: 128,
                    activation: Activation::Relu,
                    packed_weights: (!self.packed_b5_0.is_empty()).then_some(&self.packed_b5_0),
                },
                &mut self.buf_a[..h32 * w32 * 128],
                b5_0_f16,
            );
        }
        {
            // block5.1: conv3x3(128→128, relu)  [Winograd-eligible]
            let bt = self.weights.get("block5.1.bias")?;
            let wt_sp = self.weights.get("block5.1.weight")?;
            let hit = winograd_conv(
                use_winograd, use_fp16, &self.winograd_cache, wt_sp,
                bt, "block5.1",
                &self.buf_a[..h32 * w32 * 128],
                h32, w32, Activation::Relu,
                &mut self.buf_b[..h32 * w32 * 128],
            );
            if !hit {
                let wt = self.weights.get("block5.1.weight")?;
                (vt.conv3x3)(
                    &Conv3x3Args {
                        input: &self.buf_a[..h32 * w32 * 128],
                        residual: None, weights: wt, bias: bt, packed_weights: None,
                        h_in: h32, w_in: w32, c_in: 128, c_out: 128,
                        activation: Activation::Relu,
                    },
                    &mut self.buf_b[..h32 * w32 * 128],
                );
            }
        }
        {
            // block5.2: conv3x3(128→128, relu)  [Winograd-eligible]
            let bt = self.weights.get("block5.2.bias")?;
            let wt_sp = self.weights.get("block5.2.weight")?;
            let hit = winograd_conv(
                use_winograd, use_fp16, &self.winograd_cache, wt_sp,
                bt, "block5.2",
                &self.buf_b[..h32 * w32 * 128],
                h32, w32, Activation::Relu,
                &mut self.buf_a[..h32 * w32 * 128],
            );
            if !hit {
                let wt = self.weights.get("block5.2.weight")?;
                (vt.conv3x3)(
                    &Conv3x3Args {
                        input: &self.buf_b[..h32 * w32 * 128],
                        residual: None, weights: wt, bias: bt, packed_weights: None,
                        h_in: h32, w_in: w32, c_in: 128, c_out: 128,
                        activation: Activation::Relu,
                    },
                    &mut self.buf_a[..h32 * w32 * 128],
                );
            }
        }
        // block5.3: 1×1 BasicLayer (BN-folded, relu) → buf_b = x5 (H/32,W/32,64)
        {
            let wt = self.weights.get("block5.3.weight")?;
            let bt = self.weights.get("block5.3.bias")?;
            conv1x1_f16_direct(use_fp16, vt,
                &Conv1x1Args {
                    input: &self.buf_a[..h32 * w32 * 128],
                    weights: wt,
                    bias: bt,
                    h: h32,
                    w: w32,
                    c_in: 128,
                    c_out: 64,
                    activation: Activation::Relu,
                },
                &mut self.buf_b[..h32 * w32 * 64],
                f16_sa_ptr, f16_sa_len,
                f16_sb_ptr, f16_sb_len,
                f16_sc_ptr, f16_sc_len,
                f16_pa_ptr, f16_pa_len,
                f16_pb_ptr, f16_pb_len,
            );
        }

        t_lap!("11.block5 s2+2xwino+1x1", _t);
        // ── 9. FPN: upsample x4, x5 to H/8×W/8, sum into x3 ────────────
        bilinear_upsample(&self.x4, &mut self.x4_up, h16, w16, 64, h8, w8);
        bilinear_upsample(
            &self.buf_b[..h32 * w32 * 64],
            &mut self.x5_up,
            h32,
            w32,
            64,
            h8,
            w8,
        );
        add3_inplace(&mut self.x3, &self.x4_up, &self.x5_up);

        t_lap!("12.FPN upsample+add3", _t);
        // ── 10. block_fusion (two 3×3 BasicLayers + one 1×1 plain conv) ──
        {
            // block_fusion.0: conv3x3(64→64, relu)  [Winograd-eligible]
            let bt = self.weights.get("block_fusion.0.bias")?;
            let wt_sp = self.weights.get("block_fusion.0.weight")?;
            let hit = winograd_conv(
                use_winograd, use_fp16, &self.winograd_cache, wt_sp,
                bt, "block_fusion.0",
                &self.x3,
                h8, w8, Activation::Relu,
                &mut self.buf_a[..h8 * w8 * 64],
            );
            if !hit {
                let wt = self.weights.get("block_fusion.0.weight")?;
                (vt.conv3x3)(
                    &Conv3x3Args {
                        input: &self.x3,
                        residual: None, weights: wt, bias: bt, packed_weights: None,
                        h_in: h8, w_in: w8, c_in: 64, c_out: 64,
                        activation: Activation::Relu,
                    },
                    &mut self.buf_a[..h8 * w8 * 64],
                );
            }
        }
        {
            // block_fusion.1: conv3x3(64→64, relu)  [Winograd-eligible]
            let bt = self.weights.get("block_fusion.1.bias")?;
            let wt_sp = self.weights.get("block_fusion.1.weight")?;
            let hit = winograd_conv(
                use_winograd, use_fp16, &self.winograd_cache, wt_sp,
                bt, "block_fusion.1",
                &self.buf_a[..h8 * w8 * 64],
                h8, w8, Activation::Relu,
                &mut self.buf_b[..h8 * w8 * 64],
            );
            if !hit {
                let wt = self.weights.get("block_fusion.1.weight")?;
                (vt.conv3x3)(
                    &Conv3x3Args {
                        input: &self.buf_a[..h8 * w8 * 64],
                        residual: None, weights: wt, bias: bt, packed_weights: None,
                        h_in: h8, w_in: w8, c_in: 64, c_out: 64,
                        activation: Activation::Relu,
                    },
                    &mut self.buf_b[..h8 * w8 * 64],
                );
            }
        }
        // block_fusion.2: plain conv1x1 with real bias, no activation
        {
            let wt = self.weights.get("block_fusion.2.weight")?;
            let bt = self.weights.get("block_fusion.2.bias")?;
            conv1x1_f16_direct(use_fp16, vt,
                &Conv1x1Args {
                    input: &self.buf_b[..h8 * w8 * 64],
                    weights: wt,
                    bias: bt,
                    h: h8,
                    w: w8,
                    c_in: 64,
                    c_out: 64,
                    activation: Activation::Identity,
                },
                &mut self.feats,
                f16_sa_ptr, f16_sa_len,
                f16_sb_ptr, f16_sb_len,
                f16_sc_ptr, f16_sc_len,
                f16_pa_ptr, f16_pa_len,
                f16_pb_ptr, f16_pb_len,
            );
        }

        t_lap!("13.block_fusion 2xwino+1x1", _t);
        // ── 11. Heatmap head (2 × 1×1 BasicLayer + conv1x1(64→1)+sigmoid) ─
        // Outputs single-channel reliability map h1_rel.
        {
            let wt = self.weights.get("heatmap_head.0.weight")?;
            let bt = self.weights.get("heatmap_head.0.bias")?;
            conv1x1_f16_direct(use_fp16, vt,
                &Conv1x1Args {
                    input: &self.feats,
                    weights: wt,
                    bias: bt,
                    h: h8,
                    w: w8,
                    c_in: 64,
                    c_out: 64,
                    activation: Activation::Relu,
                },
                &mut self.buf_a[..h8 * w8 * 64],
                f16_sa_ptr, f16_sa_len,
                f16_sb_ptr, f16_sb_len,
                f16_sc_ptr, f16_sc_len,
                f16_pa_ptr, f16_pa_len,
                f16_pb_ptr, f16_pb_len,
            );
        }
        {
            let wt = self.weights.get("heatmap_head.1.weight")?;
            let bt = self.weights.get("heatmap_head.1.bias")?;
            conv1x1_f16_direct(use_fp16, vt,
                &Conv1x1Args {
                    input: &self.buf_a[..h8 * w8 * 64],
                    weights: wt,
                    bias: bt,
                    h: h8,
                    w: w8,
                    c_in: 64,
                    c_out: 64,
                    activation: Activation::Relu,
                },
                &mut self.buf_b[..h8 * w8 * 64],
                f16_sa_ptr, f16_sa_len,
                f16_sb_ptr, f16_sb_len,
                f16_sc_ptr, f16_sc_len,
                f16_pa_ptr, f16_pa_len,
                f16_pb_ptr, f16_pb_len,
            );
        }
        // heatmap_head.2: conv1x1(64→1) + Sigmoid → h1_rel (H/8,W/8)
        {
            let wt = self.weights.get("heatmap_head.2.weight")?;
            let bt = self.weights.get("heatmap_head.2.bias")?;
            conv1x1_f16_direct(use_fp16, vt,
                &Conv1x1Args {
                    input: &self.buf_b[..h8 * w8 * 64],
                    weights: wt,
                    bias: bt,
                    h: h8,
                    w: w8,
                    c_in: 64,
                    c_out: 1,
                    activation: Activation::Sigmoid,
                },
                &mut self.h1_rel,
                f16_sa_ptr, f16_sa_len,
                f16_sb_ptr, f16_sb_len,
                f16_sc_ptr, f16_sc_len,
                f16_pa_ptr, f16_pa_len,
                f16_pb_ptr, f16_pb_len,
            );
        }

        t_lap!("14.heatmap_head 3x1x1", _t);
        // ── 12. Keypoint head (input = unfold_8x8(norm_gray)) ────────────
        // Takes the 8×8 patches of the InstanceNorm'd gray image at H/8 × W/8
        // grid positions (64 channels), then processes with 1×1 layers.
        unfold_8x8(&self.norm_gray, &mut self.buf_a[..h8 * w8 * 64], h, w);
        {
            let wt = self.weights.get("keypoint_head.0.weight")?;
            let bt = self.weights.get("keypoint_head.0.bias")?;
            conv1x1_f16_direct(use_fp16, vt,
                &Conv1x1Args {
                    input: &self.buf_a[..h8 * w8 * 64],
                    weights: wt,
                    bias: bt,
                    h: h8,
                    w: w8,
                    c_in: 64,
                    c_out: 64,
                    activation: Activation::Relu,
                },
                &mut self.buf_b[..h8 * w8 * 64],
                f16_sa_ptr, f16_sa_len,
                f16_sb_ptr, f16_sb_len,
                f16_sc_ptr, f16_sc_len,
                f16_pa_ptr, f16_pa_len,
                f16_pb_ptr, f16_pb_len,
            );
        }
        {
            let wt = self.weights.get("keypoint_head.1.weight")?;
            let bt = self.weights.get("keypoint_head.1.bias")?;
            conv1x1_f16_direct(use_fp16, vt,
                &Conv1x1Args {
                    input: &self.buf_b[..h8 * w8 * 64],
                    weights: wt,
                    bias: bt,
                    h: h8,
                    w: w8,
                    c_in: 64,
                    c_out: 64,
                    activation: Activation::Relu,
                },
                &mut self.buf_a[..h8 * w8 * 64],
                f16_sa_ptr, f16_sa_len,
                f16_sb_ptr, f16_sb_len,
                f16_sc_ptr, f16_sc_len,
                f16_pa_ptr, f16_pa_len,
                f16_pb_ptr, f16_pb_len,
            );
        }
        {
            let wt = self.weights.get("keypoint_head.2.weight")?;
            let bt = self.weights.get("keypoint_head.2.bias")?;
            conv1x1_f16_direct(use_fp16, vt,
                &Conv1x1Args {
                    input: &self.buf_a[..h8 * w8 * 64],
                    weights: wt,
                    bias: bt,
                    h: h8,
                    w: w8,
                    c_in: 64,
                    c_out: 64,
                    activation: Activation::Relu,
                },
                &mut self.buf_b[..h8 * w8 * 64],
                f16_sa_ptr, f16_sa_len,
                f16_sb_ptr, f16_sb_len,
                f16_sc_ptr, f16_sc_len,
                f16_pa_ptr, f16_pa_len,
                f16_pb_ptr, f16_pb_len,
            );
        }
        // keypoint_head.3: plain conv1x1(64→65) with real bias → k1_raw
        {
            let wt = self.weights.get("keypoint_head.3.weight")?;
            let bt = self.weights.get("keypoint_head.3.bias")?;
            conv1x1_f16_direct(use_fp16, vt,
                &Conv1x1Args {
                    input: &self.buf_b[..h8 * w8 * 64],
                    weights: wt,
                    bias: bt,
                    h: h8,
                    w: w8,
                    c_in: 64,
                    c_out: 65,
                    activation: Activation::Identity,
                },
                &mut self.k1_raw,
                f16_sa_ptr, f16_sa_len,
                f16_sb_ptr, f16_sb_len,
                f16_sc_ptr, f16_sc_len,
                f16_pa_ptr, f16_pa_len,
                f16_pb_ptr, f16_pb_len,
            );
        }
        t_lap!("15.kp_head unfold+4x1x1", _t);
        // softmax(65 ch) → drop dustbin → pixel_shuffle(8) → K1h (H, W)
        channel_softmax(&mut self.k1_raw, h8, w8, 65);
        drop_last_channel_nhwc(
            &self.k1_raw,
            &mut self.buf_a[..h8 * w8 * 64],
            h8,
            w8,
            65,
            64,
        );
        pixel_shuffle_8(&self.buf_a[..h8 * w8 * 64], &mut self.k1h, h8, w8);

        t_lap!("16.softmax+shuffle", _t);
        // ── 13. L2-normalise descriptor map (M1 = feats) ─────────────────
        l2_normalize_channel(&mut self.feats, h8, w8, 64);

        t_lap!("17.l2_norm", _t);
        // ── 14. NMS + sparse descriptor sampling ─────────────────────────
        let rw = self.last_scale.rw;
        let rh = self.last_scale.rh;

        // Reliability is the heatmap_head sigmoid output sampled at each
        // keypoint position (in H/8×W/8 coords). h1_rel is already a flat
        // (H/8 × W/8) single-channel map ready for nms_topk.
        let kps = nms_topk(
            &self.k1h,
            &self.h1_rel,
            h,
            w,
            h8,
            w8,
            self.config.score_threshold,
            self.config.top_k,
            rw,
            rh,
        );

        // Match PyTorch's InterpolateSparse2d(bicubic) coordinate convention:
        //   x_desc = x_full * W_desc / (W_full - 1) - 0.5
        let x_desc_scale = (w8 as f32) / ((w as f32) - 1.0);
        let y_desc_scale = (h8 as f32) / ((h as f32) - 1.0);
        let kps_desc_coords: Vec<(f32, f32)> = kps
            .iter()
            .map(|kp| {
                let xf = kp.x / rw;
                let yf = kp.y / rh;
                (xf * x_desc_scale - 0.5, yf * y_desc_scale - 0.5)
            })
            .collect();

        self.descriptors.resize(kps.len() * 64, 0.0);
        if !kps.is_empty() {
            bicubic_sample_descriptors(
                &self.feats,
                h8,
                w8,
                64,
                &kps_desc_coords,
                &mut self.descriptors,
            );
            // Re-normalise after bicubic interpolation shifts the norm.
            // Each 64-dim chunk is independent — parallel over keypoints.
            self.descriptors.par_chunks_mut(64).for_each(|chunk| {
                let norm = chunk.iter().map(|&x| x * x).sum::<f32>().sqrt();
                let inv = 1.0 / (norm + 1e-12);
                for x in chunk {
                    *x *= inv;
                }
            });
        }

        self.reliability_per_kp.resize(kps.len(), 0.0);
        for (i, kp) in kps.iter().enumerate() {
            self.reliability_per_kp[i] = kp.score;
        }
        self.keypoints = kps;

        t_lap!("18.NMS+bicubic", _t);
        Ok(crate::XFeatOutput {
            keypoints: &self.keypoints,
            descriptors: &self.descriptors,
            reliability: &self.reliability_per_kp,
        })
    }
}

impl XFeat {
    /// Full-resolution keypoint heatmap (H × W) from the last `extract` call.
    #[doc(hidden)]
    pub fn k1h_slice(&self) -> &[f32] {
        &self.k1h
    }
    /// Half-resolution reliability map (H/8 × W/8) from the last `extract` call.
    #[doc(hidden)]
    pub fn h1_rel_slice(&self) -> &[f32] {
        &self.h1_rel
    }
}
