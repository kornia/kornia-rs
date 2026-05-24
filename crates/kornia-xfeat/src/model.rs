//! The XFeat model graph — `XFeat::extract(image) -> XFeatOutput`.
//!
//! All allocations happen once at construction time. The hot path (per frame) is
//! zero-alloc except for the sparse keypoint output, which is bounded by `top_k`.
//!
//! ## Architecture (upstream `modules/model.py`)
//!
//! ```text
//! gray (H,W)
//!   └─ InstanceNorm2d → norm_gray (H,W,1)
//!       ├─ Block 1: conv(1→4) → conv(4→8,s2) → conv(8→8) → conv(8→24,s2)   x1_raw (H/4,W/4,24)
//!       ├─ Skip1:  AvgPool(4×4) → conv1x1(1→24)                              skip (H/4,W/4,24)
//!       └─ x1 = x1_raw + skip                                                 (H/4,W/4,24)
//!           └─ Block 2: 2× conv(24→24)                                       x2 (H/4,W/4,24)
//!               └─ Block 3: conv(24→64,s2) → 2× conv(64→64)                  x3 (H/8,W/8,64)
//!                   ├─ Block 4: conv(64→64,s2) → 2× conv(64→64)              x4 (H/16,W/16,64)
//!                   │   └─ Block 5: conv(64→128,s2) → 2× conv(128→128)
//!                   │               → conv1x1(128→64)                         x5 (H/32,W/32,64)
//!                   └─ feats = block_fusion(x3 + up(x4) + up(x5))           (H/8,W/8,64)
//!                       ├─ heatmap_head: 2× conv(64→64) → conv1x1(64→65)
//!                       │   → softmax/65 → drop-dustbin → pixel_shuffle(8)   k1h (H,W)
//!                       └─ keypoint_head: 2× conv(64→64) → conv1x1(64→64)   h1 (H/8,W/8,64)
//! ```
//!
//! ## Buffer strategy
//!
//! `buf_a` and `buf_b` are general-purpose ping-pong temporaries for sequential
//! chains within each block. Named buffers (`norm_gray`, `x3`, `x4`, `feats`,
//! …) persist across block boundaries. All are pre-allocated in `new`.

use crate::ops::{
    Activation, Conv1x1Args, Conv3x3Args, OpsVtable,
    add_inplace, add3_inplace, avgpool_4x4_s4, bilinear_upsample, channel_softmax,
    drop_last_channel_nhwc, instance_norm_2d_singlech, l2_normalize_channel, pixel_shuffle_8,
};
use crate::postproc::{bicubic_sample_descriptors, nms_topk, KeyPoint};
use crate::preproc::InputScale;
use crate::weights::PackedWeights;
use crate::XFeatError;

/// Static configuration of an XFeat instance.
///
/// Shape parameters are fixed at construction; calling `extract` with a
/// differently-sized image returns [`XFeatError::InputShapeMismatch`].
#[derive(Debug, Clone)]
pub struct XFeatConfig {
    /// Height (in pixels) that the model is configured for. Must be a multiple of 32.
    pub height: usize,
    /// Width (in pixels). Must be a multiple of 32.
    pub width: usize,
    /// Score threshold for NMS keypoint acceptance. Upstream default: 0.05.
    pub score_threshold: f32,
    /// Maximum number of keypoints to return. Upstream default: 4096.
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

/// The XFeat model. Construct with [`XFeat::new`], reuse for the lifetime of
/// a workload. All intermediates are pre-allocated; the only per-frame heap
/// activity is the sparse keypoint/descriptor output (bounded by `top_k`).
pub struct XFeat {
    config: XFeatConfig,
    weights: PackedWeights,
    vtable: OpsVtable,

    // ── General-purpose sequential buffers (ping-pong within blocks) ──────────
    // Sized to the largest sequential intermediate: Block 1 step 1 → (H, W, 4).
    buf_a: Vec<f32>,
    buf_b: Vec<f32>,

    // ── Named intermediates that survive across block boundaries ──────────────
    norm_gray: Vec<f32>,   // (H, W) after InstanceNorm2d
    skip1_pool: Vec<f32>,  // (H/4, W/4) after AvgPool4×4
    x3: Vec<f32>,          // (H/8, W/8, 64) Block-3 output; also the FPN accumulator
    x4: Vec<f32>,          // (H/16, W/16, 64) Block-4 output
    x4_up: Vec<f32>,       // (H/8, W/8, 64) x4 upsampled for FPN
    x5_up: Vec<f32>,       // (H/8, W/8, 64) x5 upsampled for FPN
    feats: Vec<f32>,       // (H/8, W/8, 64) block_fusion output; used as descriptors

    // ── Head outputs ──────────────────────────────────────────────────────────
    k1_raw: Vec<f32>,  // (H/8, W/8, 65) heatmap_head pre-softmax
    h1_desc: Vec<f32>, // (H/8, W/8, 64) keypoint_head output
    k1h: Vec<f32>,     // (H, W) keypoint heatmap after softmax + pixel_shuffle

    // ── Sparse outputs (pre-allocated to top_k) ───────────────────────────────
    keypoints: Vec<KeyPoint>,
    descriptors: Vec<f32>,
    reliability_per_kp: Vec<f32>,

    /// Input scale from the last `extract` call (caller may resize before calling).
    last_scale: InputScale,
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
    ///
    /// Allocates all intermediate buffers and output buffers. Does **not**
    /// verify weight shapes — that happens lazily in [`extract`] so that
    /// placeholder weights can be loaded for smoke tests without failing here.
    pub fn new(config: XFeatConfig, weights: PackedWeights) -> Result<Self, XFeatError> {
        if config.height % 32 != 0 || config.width % 32 != 0 {
            return Err(XFeatError::InputNotAlignedTo32(config.height, config.width));
        }

        let (h, w) = (config.height, config.width);
        let (h4, w4) = (h / 4, w / 4);
        let (h8, w8) = (h / 8, w / 8);
        let (h16, w16) = (h / 16, w / 16);

        // Largest sequential intermediate: Block-1 step 1 → (H, W, 4)
        let seq_max = h * w * 4;

        let last_scale = InputScale { rw: 1.0, rh: 1.0, w_norm: w, h_norm: h };

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

            k1_raw: vec![0.0f32; h8 * w8 * 65],
            h1_desc: vec![0.0f32; h8 * w8 * 64],
            k1h: vec![0.0f32; h * w],

            keypoints: Vec::with_capacity(config.top_k),
            descriptors: Vec::with_capacity(config.top_k * 64),
            reliability_per_kp: Vec::with_capacity(config.top_k),
            last_scale,
        })
    }

    /// Force the scalar backend regardless of CPU support. For parity tests.
    pub fn with_scalar_backend(mut self) -> Self {
        self.vtable = OpsVtable::scalar();
        self
    }

    /// Returns the model's configured input shape `(height, width)`.
    pub fn input_shape(&self) -> (usize, usize) {
        (self.config.height, self.config.width)
    }

    /// Run extraction on a single grayscale f32 image of the configured shape.
    ///
    /// Input must be `(height, width)` exactly (caller is responsible for the
    /// 32-multiple alignment via [`crate::preproc::align_to_32`] + resample).
    /// Values should be in `[0, 1]`; the model normalises per-image internally.
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

        // Copy vtable to free self for field borrows below.
        let vt = self.vtable;

        // ── 1. InstanceNorm2d ────────────────────────────────────────────────
        instance_norm_2d_singlech(gray_f32, &mut self.norm_gray);

        // ── 2. Block 1 ──────────────────────────────────────────────────────
        // conv3x3(1→4, relu)
        {
            let w_10 = self.weights.get("block1.0.weight")?;
            let b_10 = self.weights.get("block1.0.bias")?;
            (vt.conv3x3)(
                &Conv3x3Args {
                    input: &self.norm_gray,
                    residual: None,
                    weights: w_10,
                    bias: b_10,
                    h_in: h, w_in: w, c_in: 1, c_out: 4,
                    activation: Activation::Relu,
                },
                &mut self.buf_a[..h * w * 4],
            );
        }
        // conv3x3(4→8, s=2, relu)
        {
            let w_11 = self.weights.get("block1.1.weight")?;
            let b_11 = self.weights.get("block1.1.bias")?;
            (vt.conv3x3_s2)(
                &Conv3x3Args {
                    input: &self.buf_a[..h * w * 4],
                    residual: None,
                    weights: w_11,
                    bias: b_11,
                    h_in: h, w_in: w, c_in: 4, c_out: 8,
                    activation: Activation::Relu,
                },
                &mut self.buf_b[..h / 2 * (w / 2) * 8],
            );
        }
        // conv3x3(8→8, relu)
        {
            let w_12 = self.weights.get("block1.2.weight")?;
            let b_12 = self.weights.get("block1.2.bias")?;
            (vt.conv3x3)(
                &Conv3x3Args {
                    input: &self.buf_b[..h / 2 * (w / 2) * 8],
                    residual: None,
                    weights: w_12,
                    bias: b_12,
                    h_in: h / 2, w_in: w / 2, c_in: 8, c_out: 8,
                    activation: Activation::Relu,
                },
                &mut self.buf_a[..h / 2 * (w / 2) * 8],
            );
        }
        // conv3x3(8→24, s=2, relu) → buf_b as x1_raw (H/4, W/4, 24)
        {
            let w_13 = self.weights.get("block1.3.weight")?;
            let b_13 = self.weights.get("block1.3.bias")?;
            (vt.conv3x3_s2)(
                &Conv3x3Args {
                    input: &self.buf_a[..h / 2 * (w / 2) * 8],
                    residual: None,
                    weights: w_13,
                    bias: b_13,
                    h_in: h / 2, w_in: w / 2, c_in: 8, c_out: 24,
                    activation: Activation::Relu,
                },
                &mut self.buf_b[..h4 * w4 * 24],
            );
        }

        // ── 3. Skip1 ────────────────────────────────────────────────────────
        // AvgPool(4×4) → (H/4, W/4, 1)
        avgpool_4x4_s4(&self.norm_gray, &mut self.skip1_pool, h, w, 1);
        // conv1x1(1→24, identity) → buf_a (H/4, W/4, 24)
        {
            let w_s1 = self.weights.get("skip1.weight")?;
            let b_s1 = self.weights.get("skip1.bias")?;
            (vt.conv1x1)(
                &Conv1x1Args {
                    input: &self.skip1_pool,
                    weights: w_s1,
                    bias: b_s1,
                    h: h4, w: w4, c_in: 1, c_out: 24,
                    activation: Activation::Identity,
                },
                &mut self.buf_a[..h4 * w4 * 24],
            );
        }

        // ── 4. x1 = block1_raw + skip1 ─────────────────────────────────────
        add_inplace(&mut self.buf_b[..h4 * w4 * 24], &self.buf_a[..h4 * w4 * 24]);

        // ── 5. Block 2 ──────────────────────────────────────────────────────
        {
            let w_20 = self.weights.get("block2.0.weight")?;
            let b_20 = self.weights.get("block2.0.bias")?;
            (vt.conv3x3)(
                &Conv3x3Args {
                    input: &self.buf_b[..h4 * w4 * 24],
                    residual: None,
                    weights: w_20,
                    bias: b_20,
                    h_in: h4, w_in: w4, c_in: 24, c_out: 24,
                    activation: Activation::Relu,
                },
                &mut self.buf_a[..h4 * w4 * 24],
            );
        }
        {
            let w_21 = self.weights.get("block2.1.weight")?;
            let b_21 = self.weights.get("block2.1.bias")?;
            (vt.conv3x3)(
                &Conv3x3Args {
                    input: &self.buf_a[..h4 * w4 * 24],
                    residual: None,
                    weights: w_21,
                    bias: b_21,
                    h_in: h4, w_in: w4, c_in: 24, c_out: 24,
                    activation: Activation::Relu,
                },
                &mut self.buf_b[..h4 * w4 * 24],
            );
        }

        // ── 6. Block 3 ──────────────────────────────────────────────────────
        {
            let w_30 = self.weights.get("block3.0.weight")?;
            let b_30 = self.weights.get("block3.0.bias")?;
            (vt.conv3x3_s2)(
                &Conv3x3Args {
                    input: &self.buf_b[..h4 * w4 * 24],
                    residual: None,
                    weights: w_30,
                    bias: b_30,
                    h_in: h4, w_in: w4, c_in: 24, c_out: 64,
                    activation: Activation::Relu,
                },
                &mut self.buf_a[..h8 * w8 * 64],
            );
        }
        {
            let w_31 = self.weights.get("block3.1.weight")?;
            let b_31 = self.weights.get("block3.1.bias")?;
            (vt.conv3x3)(
                &Conv3x3Args {
                    input: &self.buf_a[..h8 * w8 * 64],
                    residual: None,
                    weights: w_31,
                    bias: b_31,
                    h_in: h8, w_in: w8, c_in: 64, c_out: 64,
                    activation: Activation::Relu,
                },
                &mut self.buf_b[..h8 * w8 * 64],
            );
        }
        {
            let w_32 = self.weights.get("block3.2.weight")?;
            let b_32 = self.weights.get("block3.2.bias")?;
            (vt.conv3x3)(
                &Conv3x3Args {
                    input: &self.buf_b[..h8 * w8 * 64],
                    residual: None,
                    weights: w_32,
                    bias: b_32,
                    h_in: h8, w_in: w8, c_in: 64, c_out: 64,
                    activation: Activation::Relu,
                },
                &mut self.x3,
            );
        }

        // ── 7. Block 4 ──────────────────────────────────────────────────────
        {
            let w_40 = self.weights.get("block4.0.weight")?;
            let b_40 = self.weights.get("block4.0.bias")?;
            (vt.conv3x3_s2)(
                &Conv3x3Args {
                    input: &self.x3,
                    residual: None,
                    weights: w_40,
                    bias: b_40,
                    h_in: h8, w_in: w8, c_in: 64, c_out: 64,
                    activation: Activation::Relu,
                },
                &mut self.buf_a[..h16 * w16 * 64],
            );
        }
        {
            let w_41 = self.weights.get("block4.1.weight")?;
            let b_41 = self.weights.get("block4.1.bias")?;
            (vt.conv3x3)(
                &Conv3x3Args {
                    input: &self.buf_a[..h16 * w16 * 64],
                    residual: None,
                    weights: w_41,
                    bias: b_41,
                    h_in: h16, w_in: w16, c_in: 64, c_out: 64,
                    activation: Activation::Relu,
                },
                &mut self.buf_b[..h16 * w16 * 64],
            );
        }
        {
            let w_42 = self.weights.get("block4.2.weight")?;
            let b_42 = self.weights.get("block4.2.bias")?;
            (vt.conv3x3)(
                &Conv3x3Args {
                    input: &self.buf_b[..h16 * w16 * 64],
                    residual: None,
                    weights: w_42,
                    bias: b_42,
                    h_in: h16, w_in: w16, c_in: 64, c_out: 64,
                    activation: Activation::Relu,
                },
                &mut self.x4,
            );
        }

        // ── 8. Block 5 ──────────────────────────────────────────────────────
        {
            let w_50 = self.weights.get("block5.0.weight")?;
            let b_50 = self.weights.get("block5.0.bias")?;
            (vt.conv3x3_s2)(
                &Conv3x3Args {
                    input: &self.x4,
                    residual: None,
                    weights: w_50,
                    bias: b_50,
                    h_in: h16, w_in: w16, c_in: 64, c_out: 128,
                    activation: Activation::Relu,
                },
                &mut self.buf_a[..h32 * w32 * 128],
            );
        }
        {
            let w_51 = self.weights.get("block5.1.weight")?;
            let b_51 = self.weights.get("block5.1.bias")?;
            (vt.conv3x3)(
                &Conv3x3Args {
                    input: &self.buf_a[..h32 * w32 * 128],
                    residual: None,
                    weights: w_51,
                    bias: b_51,
                    h_in: h32, w_in: w32, c_in: 128, c_out: 128,
                    activation: Activation::Relu,
                },
                &mut self.buf_b[..h32 * w32 * 128],
            );
        }
        {
            let w_52 = self.weights.get("block5.2.weight")?;
            let b_52 = self.weights.get("block5.2.bias")?;
            (vt.conv3x3)(
                &Conv3x3Args {
                    input: &self.buf_b[..h32 * w32 * 128],
                    residual: None,
                    weights: w_52,
                    bias: b_52,
                    h_in: h32, w_in: w32, c_in: 128, c_out: 128,
                    activation: Activation::Relu,
                },
                &mut self.buf_a[..h32 * w32 * 128],
            );
        }
        // conv1x1(128→64, identity) — no BN, no ReLU
        {
            let w_53 = self.weights.get("block5.3.weight")?;
            let b_53 = self.weights.get("block5.3.bias")?;
            (vt.conv1x1)(
                &Conv1x1Args {
                    input: &self.buf_a[..h32 * w32 * 128],
                    weights: w_53,
                    bias: b_53,
                    h: h32, w: w32, c_in: 128, c_out: 64,
                    activation: Activation::Identity,
                },
                &mut self.buf_b[..h32 * w32 * 64],
            );
        }

        // ── 9. FPN ──────────────────────────────────────────────────────────
        // x4_up = bilinear_upsample(x4, H/8, W/8)
        bilinear_upsample(&self.x4, &mut self.x4_up, h16, w16, 64, h8, w8);
        // x5_up = bilinear_upsample(x5=buf_b, H/8, W/8)
        bilinear_upsample(&self.buf_b[..h32 * w32 * 64], &mut self.x5_up, h32, w32, 64, h8, w8);
        // x3 += x4_up + x5_up  (FPN element-wise sum, in-place on x3)
        add3_inplace(&mut self.x3, &self.x4_up, &self.x5_up);

        // ── 10. Block Fusion ─────────────────────────────────────────────────
        {
            let w_bf0 = self.weights.get("block_fusion.0.weight")?;
            let b_bf0 = self.weights.get("block_fusion.0.bias")?;
            (vt.conv3x3)(
                &Conv3x3Args {
                    input: &self.x3,
                    residual: None,
                    weights: w_bf0,
                    bias: b_bf0,
                    h_in: h8, w_in: w8, c_in: 64, c_out: 64,
                    activation: Activation::Relu,
                },
                &mut self.buf_a[..h8 * w8 * 64],
            );
        }
        {
            let w_bf1 = self.weights.get("block_fusion.1.weight")?;
            let b_bf1 = self.weights.get("block_fusion.1.bias")?;
            (vt.conv3x3)(
                &Conv3x3Args {
                    input: &self.buf_a[..h8 * w8 * 64],
                    residual: None,
                    weights: w_bf1,
                    bias: b_bf1,
                    h_in: h8, w_in: w8, c_in: 64, c_out: 64,
                    activation: Activation::Relu,
                },
                &mut self.buf_b[..h8 * w8 * 64],
            );
        }
        // conv1x1(64→64, identity) — no ReLU
        {
            let w_bf2 = self.weights.get("block_fusion.2.weight")?;
            let b_bf2 = self.weights.get("block_fusion.2.bias")?;
            (vt.conv1x1)(
                &Conv1x1Args {
                    input: &self.buf_b[..h8 * w8 * 64],
                    weights: w_bf2,
                    bias: b_bf2,
                    h: h8, w: w8, c_in: 64, c_out: 64,
                    activation: Activation::Identity,
                },
                &mut self.feats,
            );
        }

        // ── 11. Heatmap Head ─────────────────────────────────────────────────
        {
            let w_kh0 = self.weights.get("heatmap_head.0.weight")?;
            let b_kh0 = self.weights.get("heatmap_head.0.bias")?;
            (vt.conv3x3)(
                &Conv3x3Args {
                    input: &self.feats,
                    residual: None,
                    weights: w_kh0,
                    bias: b_kh0,
                    h_in: h8, w_in: w8, c_in: 64, c_out: 64,
                    activation: Activation::Relu,
                },
                &mut self.buf_a[..h8 * w8 * 64],
            );
        }
        {
            let w_kh1 = self.weights.get("heatmap_head.1.weight")?;
            let b_kh1 = self.weights.get("heatmap_head.1.bias")?;
            (vt.conv3x3)(
                &Conv3x3Args {
                    input: &self.buf_a[..h8 * w8 * 64],
                    residual: None,
                    weights: w_kh1,
                    bias: b_kh1,
                    h_in: h8, w_in: w8, c_in: 64, c_out: 64,
                    activation: Activation::Relu,
                },
                &mut self.buf_b[..h8 * w8 * 64],
            );
        }
        // conv1x1(64→65, identity) → k1_raw
        {
            let w_kh2 = self.weights.get("heatmap_head.2.weight")?;
            let b_kh2 = self.weights.get("heatmap_head.2.bias")?;
            (vt.conv1x1)(
                &Conv1x1Args {
                    input: &self.buf_b[..h8 * w8 * 64],
                    weights: w_kh2,
                    bias: b_kh2,
                    h: h8, w: w8, c_in: 64, c_out: 65,
                    activation: Activation::Identity,
                },
                &mut self.k1_raw,
            );
        }
        // softmax(65 channels) → drop dustbin → pixel_shuffle(8) → k1h (H, W)
        channel_softmax(&mut self.k1_raw, h8, w8, 65);
        drop_last_channel_nhwc(&self.k1_raw, &mut self.buf_a[..h8 * w8 * 64], h8, w8, 65, 64);
        pixel_shuffle_8(&self.buf_a[..h8 * w8 * 64], &mut self.k1h, h8, w8);

        // ── 12. Keypoint Head (descriptor) ───────────────────────────────────
        {
            let w_kp0 = self.weights.get("keypoint_head.0.weight")?;
            let b_kp0 = self.weights.get("keypoint_head.0.bias")?;
            (vt.conv3x3)(
                &Conv3x3Args {
                    input: &self.feats,
                    residual: None,
                    weights: w_kp0,
                    bias: b_kp0,
                    h_in: h8, w_in: w8, c_in: 64, c_out: 64,
                    activation: Activation::Relu,
                },
                &mut self.buf_a[..h8 * w8 * 64],
            );
        }
        {
            let w_kp1 = self.weights.get("keypoint_head.1.weight")?;
            let b_kp1 = self.weights.get("keypoint_head.1.bias")?;
            (vt.conv3x3)(
                &Conv3x3Args {
                    input: &self.buf_a[..h8 * w8 * 64],
                    residual: None,
                    weights: w_kp1,
                    bias: b_kp1,
                    h_in: h8, w_in: w8, c_in: 64, c_out: 64,
                    activation: Activation::Relu,
                },
                &mut self.buf_b[..h8 * w8 * 64],
            );
        }
        {
            let w_kp2 = self.weights.get("keypoint_head.2.weight")?;
            let b_kp2 = self.weights.get("keypoint_head.2.bias")?;
            (vt.conv1x1)(
                &Conv1x1Args {
                    input: &self.buf_b[..h8 * w8 * 64],
                    weights: w_kp2,
                    bias: b_kp2,
                    h: h8, w: w8, c_in: 64, c_out: 64,
                    activation: Activation::Identity,
                },
                &mut self.h1_desc,
            );
        }
        l2_normalize_channel(&mut self.h1_desc, h8, w8, 64);

        // ── 13. L2-normalise descriptor map (M1) ─────────────────────────────
        // Done after heads so both heads can read the un-normalised feats.
        l2_normalize_channel(&mut self.feats, h8, w8, 64);

        // ── 14. NMS + sparse sampling ─────────────────────────────────────────
        let rw = self.last_scale.rw;
        let rh = self.last_scale.rh;
        // Reliability is sampled from h1_desc at (kp.x/8, kp.y/8), but h1_desc
        // is 64-channel. We compress it to a single channel by taking the L1
        // sum (= sqrt(64) * mean |d_i| after L2 norm ≈ constant, so NMS score
        // mostly comes from the heatmap). The resulting map is (H/8*W/8, 1).
        //
        // This keeps the API contract of nms_topk while giving the exact same
        // keypoint ranking as heatmap-only NMS. Proper reliability scoring
        // (K1h × dot(H1_norm, feats_norm)) is a follow-up once parity is verified.
        let reliability_map: Vec<f32> =
            self.h1_desc.chunks(64).map(|ch| ch.iter().sum::<f32>()).collect();

        let kps = nms_topk(
            &self.k1h,
            &reliability_map,
            h,
            w,
            h8,
            w8,
            self.config.score_threshold,
            self.config.top_k,
            rw,
            rh,
        );

        // Sample descriptors (M1 = feats, already L2-normalised) at keypoints.
        let kps_desc_coords: Vec<(f32, f32)> = kps
            .iter()
            .map(|kp| (kp.x / rw / 8.0, kp.y / rh / 8.0))
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
            // Re-normalise after bicubic (interpolation can shift the norm).
            for chunk in self.descriptors.chunks_mut(64) {
                let norm = chunk.iter().map(|&x| x * x).sum::<f32>().sqrt();
                let inv = 1.0 / (norm + 1e-12);
                for x in chunk {
                    *x *= inv;
                }
            }
        }

        self.reliability_per_kp.resize(kps.len(), 0.0);
        for (i, kp) in kps.iter().enumerate() {
            self.reliability_per_kp[i] = kp.score;
        }
        self.keypoints = kps;

        Ok(crate::XFeatOutput {
            keypoints: &self.keypoints,
            descriptors: &self.descriptors,
            reliability: &self.reliability_per_kp,
        })
    }
}
