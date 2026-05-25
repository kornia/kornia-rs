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

use crate::ops::{
    Activation, Conv1x1Args, Conv3x3Args, OpsVtable,
    add_inplace, add3_inplace, avgpool_4x4_s4, bilinear_upsample, channel_softmax,
    drop_last_channel_nhwc, instance_norm_2d_singlech, l2_normalize_channel, pixel_shuffle_8,
    unfold_8x8,
};
use crate::postproc::{bicubic_sample_descriptors, nms_topk, KeyPoint};
use crate::preproc::InputScale;
use crate::weights::PackedWeights;
use crate::XFeatError;

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
        Self { height: 480, width: 640, score_threshold: 0.05, top_k: 4096 }
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
    norm_gray: Vec<f32>,   // (H, W)    InstanceNorm output; also unfold input
    skip1_pool: Vec<f32>,  // (H/4, W/4)  AvgPool output
    x3: Vec<f32>,          // (H/8, W/8, 64)  Block-3 output; FPN accumulator
    x4: Vec<f32>,          // (H/16, W/16, 64)  Block-4 output
    x4_up: Vec<f32>,       // (H/8, W/8, 64)   x4 bilinear-upsampled
    x5_up: Vec<f32>,       // (H/8, W/8, 64)   x5 bilinear-upsampled
    feats: Vec<f32>,       // (H/8, W/8, 64)   block_fusion output; descriptor map

    // ── Head outputs ──────────────────────────────────────────────────────
    h1_rel: Vec<f32>,  // (H/8, W/8)     heatmap_head 1-ch sigmoid reliability
    k1_raw: Vec<f32>,  // (H/8, W/8, 65) keypoint_head logits (before softmax)
    k1h: Vec<f32>,     // (H, W)         full-res keypoint heatmap after pixel_shuffle

    // ── Sparse output (pre-allocated to top_k) ────────────────────────────
    keypoints: Vec<KeyPoint>,
    descriptors: Vec<f32>,
    reliability_per_kp: Vec<f32>,

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

        Ok(Self {
            config: config.clone(),
            weights,
            vtable: OpsVtable::select(),

            buf_a: vec![0.0f32; seq_max],
            buf_b: vec![0.0f32; seq_max],

            norm_gray:  vec![0.0f32; h * w],
            skip1_pool: vec![0.0f32; h4 * w4],
            x3:    vec![0.0f32; h8 * w8 * 64],
            x4:    vec![0.0f32; h16 * w16 * 64],
            x4_up: vec![0.0f32; h8 * w8 * 64],
            x5_up: vec![0.0f32; h8 * w8 * 64],
            feats:     vec![0.0f32; h8 * w8 * 64],

            h1_rel: vec![0.0f32; h8 * w8],
            k1_raw: vec![0.0f32; h8 * w8 * 65],
            k1h:    vec![0.0f32; h * w],

            keypoints:         Vec::with_capacity(config.top_k),
            descriptors:       Vec::with_capacity(config.top_k * 64),
            reliability_per_kp: Vec::with_capacity(config.top_k),

            last_scale: InputScale { rw: 1.0, rh: 1.0, w_norm: w, h_norm: h },
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

        let (h4, w4)   = (h / 4,  w / 4);
        let (h8, w8)   = (h / 8,  w / 8);
        let (h16, w16) = (h / 16, w / 16);
        let (h32, w32) = (h / 32, w / 32);

        let vt = self.vtable;  // Copy — no ongoing borrow of self

        // ── 1. InstanceNorm2d ────────────────────────────────────────────
        instance_norm_2d_singlech(gray_f32, &mut self.norm_gray);

        // ── 2. Block 1 (four 3×3 BasicLayers) ───────────────────────────
        // step 1: conv3x3(1→4, relu)
        {
            let wt = self.weights.get("block1.0.weight")?;
            let bt = self.weights.get("block1.0.bias")?;
            (vt.conv3x3)(&Conv3x3Args {
                input: &self.norm_gray, residual: None,
                weights: wt, bias: bt,
                h_in: h, w_in: w, c_in: 1, c_out: 4,
                activation: Activation::Relu,
            }, &mut self.buf_a[..h * w * 4]);
        }
        // step 2: conv3x3(4→8, s=2, relu)
        {
            let wt = self.weights.get("block1.1.weight")?;
            let bt = self.weights.get("block1.1.bias")?;
            (vt.conv3x3_s2)(&Conv3x3Args {
                input: &self.buf_a[..h * w * 4], residual: None,
                weights: wt, bias: bt,
                h_in: h, w_in: w, c_in: 4, c_out: 8,
                activation: Activation::Relu,
            }, &mut self.buf_b[..h / 2 * (w / 2) * 8]);
        }
        // step 3: conv3x3(8→8, relu)
        {
            let wt = self.weights.get("block1.2.weight")?;
            let bt = self.weights.get("block1.2.bias")?;
            (vt.conv3x3)(&Conv3x3Args {
                input: &self.buf_b[..h / 2 * (w / 2) * 8], residual: None,
                weights: wt, bias: bt,
                h_in: h / 2, w_in: w / 2, c_in: 8, c_out: 8,
                activation: Activation::Relu,
            }, &mut self.buf_a[..h / 2 * (w / 2) * 8]);
        }
        // step 4: conv3x3(8→24, s=2, relu) → buf_b = x1_raw (H/4,W/4,24)
        {
            let wt = self.weights.get("block1.3.weight")?;
            let bt = self.weights.get("block1.3.bias")?;
            (vt.conv3x3_s2)(&Conv3x3Args {
                input: &self.buf_a[..h / 2 * (w / 2) * 8], residual: None,
                weights: wt, bias: bt,
                h_in: h / 2, w_in: w / 2, c_in: 8, c_out: 24,
                activation: Activation::Relu,
            }, &mut self.buf_b[..h4 * w4 * 24]);
        }

        // ── 3. Skip1 (AvgPool + conv1x1 with real bias) ──────────────────
        avgpool_4x4_s4(&self.norm_gray, &mut self.skip1_pool, h, w, 1);
        {
            let wt = self.weights.get("skip1.weight")?;
            let bt = self.weights.get("skip1.bias")?;
            (vt.conv1x1)(&Conv1x1Args {
                input: &self.skip1_pool,
                weights: wt, bias: bt,
                h: h4, w: w4, c_in: 1, c_out: 24,
                activation: Activation::Identity,
            }, &mut self.buf_a[..h4 * w4 * 24]);
        }

        // ── 4. x1 = x1_raw + skip1 ───────────────────────────────────────
        // buf_b holds x1_raw, buf_a holds skip1; add buf_a into buf_b.
        add_inplace(&mut self.buf_b[..h4 * w4 * 24], &self.buf_a[..h4 * w4 * 24]);

        // ── 5. Block 2 (two 3×3 BasicLayers) ─────────────────────────────
        {
            let wt = self.weights.get("block2.0.weight")?;
            let bt = self.weights.get("block2.0.bias")?;
            (vt.conv3x3)(&Conv3x3Args {
                input: &self.buf_b[..h4 * w4 * 24], residual: None,
                weights: wt, bias: bt,
                h_in: h4, w_in: w4, c_in: 24, c_out: 24,
                activation: Activation::Relu,
            }, &mut self.buf_a[..h4 * w4 * 24]);
        }
        {
            let wt = self.weights.get("block2.1.weight")?;
            let bt = self.weights.get("block2.1.bias")?;
            (vt.conv3x3)(&Conv3x3Args {
                input: &self.buf_a[..h4 * w4 * 24], residual: None,
                weights: wt, bias: bt,
                h_in: h4, w_in: w4, c_in: 24, c_out: 24,
                activation: Activation::Relu,
            }, &mut self.buf_b[..h4 * w4 * 24]);
        }

        // ── 6. Block 3 (two 3×3 + one 1×1 BasicLayer) ───────────────────
        {
            let wt = self.weights.get("block3.0.weight")?;
            let bt = self.weights.get("block3.0.bias")?;
            (vt.conv3x3_s2)(&Conv3x3Args {
                input: &self.buf_b[..h4 * w4 * 24], residual: None,
                weights: wt, bias: bt,
                h_in: h4, w_in: w4, c_in: 24, c_out: 64,
                activation: Activation::Relu,
            }, &mut self.buf_a[..h8 * w8 * 64]);
        }
        {
            let wt = self.weights.get("block3.1.weight")?;
            let bt = self.weights.get("block3.1.bias")?;
            (vt.conv3x3)(&Conv3x3Args {
                input: &self.buf_a[..h8 * w8 * 64], residual: None,
                weights: wt, bias: bt,
                h_in: h8, w_in: w8, c_in: 64, c_out: 64,
                activation: Activation::Relu,
            }, &mut self.buf_b[..h8 * w8 * 64]);
        }
        // block3.2: 1×1 BasicLayer (BN-folded, relu) → x3
        {
            let wt = self.weights.get("block3.2.weight")?;
            let bt = self.weights.get("block3.2.bias")?;
            (vt.conv1x1)(&Conv1x1Args {
                input: &self.buf_b[..h8 * w8 * 64],
                weights: wt, bias: bt,
                h: h8, w: w8, c_in: 64, c_out: 64,
                activation: Activation::Relu,
            }, &mut self.x3);
        }

        // ── 7. Block 4 (three 3×3 BasicLayers) ───────────────────────────
        {
            let wt = self.weights.get("block4.0.weight")?;
            let bt = self.weights.get("block4.0.bias")?;
            (vt.conv3x3_s2)(&Conv3x3Args {
                input: &self.x3, residual: None,
                weights: wt, bias: bt,
                h_in: h8, w_in: w8, c_in: 64, c_out: 64,
                activation: Activation::Relu,
            }, &mut self.buf_a[..h16 * w16 * 64]);
        }
        {
            let wt = self.weights.get("block4.1.weight")?;
            let bt = self.weights.get("block4.1.bias")?;
            (vt.conv3x3)(&Conv3x3Args {
                input: &self.buf_a[..h16 * w16 * 64], residual: None,
                weights: wt, bias: bt,
                h_in: h16, w_in: w16, c_in: 64, c_out: 64,
                activation: Activation::Relu,
            }, &mut self.buf_b[..h16 * w16 * 64]);
        }
        {
            let wt = self.weights.get("block4.2.weight")?;
            let bt = self.weights.get("block4.2.bias")?;
            (vt.conv3x3)(&Conv3x3Args {
                input: &self.buf_b[..h16 * w16 * 64], residual: None,
                weights: wt, bias: bt,
                h_in: h16, w_in: w16, c_in: 64, c_out: 64,
                activation: Activation::Relu,
            }, &mut self.x4);
        }

        // ── 8. Block 5 (three 3×3 + one 1×1 BasicLayer) ─────────────────
        {
            let wt = self.weights.get("block5.0.weight")?;
            let bt = self.weights.get("block5.0.bias")?;
            (vt.conv3x3_s2)(&Conv3x3Args {
                input: &self.x4, residual: None,
                weights: wt, bias: bt,
                h_in: h16, w_in: w16, c_in: 64, c_out: 128,
                activation: Activation::Relu,
            }, &mut self.buf_a[..h32 * w32 * 128]);
        }
        {
            let wt = self.weights.get("block5.1.weight")?;
            let bt = self.weights.get("block5.1.bias")?;
            (vt.conv3x3)(&Conv3x3Args {
                input: &self.buf_a[..h32 * w32 * 128], residual: None,
                weights: wt, bias: bt,
                h_in: h32, w_in: w32, c_in: 128, c_out: 128,
                activation: Activation::Relu,
            }, &mut self.buf_b[..h32 * w32 * 128]);
        }
        {
            let wt = self.weights.get("block5.2.weight")?;
            let bt = self.weights.get("block5.2.bias")?;
            (vt.conv3x3)(&Conv3x3Args {
                input: &self.buf_b[..h32 * w32 * 128], residual: None,
                weights: wt, bias: bt,
                h_in: h32, w_in: w32, c_in: 128, c_out: 128,
                activation: Activation::Relu,
            }, &mut self.buf_a[..h32 * w32 * 128]);
        }
        // block5.3: 1×1 BasicLayer (BN-folded, relu) → buf_b = x5 (H/32,W/32,64)
        {
            let wt = self.weights.get("block5.3.weight")?;
            let bt = self.weights.get("block5.3.bias")?;
            (vt.conv1x1)(&Conv1x1Args {
                input: &self.buf_a[..h32 * w32 * 128],
                weights: wt, bias: bt,
                h: h32, w: w32, c_in: 128, c_out: 64,
                activation: Activation::Relu,
            }, &mut self.buf_b[..h32 * w32 * 64]);
        }

        // ── 9. FPN: upsample x4, x5 to H/8×W/8, sum into x3 ────────────
        bilinear_upsample(&self.x4,                         &mut self.x4_up, h16, w16, 64, h8, w8);
        bilinear_upsample(&self.buf_b[..h32 * w32 * 64],   &mut self.x5_up, h32, w32, 64, h8, w8);
        add3_inplace(&mut self.x3, &self.x4_up, &self.x5_up);

        // ── 10. block_fusion (two 3×3 BasicLayers + one 1×1 plain conv) ──
        {
            let wt = self.weights.get("block_fusion.0.weight")?;
            let bt = self.weights.get("block_fusion.0.bias")?;
            (vt.conv3x3)(&Conv3x3Args {
                input: &self.x3, residual: None,
                weights: wt, bias: bt,
                h_in: h8, w_in: w8, c_in: 64, c_out: 64,
                activation: Activation::Relu,
            }, &mut self.buf_a[..h8 * w8 * 64]);
        }
        {
            let wt = self.weights.get("block_fusion.1.weight")?;
            let bt = self.weights.get("block_fusion.1.bias")?;
            (vt.conv3x3)(&Conv3x3Args {
                input: &self.buf_a[..h8 * w8 * 64], residual: None,
                weights: wt, bias: bt,
                h_in: h8, w_in: w8, c_in: 64, c_out: 64,
                activation: Activation::Relu,
            }, &mut self.buf_b[..h8 * w8 * 64]);
        }
        // block_fusion.2: plain conv1x1 with real bias, no activation
        {
            let wt = self.weights.get("block_fusion.2.weight")?;
            let bt = self.weights.get("block_fusion.2.bias")?;
            (vt.conv1x1)(&Conv1x1Args {
                input: &self.buf_b[..h8 * w8 * 64],
                weights: wt, bias: bt,
                h: h8, w: w8, c_in: 64, c_out: 64,
                activation: Activation::Identity,
            }, &mut self.feats);
        }

        // ── 11. Heatmap head (2 × 1×1 BasicLayer + conv1x1(64→1)+sigmoid) ─
        // Outputs single-channel reliability map h1_rel.
        {
            let wt = self.weights.get("heatmap_head.0.weight")?;
            let bt = self.weights.get("heatmap_head.0.bias")?;
            (vt.conv1x1)(&Conv1x1Args {
                input: &self.feats,
                weights: wt, bias: bt,
                h: h8, w: w8, c_in: 64, c_out: 64,
                activation: Activation::Relu,
            }, &mut self.buf_a[..h8 * w8 * 64]);
        }
        {
            let wt = self.weights.get("heatmap_head.1.weight")?;
            let bt = self.weights.get("heatmap_head.1.bias")?;
            (vt.conv1x1)(&Conv1x1Args {
                input: &self.buf_a[..h8 * w8 * 64],
                weights: wt, bias: bt,
                h: h8, w: w8, c_in: 64, c_out: 64,
                activation: Activation::Relu,
            }, &mut self.buf_b[..h8 * w8 * 64]);
        }
        // heatmap_head.2: conv1x1(64→1) + Sigmoid → h1_rel (H/8,W/8)
        {
            let wt = self.weights.get("heatmap_head.2.weight")?;
            let bt = self.weights.get("heatmap_head.2.bias")?;
            (vt.conv1x1)(&Conv1x1Args {
                input: &self.buf_b[..h8 * w8 * 64],
                weights: wt, bias: bt,
                h: h8, w: w8, c_in: 64, c_out: 1,
                activation: Activation::Sigmoid,
            }, &mut self.h1_rel);
        }

        // ── 12. Keypoint head (input = unfold_8x8(norm_gray)) ────────────
        // Takes the 8×8 patches of the InstanceNorm'd gray image at H/8 × W/8
        // grid positions (64 channels), then processes with 1×1 layers.
        unfold_8x8(&self.norm_gray, &mut self.buf_a[..h8 * w8 * 64], h, w);
        {
            let wt = self.weights.get("keypoint_head.0.weight")?;
            let bt = self.weights.get("keypoint_head.0.bias")?;
            (vt.conv1x1)(&Conv1x1Args {
                input: &self.buf_a[..h8 * w8 * 64],
                weights: wt, bias: bt,
                h: h8, w: w8, c_in: 64, c_out: 64,
                activation: Activation::Relu,
            }, &mut self.buf_b[..h8 * w8 * 64]);
        }
        {
            let wt = self.weights.get("keypoint_head.1.weight")?;
            let bt = self.weights.get("keypoint_head.1.bias")?;
            (vt.conv1x1)(&Conv1x1Args {
                input: &self.buf_b[..h8 * w8 * 64],
                weights: wt, bias: bt,
                h: h8, w: w8, c_in: 64, c_out: 64,
                activation: Activation::Relu,
            }, &mut self.buf_a[..h8 * w8 * 64]);
        }
        {
            let wt = self.weights.get("keypoint_head.2.weight")?;
            let bt = self.weights.get("keypoint_head.2.bias")?;
            (vt.conv1x1)(&Conv1x1Args {
                input: &self.buf_a[..h8 * w8 * 64],
                weights: wt, bias: bt,
                h: h8, w: w8, c_in: 64, c_out: 64,
                activation: Activation::Relu,
            }, &mut self.buf_b[..h8 * w8 * 64]);
        }
        // keypoint_head.3: plain conv1x1(64→65) with real bias → k1_raw
        {
            let wt = self.weights.get("keypoint_head.3.weight")?;
            let bt = self.weights.get("keypoint_head.3.bias")?;
            (vt.conv1x1)(&Conv1x1Args {
                input: &self.buf_b[..h8 * w8 * 64],
                weights: wt, bias: bt,
                h: h8, w: w8, c_in: 64, c_out: 65,
                activation: Activation::Identity,
            }, &mut self.k1_raw);
        }
        // softmax(65 ch) → drop dustbin → pixel_shuffle(8) → K1h (H, W)
        channel_softmax(&mut self.k1_raw, h8, w8, 65);
        drop_last_channel_nhwc(&self.k1_raw, &mut self.buf_a[..h8 * w8 * 64], h8, w8, 65, 64);
        pixel_shuffle_8(&self.buf_a[..h8 * w8 * 64], &mut self.k1h, h8, w8);

        // ── 13. L2-normalise descriptor map (M1 = feats) ─────────────────
        l2_normalize_channel(&mut self.feats, h8, w8, 64);

        // ── 14. NMS + sparse descriptor sampling ─────────────────────────
        let rw = self.last_scale.rw;
        let rh = self.last_scale.rh;

        // Reliability is the heatmap_head sigmoid output sampled at each
        // keypoint position (in H/8×W/8 coords). h1_rel is already a flat
        // (H/8 × W/8) single-channel map ready for nms_topk.
        let kps = nms_topk(
            &self.k1h,
            &self.h1_rel,
            h, w, h8, w8,
            self.config.score_threshold,
            self.config.top_k,
            rw, rh,
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
            bicubic_sample_descriptors(&self.feats, h8, w8, 64, &kps_desc_coords,
                                       &mut self.descriptors);
            // Re-normalise after bicubic interpolation shifts the norm.
            for chunk in self.descriptors.chunks_mut(64) {
                let norm = chunk.iter().map(|&x| x * x).sum::<f32>().sqrt();
                let inv = 1.0 / (norm + 1e-12);
                for x in chunk { *x *= inv; }
            }
        }

        self.reliability_per_kp.resize(kps.len(), 0.0);
        for (i, kp) in kps.iter().enumerate() {
            self.reliability_per_kp[i] = kp.score;
        }
        self.keypoints = kps;

        Ok(crate::XFeatOutput {
            keypoints:   &self.keypoints,
            descriptors: &self.descriptors,
            reliability: &self.reliability_per_kp,
        })
    }
}

impl XFeat {
    /// Full-resolution keypoint heatmap (H × W) from the last `extract` call.
    #[doc(hidden)]
    pub fn k1h_slice(&self) -> &[f32] { &self.k1h }
    /// Half-resolution reliability map (H/8 × W/8) from the last `extract` call.
    #[doc(hidden)]
    pub fn h1_rel_slice(&self) -> &[f32] { &self.h1_rel }
}
