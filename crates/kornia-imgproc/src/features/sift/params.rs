//! Scale-space parameters and Gaussian kernel generation, shared by the CPU and
//! CUDA SIFT paths.
//!
//! These are host-side numerics with no backend dependency, and both paths must
//! agree on them exactly: a one-ULP difference in a single coefficient shifts
//! every layer of the pyramid and therefore every keypoint.

/// Round half to even, matching the reference's `cvRound`.
fn round_ties_even(v: f64) -> f64 {
    let r = v.round();
    if (v - v.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - v.signum()
    } else {
        r
    }
}

/// Tap count for a given sigma, matching the reference's
/// `cvRound(sigma * 4 * 2 + 1) | 1`.
pub fn gaussian_ksize(sigma: f64) -> usize {
    let n = round_ties_even(sigma * 4.0 * 2.0 + 1.0) as i64;
    (n | 1) as usize
}

/// Gaussian kernel generation matching the reference implementation exactly.
///
/// The reference evaluates this in software floating point for cross-platform
/// determinism. Hardware `f64` is used here instead: the result is narrowed to
/// `f32` at the end, which absorbs any sub-ULP disagreement, and the output was
/// verified bit-identical to the reference for every sigma the default
/// configuration uses.
///
/// The structure matters as much as the formula:
/// * `x` steps over *odd integers* `1-n, 3-n, ...` (doubled coordinates) with
///   `scale2x = -0.125 / sigma^2` instead of `-0.5 / sigma^2`, so `x*x` stays an
///   exact integer and only `exp` rounds.
/// * Only the half kernel is evaluated; the centre tap is exactly `1.0` before
///   normalisation and the other half is mirrored, so symmetry is exact rather
///   than approximate.
pub fn gaussian_kernel_f32(n: usize, sigma: f64) -> Vec<f32> {
    assert!(n > 0, "kernel size must be positive");
    assert!(sigma > 0.0, "only the sigma > 0 branch is used by SIFT");

    let scale2x = -0.125 / (sigma * sigma);
    let n2 = (n - 1) / 2;

    let mut values = Vec::with_capacity(n2);
    let mut sum = 0.0f64;
    let mut x = 1i64 - n as i64;
    for _ in 0..n2 {
        let t = ((x * x) as f64 * scale2x).exp();
        values.push(t);
        sum += t;
        x += 2;
    }
    sum = sum * 2.0 + 1.0;
    if n.is_multiple_of(2) {
        sum += 1.0;
    }

    let mul1 = 1.0 / sum;
    let mut result = vec![0.0f32; n];
    for (i, &v) in values.iter().enumerate() {
        let t = (v * mul1) as f32;
        result[i] = t;
        result[n - 1 - i] = t;
    }
    result[n2] = mul1 as f32;
    if n.is_multiple_of(2) {
        result[n2 + 1] = result[n2];
    }
    result
}

/// Reflect-101 index: `... 2 1 | 0 1 2 ... n-1 | n-2 n-3 ...` (edge not repeated).
#[inline]
pub fn refl101(mut i: i64, n: i64) -> usize {
    if n == 1 {
        return 0;
    }
    while i < 0 || i >= n {
        i = if i < 0 { -i } else { 2 * n - i - 2 };
    }
    i as usize
}

/// Why a configuration was rejected.
///
/// Both backends validate through [`SiftConfig::validate`] and
/// [`validate_source`], so which inputs are rejected — and the wording of the
/// complaint — is residency-independent by construction. Each backend wraps this
/// in its own error type rather than sharing one, but neither decides the rule
/// itself: they drifted when they did (`max_octaves = 0` errored on GPU and
/// silently clamped on CPU, and a non-positive sigma errored on GPU and panicked
/// on CPU).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SiftConfigError {
    /// A parameter is outside the range the pipeline supports.
    #[error("invalid SIFT configuration: {0}")]
    Invalid(String),

    /// The source buffer does not match the stated geometry.
    #[error("source has {got} elements, need {need} for {width}x{height}")]
    SourceLen {
        /// Length of the buffer passed in.
        got: usize,
        /// Length the geometry requires.
        need: usize,
        /// Stated width.
        width: usize,
        /// Stated height.
        height: usize,
    },
}

/// Check a source buffer against its stated geometry.
pub fn validate_source(len: usize, width: usize, height: usize) -> Result<(), SiftConfigError> {
    if width == 0 || height == 0 {
        return Err(SiftConfigError::Invalid(
            "image dimensions must be non-zero".into(),
        ));
    }
    let need = width
        .checked_mul(height)
        .ok_or_else(|| SiftConfigError::Invalid("image dimensions overflow".into()))?;
    if len < need {
        return Err(SiftConfigError::SourceLen {
            got: len,
            need,
            width,
            height,
        });
    }
    Ok(())
}

/// Detector configuration, matching `cv::SIFT::create`'s defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SiftConfig {
    /// Maximum keypoints to retain; `0` means unlimited.
    pub n_features: usize,
    /// Layers per octave in which extrema are searched.
    pub n_octave_layers: usize,
    /// Contrast rejection threshold.
    pub contrast_threshold: f64,
    /// Edge rejection threshold (principal-curvature ratio).
    pub edge_threshold: f64,
    /// Blur of the base image of octave 0.
    pub sigma: f64,
}

impl Default for SiftConfig {
    fn default() -> Self {
        Self {
            n_features: 0,
            n_octave_layers: 3,
            contrast_threshold: 0.04,
            edge_threshold: 10.0,
            sigma: 1.6,
        }
    }
}

impl SiftConfig {
    /// Blur applied to the base image.
    ///
    /// `doubled` selects the `firstOctave = -1` form, which subtracts four times
    /// the assumed input blur because the image has been upsampled.
    pub fn base_sig_diff(&self, doubled: bool) -> f32 {
        const INIT_SIGMA: f32 = 0.5;
        let sigma = self.sigma as f32;
        let init2 = INIT_SIGMA * INIT_SIGMA * if doubled { 4.0 } else { 1.0 };
        (sigma * sigma - init2).max(0.01).sqrt()
    }

    /// Per-layer incremental blur sigmas. Layer 0 is the base and is not blurred
    /// again, so its entry is the absolute sigma rather than an increment.
    pub fn layer_sigmas(&self) -> Vec<f64> {
        let n = self.n_octave_layers + 3;
        let k = 2f64.powf(1.0 / self.n_octave_layers as f64);
        let mut out = vec![0.0; n];
        out[0] = self.sigma;
        for (i, slot) in out.iter_mut().enumerate().skip(1) {
            // `powf`, not `powi`: the reference calls the generic
            // `pow(double, double)`, and repeated multiplication gives a
            // different last bit. This must stay spelled exactly as
            // `SiftCudaConfig::layer_sigmas` spells it or the CPU and CUDA
            // pyramids diverge — see that function's comment.
            let prev = k.powf(i as f64 - 1.0) * self.sigma;
            let total = prev * k;
            // Contracted in the reference's build; the two-rounding form shifts
            // every coefficient of the resulting kernel.
            *slot = (total * total - prev * prev).sqrt();
        }
        out
    }

    /// Reject configurations neither backend can honour.
    ///
    /// The single place that decision is made; see [`SiftConfigError`].
    pub fn validate(&self, max_octaves: usize) -> Result<(), SiftConfigError> {
        if self.n_octave_layers == 0 {
            return Err(SiftConfigError::Invalid(
                "n_octave_layers must be non-zero".into(),
            ));
        }
        if !(self.sigma.is_finite() && self.sigma > 0.0) {
            return Err(SiftConfigError::Invalid(format!(
                "sigma must be finite and positive, got {}",
                self.sigma
            )));
        }
        if !self.contrast_threshold.is_finite() || !self.edge_threshold.is_finite() {
            return Err(SiftConfigError::Invalid(
                "contrast and edge thresholds must be finite".into(),
            ));
        }
        if max_octaves == 0 {
            return Err(SiftConfigError::Invalid(
                "max_octaves must be non-zero; use usize::MAX for unlimited".into(),
            ));
        }
        Ok(())
    }

    /// Octaves built for a base image whose smaller side is `base_min_dim`.
    pub fn n_octaves(&self, base_min_dim: usize, first_octave: i32) -> usize {
        let v = (base_min_dim as f64).ln() / std::f64::consts::LN_2 - 2.0;
        (round_ties_even(v) as i32 - first_octave).max(1) as usize
    }
}
