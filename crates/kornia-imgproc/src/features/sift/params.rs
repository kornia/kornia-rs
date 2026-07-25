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
            let prev = self.sigma * k.powi(i as i32 - 1);
            let total = prev * k;
            // Contracted in the reference's build; the two-rounding form shifts
            // every coefficient of the resulting kernel.
            *slot = (total * total - prev * prev).sqrt();
        }
        out
    }

    /// Octaves built for a base image whose smaller side is `base_min_dim`.
    pub fn n_octaves(&self, base_min_dim: usize, first_octave: i32) -> usize {
        let v = (base_min_dim as f64).ln() / std::f64::consts::LN_2 - 2.0;
        (round_ties_even(v) as i32 - first_octave).max(1) as usize
    }
}
