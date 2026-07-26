//! CUDA SIFT — scale-invariant feature detector, descriptor and matcher.
//!
//! # Attribution and licensing
//!
//! This module is written to reproduce **OpenCV's** SIFT bit-for-bit, and was
//! developed by reading OpenCV's implementation (`modules/features2d/src/
//! sift.dispatch.cpp` and `sift.simd.hpp`, plus `getGaussianKernelBitExact` in
//! `modules/imgproc/src/smooth.dispatch.cpp`) and reproducing its expression
//! trees, constants and rounding behaviour. It should be treated as a
//! derivative work of OpenCV, which is licensed under **Apache-2.0** — the same
//! licence as this crate. Retain this notice.
//!
//! On this platform OpenCV's scalar-math entry points are additionally
//! overridden by the bundled **carotene / Tegra HAL** (also Apache-2.0); the
//! reciprocal- and reciprocal-square-root-estimate step counts reproduced in
//! [`hal`] follow that implementation.
//!
//! [`detect::exp2f_device_src`] reproduces **glibc's** `exp2f`
//! (`sysdeps/ieee754/flt-32/e_exp2f.c`). The lookup table is derived
//! numerically rather than copied, but the three polynomial coefficients are
//! glibc's literal values; glibc is **LGPL-2.1-or-later**, which differs from
//! this crate's licence. If that is a concern for redistribution, replace them
//! with independently fitted coefficients — the surrounding algorithm is a
//! standard table-plus-polynomial exponential and is not specific to glibc.
//!
//! The SIFT algorithm itself is due to D. Lowe; the associated patent
//! (US 6,711,293, filed 1999) has expired, which is why OpenCV moved SIFT out
//! of its `nonfree` module and into the main tree in 4.4.0.
//!
//! # Numerical contract: bit equality with the reference implementation
//!
//! Unlike the rest of this crate, SIFT has no CPU twin in kornia-rs. The
//! contract is therefore **bit equality with the reference implementation**
//! (`assert_eq!` on descriptor bytes and keypoint fields), not a tolerance.
//!
//! Matching bit-for-bit means matching the reference's *expression trees*, not
//! just its formulas:
//!
//! * NVRTC runs with `--fmad=false` (the workspace default in
//!   [`kornia_tensor::CudaKernel`]), so a plain `a * b + c` rounds twice, like
//!   scalar C++ compiled without contraction. Every place where the reference's
//!   aarch64 build *does* contract — its vector multiply-add lowers to
//!   `vfmaq_f32`, a true single-rounding FMA — is written here as an explicit
//!   `__fmaf_rn`.
//! * The separable Gaussian mirrors the reference's row and symmetric-column
//!   vector filters:
//!
//!   ```text
//!   row: acc = s[0]*kx[0];              for k in 1..ksize:  acc = fma(s[k], kx[k], acc)
//!   col: acc = fma(s[0][i], ky[0], 0);  for k in 1..=ksize2: acc = fma(s[k][i] + s[-k][i], ky[k], acc)
//!   ```
//!
//!   Note the column pass sums the symmetric pair *before* the FMA — swapping
//!   that for two FMAs changes the result.
//! * Kernel coefficients come from [`kernels::gaussian_kernel_f32`]. The
//!   reference evaluates them in software floating point for cross-platform
//!   determinism; hardware `f64` was verified to produce identical `f32`
//!   coefficients, so we use `f64` and pin the result in tests.
//! * Coefficients are baked into the kernel source as `__int_as_float(0x..)`
//!   literals so no decimal round-trip can perturb them.
//!
//! ## Where the default build steps outside that contract
//!
//! Scale space, detection and orientation are bit-exact as shipped. The
//! **descriptor is not**: the default kernel accumulates its histogram with
//! shared-memory float atomics, so the summation order varies run to run and
//! the last bits of a descriptor with it (an order of magnitude faster than the
//! sequential form — see [`descriptor`]). Set `KORNIA_SIFT_DESC=exact` for the
//! bit-exact one-thread-per-keypoint kernel. `KORNIA_SIFT_FASTMATH=1`
//! additionally replaces the [`hal`] primitives with SFU approximations, which
//! moves angles and descriptors but leaves keypoint positions exact.
//!
//! # Version scope
//!
//! Verified against the reference version the `opencv` dev-dependency links.
//! The reference's SIFT sources are numerically unchanged across the three
//! most recent major/minor releases (their diffs are a vector-intrinsics API
//! migration, a descriptor scratch-buffer size fix, and comment typos), and two
//! builds with different accelerated backends and thread pools were measured to
//! produce bitwise-identical descriptors on the test image. Three of 128
//! keypoint *angles* differ by 1-2 ULP between those builds; that divergence
//! does not reach the descriptors.
//!
//! # Input convention
//!
//! The reference converts its input with a unit fixed-point scale, so an 8-bit
//! image becomes f32 values in `0..=255`. Device inputs here follow the same
//! convention: **f32 in `0..=255`**, not `0..=1`.

mod descriptor;
mod detect;
mod hal;
mod kernels;
mod matcher;
mod orientation;
mod plan;
mod pyramid;

pub use descriptor::{
    launch_sift_descriptor_cuda, DESCR_HIST_BINS, DESCR_LEN, DESCR_MAG_THR, DESCR_SCL_FCTR,
    DESCR_WIDTH, INT_DESCR_FCTR,
};
pub use detect::{
    decode_keypoints, extrema_threshold, launch_sift_find_extrema_cuda, SiftRawKeypoint, KP_STRIDE,
};
pub use hal::{exp_table_f32, vrecpe_table};
pub use kernels::gaussian_kernel_f32;
pub use matcher::{SiftMatcher, MATCH_STRIDE};
pub use orientation::{
    launch_sift_orientation_cuda, ORI_HIST_BINS, ORI_KP_STRIDE, ORI_PEAK_RATIO, ORI_RADIUS,
    ORI_SIG_FCTR,
};
pub use plan::{FirstOctave, SiftCuda, SiftFeatures, SiftKeypoint};
pub use pyramid::{
    launch_sift_blur_h_cuda, launch_sift_blur_h_tiled_cuda, launch_sift_blur_hv_cuda,
    launch_sift_blur_v_cuda, launch_sift_blur_v_dog_cuda, launch_sift_dog_cuda,
    launch_sift_downsample_nearest_cuda, launch_sift_upsample2x_cuda, TILE_P,
};

/// Errors raised by the CUDA SIFT pipeline.
#[derive(Debug, thiserror::Error)]
pub enum SiftCudaError {
    /// Kernel compilation or launch failure.
    #[error("CUDA error: {0}")]
    Cuda(String),

    /// A device slice is smaller than the geometry requires.
    #[error("device slice too small: got {got} elements, need {need}")]
    SliceTooSmall {
        /// Length of the slice that was passed in.
        got: usize,
        /// Length the kernel requires for the given geometry.
        need: usize,
    },

    /// Invalid launch geometry (zero dimension, or a block dim component of zero).
    #[error("invalid geometry: {0}")]
    Geometry(String),
}

impl From<cudarc::driver::DriverError> for SiftCudaError {
    fn from(e: cudarc::driver::DriverError) -> Self {
        SiftCudaError::Cuda(e.to_string())
    }
}

/// Blur assumed to be already present in the input image.
pub const SIFT_INIT_SIGMA: f32 = 0.5;

/// Width of the border in which extrema are not detected.
pub const SIFT_IMG_BORDER: i32 = 5;

/// Maximum sub-pixel interpolation steps during extremum refinement.
pub const SIFT_MAX_INTERP_STEPS: i32 = 5;

/// Configuration mirroring the reference detector's constructor parameters.
#[derive(Debug, Clone, Copy)]
pub struct SiftCudaConfig {
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
    /// Device capacity for detected keypoints.
    pub max_keypoints: usize,
}

impl Default for SiftCudaConfig {
    fn default() -> Self {
        Self {
            n_features: 0,
            n_octave_layers: 3,
            contrast_threshold: 0.04,
            edge_threshold: 10.0,
            sigma: 1.6,
            max_keypoints: 8192,
        }
    }
}

impl SiftCudaConfig {
    /// Per-layer incremental blur sigmas.
    ///
    /// `sig[0]` is the base sigma; `sig[i]` for `i >= 1` is the *incremental*
    /// blur that takes layer `i-1` to layer `i`, chosen so the total blur
    /// follows `sigma * k^i` with `k = 2^(1/n_octave_layers)`.
    pub fn layer_sigmas(&self) -> Vec<f64> {
        let n = self.n_octave_layers + 3;
        let mut sig = vec![0.0; n];
        sig[0] = self.sigma;
        let k = 2f64.powf(1.0 / self.n_octave_layers as f64);
        for (i, s) in sig.iter_mut().enumerate().skip(1) {
            // `powf`, not `powi`: the reference calls the generic
            // `pow(double, double)`, and repeated multiplication gives a
            // different last bit.
            let sig_prev = k.powf(i as f64 - 1.0) * self.sigma;
            let sig_total = sig_prev * k;
            *s = (sig_total * sig_total - sig_prev * sig_prev).sqrt();
        }
        sig
    }

    /// Blur applied to the base image of the first octave.
    ///
    /// `sqrt(max(sigma^2 - k*INIT^2, 0.01))` in plain `f32` with **two**
    /// roundings. The reference's build does NOT contract this into an FMA —
    /// verified by instrumenting it and reading back the raw bits
    /// (`0x3f9fdf39` = 1.2489997, not `0x3f9fdf38` = 1.2489996). One ULP here
    /// changes every coefficient of the base blur kernel and shifts all 48
    /// pyramid layers, so do not "optimise" this into a `mul_add`.
    ///
    /// `doubled` selects between the reference's two `createInitialImage`
    /// branches: doubling the image halves the effective input blur, so `k` is
    /// **4** there and **1** when starting at the native resolution. Using the
    /// doubled constant on the native path under-blurs every layer of the whole
    /// scale space (1.2490 instead of 1.5199 at the default sigma).
    pub fn base_sig_diff_for(&self, doubled: bool) -> f32 {
        let sigma = self.sigma as f32;
        // `0.25 * 4.0` and `0.25 * 1.0` are both exact, so this is the same two
        // roundings the pinned bit pattern was measured with.
        let init2 = SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * if doubled { 4.0 } else { 1.0 };
        (sigma * sigma - init2).max(0.01).sqrt()
    }

    /// Blur applied to the doubled base image — [`Self::base_sig_diff_for`] with
    /// `doubled = true`, the reference's default.
    pub fn base_sig_diff(&self) -> f32 {
        self.base_sig_diff_for(true)
    }

    /// Number of octaves for a base image whose smaller side is `base_min_dim`.
    ///
    /// `round(log2(min_dim) - 2) - first_octave`. `first_octave` is `-1` when the
    /// base image was doubled and `0` when it was not; passing the doubled value
    /// on the native path builds one octave too many.
    pub fn n_octaves_for(&self, base_min_dim: usize, first_octave: i32) -> usize {
        let v = (base_min_dim as f64).ln() / std::f64::consts::LN_2 - 2.0;
        (round_ties_even(v) as i32 - first_octave).max(1) as usize
    }

    /// [`Self::n_octaves_for`] with `first_octave = -1`, the reference's default.
    pub fn n_octaves(&self, base_min_dim: usize) -> usize {
        self.n_octaves_for(base_min_dim, -1)
    }
}

/// Round to nearest, ties to even — the rounding the reference's integer
/// rounding helper uses. Rust's `f64::round` rounds half away from zero and
/// would disagree on exact `.5` inputs.
pub(crate) fn round_ties_even(v: f64) -> f64 {
    v.round_ties_even()
}

/// Kernel length derived from sigma for float images: `round(sigma * 8 + 1) | 1`.
pub fn gaussian_ksize(sigma: f64) -> usize {
    let n = round_ties_even(sigma * 4.0 * 2.0 + 1.0) as i64;
    (n | 1) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_sigmas_match_reference_defaults() {
        let cfg = SiftCudaConfig::default();
        let sig = cfg.layer_sigmas();
        assert_eq!(sig.len(), 6);
        assert_eq!(sig[0], 1.6);
        let expected = [
            1.6f64,
            1.2262734984654078,
            1.5450077936447955,
            1.9465878414647133,
            2.4525469971672263,
            3.090015587289591,
        ];
        // Tolerance, not equality: Rust's `powf` and the reference's `pow` can
        // differ by 1 ULP in this f64 intermediate. That difference is absorbed
        // when the sigma is narrowed into f32 kernel coefficients, which are the
        // actual contract — those are pinned bitwise in
        // `kernels::tests::gaussian_kernel_matches_reference_bitwise` and
        // verified end-to-end by the bit-exact pyramid test.
        for (got, want) in sig.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-9, "got {got}, want {want}");
        }
    }

    /// The CPU and CUDA paths keep separate copies of these host numerics, and
    /// the pyramids are only comparable if the copies agree to the last bit.
    /// They did not: the CPU copy spelled the power as `powi`, which the CUDA
    /// copy's own comment already warned changes the result. Tolerance would
    /// have hidden it — the layer sigmas feed `gaussian_kernel_f32`, whose
    /// coefficients are pinned bitwise.
    #[test]
    fn layer_sigmas_agree_with_the_cpu_twin() {
        for n_octave_layers in [1usize, 3, 5, 8] {
            let gpu = SiftCudaConfig {
                n_octave_layers,
                ..Default::default()
            };
            let cpu = crate::features::SiftConfig {
                n_octave_layers,
                ..Default::default()
            };
            let (g, c) = (gpu.layer_sigmas(), cpu.layer_sigmas());
            assert_eq!(g.len(), c.len());
            for (i, (a, b)) in g.iter().zip(c.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "layer {i} at n_octave_layers={n_octave_layers}: gpu {a} != cpu {b}"
                );
            }
            for doubled in [true, false] {
                assert_eq!(
                    gpu.base_sig_diff_for(doubled).to_bits(),
                    cpu.base_sig_diff(doubled).to_bits(),
                    "base sig_diff (doubled={doubled})"
                );
            }
            for min_dim in [64usize, 195, 390, 1080] {
                for first_octave in [-1i32, 0] {
                    assert_eq!(
                        gpu.n_octaves_for(min_dim, first_octave),
                        cpu.n_octaves(min_dim, first_octave),
                        "n_octaves({min_dim}, {first_octave})"
                    );
                }
            }
        }
    }

    #[test]
    fn ksize_matches_reference() {
        assert_eq!(gaussian_ksize(1.2489995956420898), 11);
        assert_eq!(gaussian_ksize(1.2262734984654078), 11);
        assert_eq!(gaussian_ksize(1.5450077936447955), 13);
        assert_eq!(gaussian_ksize(1.9465878414647133), 17);
        assert_eq!(gaussian_ksize(2.4525469971672263), 21);
        assert_eq!(gaussian_ksize(3.090015587289591), 27);
    }

    #[test]
    fn base_sig_diff_matches_reference() {
        assert_eq!(
            SiftCudaConfig::default().base_sig_diff().to_bits(),
            0x3f9fdf39,
            "base sig_diff must match the FMA-contracted reference value"
        );
    }

    #[test]
    fn n_octaves_matches_reference() {
        // 258x195 input -> 516x390 base -> 8 octaves.
        assert_eq!(SiftCudaConfig::default().n_octaves(390), 8);
        // first_octave = 0 keeps the base at 195 and drops the extra octave the
        // doubled path gets from `- first_octave`.
        assert_eq!(SiftCudaConfig::default().n_octaves_for(195, 0), 6);
    }

    #[test]
    fn base_sig_diff_native_removes_one_init_sigma() {
        let cfg = SiftCudaConfig::default();
        // The doubled branch keeps the pinned reference value ...
        assert_eq!(cfg.base_sig_diff_for(true), cfg.base_sig_diff());
        // ... and the native branch removes INIT^2 rather than 4*INIT^2.
        let want = (1.6f32 * 1.6 - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA).sqrt();
        assert_eq!(cfg.base_sig_diff_for(false).to_bits(), want.to_bits());
    }
}
