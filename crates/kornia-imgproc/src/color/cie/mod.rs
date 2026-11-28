//! CIE perceptual color conversions: sRGB↔linear-RGB, RGB↔XYZ, RGB↔Lab, RGB↔Luv.
//!
//! Mirrors the HSV module's 4-layer dispatch (public `Image` entry → sealed-trait
//! pixel dispatch → strip split → NEON/scalar leaf). `f32` runs the fused NEON
//! kernels (single deinterleave/interleave per pixel-quad, all stages in-register);
//! `f64` runs the portable scalar oracle using std `powf`/`cbrt` of the exact
//! formulas. AVX2 currently falls through to scalar (specialized in the x86 pass).
//!
//! Channel convention (RGB in `[0, 1]`, matching OpenCV's f32 path):
//! linear-RGB `[0, 1]`; XYZ tristimulus; Lab `L∈[0,100], a,b∈~[-128,127]`;
//! Luv `L∈[0,100]`.

use crate::parallel;
use kornia_image::{Image, ImageError};

mod kernels;
mod nonlinear;
mod transfer;

use crate::color::kernel_common::{check_size, sealed};

// ===== trait + impl generation ======================================================
//
// Each conversion is a sealed trait with an f32 (SIMD) and f64 (scalar oracle) impl.
// `cie_conv!` stamps out the trait, both impls, and the public dispatch fn.

macro_rules! cie_conv {
    ($trait:ident, $method:ident, $pub_fn:ident, $f32_kernel:path, $f64_oracle:path, $doc:expr_2021) => {
        #[doc = $doc]
        pub trait $trait: sealed::Sealed + Sized {
            #[doc(hidden)]
            fn $method(src: &Image<Self, 3>, dst: &mut Image<Self, 3>) -> Result<(), ImageError>;
        }

        impl $trait for f32 {
            fn $method(src: &Image<f32, 3>, dst: &mut Image<f32, 3>) -> Result<(), ImageError> {
                check_size(src, dst)?;
                $f32_kernel(src.as_slice(), dst.as_slice_mut(), src.rows() * src.cols());
                Ok(())
            }
        }

        impl $trait for f64 {
            fn $method(src: &Image<f64, 3>, dst: &mut Image<f64, 3>) -> Result<(), ImageError> {
                check_size(src, dst)?;
                parallel::par_iter_rows(src, dst, |s, d| {
                    let (a, b, c) = $f64_oracle(s[0], s[1], s[2]);
                    d[0] = a;
                    d[1] = b;
                    d[2] = c;
                });
                Ok(())
            }
        }

        #[doc = $doc]
        pub fn $pub_fn<T>(src: &Image<T, 3>, dst: &mut Image<T, 3>) -> Result<(), ImageError>
        where
            T: $trait,
        {
            T::$method(src, dst)
        }
    };
}

cie_conv!(
    LinearRgbFromRgb,
    linear_rgb_from_rgb_impl,
    linear_rgb_from_rgb,
    kernels::linear_rgb_from_rgb_f32,
    kernels::linear_from_srgb_scalar64_px,
    "Convert sRGB (gamma-encoded) to linear RGB. Channels in `[0, 1]`."
);
cie_conv!(
    RgbFromLinearRgb,
    rgb_from_linear_rgb_impl,
    rgb_from_linear_rgb,
    kernels::rgb_from_linear_rgb_f32,
    kernels::srgb_from_linear_scalar64_px,
    "Convert linear RGB to sRGB (gamma-encoded). Inverse of [`linear_rgb_from_rgb`]."
);
cie_conv!(
    XyzFromRgb,
    xyz_from_rgb_impl,
    xyz_from_rgb,
    kernels::xyz_from_rgb_f32,
    kernels::xyz_from_rgb_scalar64,
    "Convert sRGB to CIE XYZ (D65). Gamma-decodes then applies the RGB→XYZ matrix."
);
cie_conv!(
    RgbFromXyz,
    rgb_from_xyz_impl,
    rgb_from_xyz,
    kernels::rgb_from_xyz_f32,
    kernels::rgb_from_xyz_scalar64,
    "Convert CIE XYZ (D65) to sRGB. Inverse of [`xyz_from_rgb`]."
);
cie_conv!(
    LabFromRgb,
    lab_from_rgb_impl,
    lab_from_rgb,
    kernels::lab_from_rgb_f32,
    kernels::lab_from_rgb_scalar64,
    "Convert sRGB to CIE L*a*b* (D65). `L∈[0,100]`, `a,b∈~[-128,127]`."
);
cie_conv!(
    RgbFromLab,
    rgb_from_lab_impl,
    rgb_from_lab,
    kernels::rgb_from_lab_f32,
    kernels::rgb_from_lab_scalar64,
    "Convert CIE L*a*b* (D65) to sRGB. Inverse of [`lab_from_rgb`]."
);
cie_conv!(
    LuvFromRgb,
    luv_from_rgb_impl,
    luv_from_rgb,
    kernels::luv_from_rgb_f32,
    kernels::luv_from_rgb_scalar64,
    "Convert sRGB to CIE L*u*v* (D65). `L∈[0,100]`."
);
cie_conv!(
    RgbFromLuv,
    rgb_from_luv_impl,
    rgb_from_luv,
    kernels::rgb_from_luv_f32,
    kernels::rgb_from_luv_scalar64,
    "Convert CIE L*u*v* (D65) to sRGB. Inverse of [`luv_from_rgb`]."
);

// ===== thin f32 wrappers ============================================================

macro_rules! f32_wrapper {
    ($name:ident, $generic:ident, $doc:expr_2021) => {
        #[doc = $doc]
        pub fn $name(src: &Image<f32, 3>, dst: &mut Image<f32, 3>) -> Result<(), ImageError> {
            $generic(src, dst)
        }
    };
}

f32_wrapper!(
    linear_rgb_from_rgb_f32,
    linear_rgb_from_rgb,
    "f32 [`linear_rgb_from_rgb`]."
);
f32_wrapper!(
    rgb_from_linear_rgb_f32,
    rgb_from_linear_rgb,
    "f32 [`rgb_from_linear_rgb`]."
);
f32_wrapper!(xyz_from_rgb_f32, xyz_from_rgb, "f32 [`xyz_from_rgb`].");
f32_wrapper!(rgb_from_xyz_f32, rgb_from_xyz, "f32 [`rgb_from_xyz`].");
f32_wrapper!(lab_from_rgb_f32, lab_from_rgb, "f32 [`lab_from_rgb`].");
f32_wrapper!(rgb_from_lab_f32, rgb_from_lab, "f32 [`rgb_from_lab`].");
f32_wrapper!(luv_from_rgb_f32, luv_from_rgb, "f32 [`luv_from_rgb`].");
f32_wrapper!(rgb_from_luv_f32, rgb_from_luv, "f32 [`rgb_from_luv`].");

#[cfg(test)]
mod tests {
    use kornia_image::{Image, ImageError, ImageSize};

    // Build an f32 RGB image and its f64 twin from the same [0,1] samples.
    fn pair(w: usize, h: usize) -> (Vec<f32>, Vec<f64>) {
        let n = w * h * 3;
        let f32v: Vec<f32> = (0..n).map(|v| ((v * 7 % 251) as f32) / 250.0).collect();
        let f64v: Vec<f64> = f32v.iter().map(|&x| x as f64).collect();
        (f32v, f64v)
    }

    // Conversion fn-pointer aliases (keep clippy::type_complexity quiet in test helpers).
    type ConvFn<T> = fn(&Image<T, 3>, &mut Image<T, 3>) -> Result<(), ImageError>;

    // Generic: f32-SIMD vs f64-scalar for one forward conversion.
    fn check_simd_vs_scalar(
        w: usize,
        h: usize,
        fwd_f32: ConvFn<f32>,
        fwd_f64: ConvFn<f64>,
        tol: [f64; 3],
    ) -> Result<(), ImageError> {
        let (f32v, f64v) = pair(w, h);
        let size = ImageSize {
            width: w,
            height: h,
        };
        let src32 = Image::<f32, 3>::new(size, f32v)?;
        let src64 = Image::<f64, 3>::new(size, f64v)?;
        let mut out32 = Image::<f32, 3>::from_size_val(size, 0.0)?;
        let mut out64 = Image::<f64, 3>::from_size_val(size, 0.0)?;
        fwd_f32(&src32, &mut out32)?;
        fwd_f64(&src64, &mut out64)?;
        for (i, (a, b)) in out32
            .as_slice()
            .iter()
            .zip(out64.as_slice().iter())
            .enumerate()
        {
            let t = tol[i % 3];
            assert!(
                (*a as f64 - b).abs() <= t,
                "ch{} {a} vs {b} (tol {t})",
                i % 3
            );
        }
        Ok(())
    }

    // Generic round-trip: RGB → space → RGB.
    fn check_round_trip(
        w: usize,
        h: usize,
        fwd: ConvFn<f32>,
        rev: ConvFn<f32>,
        tol: f32,
        skip_near_zero: bool,
    ) -> Result<(), ImageError> {
        let (f32v, _) = pair(w, h);
        let size = ImageSize {
            width: w,
            height: h,
        };
        let src = Image::<f32, 3>::new(size, f32v)?;
        let mut mid = Image::<f32, 3>::from_size_val(size, 0.0)?;
        let mut back = Image::<f32, 3>::from_size_val(size, 0.0)?;
        fwd(&src, &mut mid)?;
        rev(&mid, &mut back)?;
        for (a, b) in src.as_slice().iter().zip(back.as_slice().iter()) {
            if skip_near_zero && *a < 0.02 {
                continue;
            }
            assert!((a - b).abs() <= tol, "round-trip {a} != {b}");
        }
        Ok(())
    }

    #[test]
    fn srgb_linear_simd_vs_scalar() -> Result<(), ImageError> {
        // Colour-grade pow (degree-4 log2 / degree-3 exp2) → ~4e-4 rel on the gamma;
        // deliberate speed/accuracy trade for the hot path (imperceptible, ≈0.04%).
        check_simd_vs_scalar(
            7,
            3,
            super::linear_rgb_from_rgb,
            super::linear_rgb_from_rgb,
            [5e-4; 3],
        )
    }
    #[test]
    fn xyz_simd_vs_scalar() -> Result<(), ImageError> {
        check_simd_vs_scalar(7, 3, super::xyz_from_rgb, super::xyz_from_rgb, [5e-4; 3])
    }
    #[test]
    fn lab_simd_vs_scalar() -> Result<(), ImageError> {
        check_simd_vs_scalar(
            7,
            3,
            super::lab_from_rgb,
            super::lab_from_rgb,
            [1e-2, 2e-2, 2e-2],
        )
    }
    #[test]
    fn luv_simd_vs_scalar() -> Result<(), ImageError> {
        check_simd_vs_scalar(
            7,
            3,
            super::luv_from_rgb,
            super::luv_from_rgb,
            [1e-2, 5e-2, 5e-2],
        )
    }

    #[test]
    fn srgb_round_trip() -> Result<(), ImageError> {
        check_round_trip(
            7,
            3,
            super::linear_rgb_from_rgb,
            super::rgb_from_linear_rgb,
            1e-3,
            false,
        )
    }
    #[test]
    fn xyz_round_trip() -> Result<(), ImageError> {
        check_round_trip(7, 3, super::xyz_from_rgb, super::rgb_from_xyz, 1e-3, false)
    }
    #[test]
    fn lab_round_trip() -> Result<(), ImageError> {
        check_round_trip(7, 3, super::lab_from_rgb, super::rgb_from_lab, 1e-3, false)
    }
    #[test]
    fn luv_round_trip() -> Result<(), ImageError> {
        // Luv is unstable near black (L→0 divides), so exclude near-zero inputs.
        check_round_trip(7, 3, super::luv_from_rgb, super::rgb_from_luv, 1e-3, true)
    }

    /// Neutral colours have zero Luv chroma, and zero chroma maps back to a
    /// neutral colour. This checks the white point directly: the f64 oracle
    /// shares `LUV_UN` / `LUV_VN`, so the SIMD-vs-scalar tests cannot see a
    /// wrong white (it gave u = -0.130, v = +0.041 for white).
    #[test]
    fn luv_neutral_has_zero_chroma() -> Result<(), ImageError> {
        // f32 bound per arch: x86_64 runs the scalar path (measured max 1.5e-4
        // over 1000 grays); aarch64's NEON path uses a one-step reciprocal
        // (~1e-4 relative), which allows ~13·L·u′·1e-4 ≈ 0.026 at L = 100.
        #[cfg(target_arch = "aarch64")]
        const F32_CHROMA_TOL: f32 = 3e-2;
        #[cfg(not(target_arch = "aarch64"))]
        const F32_CHROMA_TOL: f32 = 1e-3;
        #[cfg(target_arch = "aarch64")]
        const F32_NEUTRAL_TOL: f32 = 1e-3;
        #[cfg(not(target_arch = "aarch64"))]
        const F32_NEUTRAL_TOL: f32 = 1e-5;
        let grays = [0.02, 0.1, 0.25, 0.5, 0.75, 1.0];
        // 29 pixels: on aarch64 the NEON driver runs three 8-px blocks, one
        // 4-px block and a 1-px scalar tail (x86_64 is scalar throughout).
        let (w, n) = (29usize, grays.len());
        let size = ImageSize {
            width: w,
            height: 1,
        };
        let g64: Vec<f64> = (0..w).flat_map(|i| [grays[i % n]; 3]).collect();
        let g32: Vec<f32> = g64.iter().map(|&v| v as f32).collect();

        let mut out64 = Image::<f64, 3>::from_size_val(size, 0.0)?;
        super::luv_from_rgb(&Image::<f64, 3>::new(size, g64)?, &mut out64)?;
        let mut out32 = Image::<f32, 3>::from_size_val(size, 0.0)?;
        super::luv_from_rgb(&Image::<f32, 3>::new(size, g32)?, &mut out32)?;
        for (p64, p32) in out64.as_slice().chunks(3).zip(out32.as_slice().chunks(3)) {
            // f32-stored constants leave a residual that grows with L; measured
            // at white: u = -2.9e-5, v = -5.9e-5. The wrong white gave 0.13 / 0.04.
            assert!(p64[1].abs() < 1e-4 && p64[2].abs() < 1e-4, "f64 {p64:?}");
            assert!(
                p32[1].abs() < F32_CHROMA_TOL && p32[2].abs() < F32_CHROMA_TOL,
                "f32 {p32:?}"
            );
        }

        // (L, 0, 0) back to RGB is neutral, f64 and f32.
        let ls: Vec<f64> = (0..w)
            .flat_map(|i| [100.0 * grays[i % n], 0.0, 0.0])
            .collect();
        let ls32: Vec<f32> = ls.iter().map(|&v| v as f32).collect();
        let mut rgb = Image::<f64, 3>::from_size_val(size, 0.0)?;
        super::rgb_from_luv(&Image::<f64, 3>::new(size, ls)?, &mut rgb)?;
        for p in rgb.as_slice().chunks(3) {
            assert!(
                (p[0] - p[1]).abs() < 1e-5 && (p[1] - p[2]).abs() < 1e-5,
                "f64 rgb_from_luv(L, 0, 0) = {p:?}"
            );
        }
        let mut rgb32 = Image::<f32, 3>::from_size_val(size, 0.0)?;
        super::rgb_from_luv(&Image::<f32, 3>::new(size, ls32)?, &mut rgb32)?;
        for p in rgb32.as_slice().chunks(3) {
            assert!(
                (p[0] - p[1]).abs() < F32_NEUTRAL_TOL && (p[1] - p[2]).abs() < F32_NEUTRAL_TOL,
                "f32 rgb_from_luv(L, 0, 0) = {p:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn black_white_no_nan() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 2,
            height: 1,
        };
        let src = Image::<f32, 3>::new(size, vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0])?;
        for conv in [
            super::linear_rgb_from_rgb as fn(&_, &mut _) -> _,
            super::xyz_from_rgb,
            super::lab_from_rgb,
            super::luv_from_rgb,
        ] {
            let mut out = Image::<f32, 3>::from_size_val(size, 0.0)?;
            conv(&src, &mut out)?;
            for v in out.as_slice() {
                assert!(v.is_finite(), "NaN/inf in output: {v}");
            }
        }
        Ok(())
    }

    #[test]
    fn large_image_strip_path() -> Result<(), ImageError> {
        // > PAR_THRESHOLD (1,048,576) to exercise the rayon strip split.
        let (w, h) = (1024, 1025);
        let n = w * h * 3;
        let data: Vec<f32> = (0..n).map(|v| ((v % 251) as f32) / 250.0).collect();
        let size = ImageSize {
            width: w,
            height: h,
        };
        let src = Image::<f32, 3>::new(size, data)?;
        let mut mid = Image::<f32, 3>::from_size_val(size, 0.0)?;
        let mut back = Image::<f32, 3>::from_size_val(size, 0.0)?;
        super::lab_from_rgb(&src, &mut mid)?;
        super::rgb_from_lab(&mid, &mut back)?;
        for (a, b) in src.as_slice().iter().zip(back.as_slice().iter()) {
            assert!((a - b).abs() < 2e-3);
        }
        Ok(())
    }
}
