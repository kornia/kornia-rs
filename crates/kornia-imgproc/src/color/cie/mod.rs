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
use kornia_image::{allocator::ImageAllocator, Image, ImageError};

mod kernels;
mod nonlinear;
mod transfer;

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

#[inline]
fn check_size<T, U, const C1: usize, const C2: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<T, C1, A1>,
    dst: &Image<U, C2, A2>,
) -> Result<(), ImageError> {
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }
    Ok(())
}

// ===== trait + impl generation ======================================================
//
// Each conversion is a sealed trait with an f32 (SIMD) and f64 (scalar oracle) impl.
// `cie_conv!` stamps out the trait, both impls, and the public dispatch fn.

macro_rules! cie_conv {
    ($trait:ident, $method:ident, $pub_fn:ident, $f32_kernel:path, $f64_oracle:path, $doc:expr) => {
        #[doc = $doc]
        pub trait $trait: sealed::Sealed + Sized {
            #[doc(hidden)]
            fn $method<A1: ImageAllocator, A2: ImageAllocator>(
                src: &Image<Self, 3, A1>,
                dst: &mut Image<Self, 3, A2>,
            ) -> Result<(), ImageError>;
        }

        impl $trait for f32 {
            fn $method<A1: ImageAllocator, A2: ImageAllocator>(
                src: &Image<f32, 3, A1>,
                dst: &mut Image<f32, 3, A2>,
            ) -> Result<(), ImageError> {
                check_size(src, dst)?;
                $f32_kernel(src.as_slice(), dst.as_slice_mut(), src.rows() * src.cols());
                Ok(())
            }
        }

        impl $trait for f64 {
            fn $method<A1: ImageAllocator, A2: ImageAllocator>(
                src: &Image<f64, 3, A1>,
                dst: &mut Image<f64, 3, A2>,
            ) -> Result<(), ImageError> {
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
        pub fn $pub_fn<T, A1, A2>(
            src: &Image<T, 3, A1>,
            dst: &mut Image<T, 3, A2>,
        ) -> Result<(), ImageError>
        where
            T: $trait,
            A1: ImageAllocator,
            A2: ImageAllocator,
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
    ($name:ident, $generic:ident, $doc:expr) => {
        #[doc = $doc]
        pub fn $name<A1: ImageAllocator, A2: ImageAllocator>(
            src: &Image<f32, 3, A1>,
            dst: &mut Image<f32, 3, A2>,
        ) -> Result<(), ImageError> {
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
    use kornia_tensor::CpuAllocator;

    // Build an f32 RGB image and its f64 twin from the same [0,1] samples.
    fn pair(w: usize, h: usize) -> (Vec<f32>, Vec<f64>) {
        let n = w * h * 3;
        let f32v: Vec<f32> = (0..n).map(|v| ((v * 7 % 251) as f32) / 250.0).collect();
        let f64v: Vec<f64> = f32v.iter().map(|&x| x as f64).collect();
        (f32v, f64v)
    }

    // Conversion fn-pointer aliases (keep clippy::type_complexity quiet in test helpers).
    type ConvFn<T> =
        fn(&Image<T, 3, CpuAllocator>, &mut Image<T, 3, CpuAllocator>) -> Result<(), ImageError>;

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
        let src32 = Image::<f32, 3, _>::new(size, f32v, CpuAllocator)?;
        let src64 = Image::<f64, 3, _>::new(size, f64v, CpuAllocator)?;
        let mut out32 = Image::<f32, 3, _>::from_size_val(size, 0.0, CpuAllocator)?;
        let mut out64 = Image::<f64, 3, _>::from_size_val(size, 0.0, CpuAllocator)?;
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
        let src = Image::<f32, 3, _>::new(size, f32v, CpuAllocator)?;
        let mut mid = Image::<f32, 3, _>::from_size_val(size, 0.0, CpuAllocator)?;
        let mut back = Image::<f32, 3, _>::from_size_val(size, 0.0, CpuAllocator)?;
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

    #[test]
    fn black_white_no_nan() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 2,
            height: 1,
        };
        let src = Image::<f32, 3, _>::new(size, vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0], CpuAllocator)?;
        for conv in [
            super::linear_rgb_from_rgb as fn(&_, &mut _) -> _,
            super::xyz_from_rgb,
            super::lab_from_rgb,
            super::luv_from_rgb,
        ] {
            let mut out = Image::<f32, 3, _>::from_size_val(size, 0.0, CpuAllocator)?;
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
        let src = Image::<f32, 3, _>::new(size, data, CpuAllocator)?;
        let mut mid = Image::<f32, 3, _>::from_size_val(size, 0.0, CpuAllocator)?;
        let mut back = Image::<f32, 3, _>::from_size_val(size, 0.0, CpuAllocator)?;
        super::lab_from_rgb(&src, &mut mid)?;
        super::rgb_from_lab(&mid, &mut back)?;
        for (a, b) in src.as_slice().iter().zip(back.as_slice().iter()) {
            assert!((a - b).abs() < 2e-3);
        }
        Ok(())
    }
}
