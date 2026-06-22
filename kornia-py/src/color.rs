use pyo3::prelude::*;

use crate::image::{
    alloc_output_pyarray, alloc_output_pyarray_f32, numpy_as_image, numpy_as_image_f32, to_pyerr,
    PyImage, PyImageF32,
};
use kornia_imgproc::color;

#[pyfunction]
pub fn rgb_from_gray(py: Python<'_>, image: PyImage) -> PyResult<PyImage> {
    let src = unsafe { numpy_as_image::<1>(py, &image)? };
    let (mut dst, out) = unsafe { alloc_output_pyarray::<3>(py, src.size())? };
    py.detach(|| color::rgb_from_gray(&src, &mut dst))
        .map_err(to_pyerr)?;
    Ok(out)
}

#[pyfunction]
pub fn bgr_from_rgb(py: Python<'_>, image: PyImage) -> PyResult<PyImage> {
    let src = unsafe { numpy_as_image::<3>(py, &image)? };
    let (mut dst, out) = unsafe { alloc_output_pyarray::<3>(py, src.size())? };
    py.detach(|| color::bgr_from_rgb(&src, &mut dst))
        .map_err(to_pyerr)?;
    Ok(out)
}

/// RGB f32 → grayscale f32, zero-copy in and out.
///
/// Accepts a (H, W, 3) numpy float32 array; returns a (H, W, 1) numpy float32 array.
/// GIL is released for the NEON/AVX2/scalar kernel invocation.
#[pyfunction]
pub fn gray_from_rgb_f32(py: Python<'_>, image: PyImageF32) -> PyResult<PyImageF32> {
    let src = unsafe { numpy_as_image_f32::<3>(py, &image)? };
    let (mut dst, out) = unsafe { alloc_output_pyarray_f32::<1>(py, src.size())? };
    py.detach(|| color::gray_from_rgb_f32(&src, &mut dst))
        .map_err(to_pyerr)?;
    Ok(out)
}

#[pyfunction]
pub fn gray_from_rgb(py: Python<'_>, image: PyImage) -> PyResult<PyImage> {
    let src = unsafe { numpy_as_image::<3>(py, &image)? };
    let (mut dst, out) = unsafe { alloc_output_pyarray::<1>(py, src.size())? };
    py.detach(|| color::gray_from_rgb_u8(&src, &mut dst))
        .map_err(to_pyerr)?;
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (image, background=None))]
pub fn rgb_from_rgba(
    py: Python<'_>,
    image: PyImage,
    background: Option<[u8; 3]>,
) -> PyResult<PyImage> {
    let src = unsafe { numpy_as_image::<4>(py, &image)? };
    let (mut dst, out) = unsafe { alloc_output_pyarray::<3>(py, src.size())? };
    py.detach(|| color::rgb_from_rgba(&src, &mut dst, background))
        .map_err(to_pyerr)?;
    Ok(out)
}

#[pyfunction]
pub fn apply_colormap(py: Python<'_>, image: PyImage, colormap: &str) -> PyResult<PyImage> {
    // Validate name before touching the image array — fail fast on bad input.
    let cm = color::ColormapType::from_name(colormap).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "unknown colormap '{colormap}'; valid names: autumn, bone, jet, winter, rainbow, \
             ocean, summer, spring, cool, hsv, pink, hot, parula, magma, inferno, plasma, \
             viridis, cividis, twilight, turbo, deepgreen"
        ))
    })?;
    let src = unsafe { numpy_as_image::<1>(py, &image)? };
    let (mut dst, out) = unsafe { alloc_output_pyarray::<3>(py, src.size())? };
    py.detach(|| color::apply_colormap(&src, &mut dst, cm))
        .map_err(to_pyerr)?;
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (image, background=None))]
pub fn rgb_from_bgra(
    py: Python<'_>,
    image: PyImage,
    background: Option<[u8; 3]>,
) -> PyResult<PyImage> {
    let src = unsafe { numpy_as_image::<4>(py, &image)? };
    let (mut dst, out) = unsafe { alloc_output_pyarray::<3>(py, src.size())? };
    py.detach(|| color::rgb_from_bgra(&src, &mut dst, background))
        .map_err(to_pyerr)?;
    Ok(out)
}

/// Generates a zero-copy f32 3→3 channel color-conversion `#[pyfunction]`.
///
/// All perceptual/cylindrical conversions share the same shape: a (H, W, 3) float32
/// input → (H, W, 3) float32 output, GIL released for the NEON/AVX2 kernel.
macro_rules! py_f32_3to3 {
    ($name:ident, $func:path, $doc:literal) => {
        #[doc = $doc]
        #[pyfunction]
        pub fn $name(py: Python<'_>, image: PyImageF32) -> PyResult<PyImageF32> {
            let src = unsafe { numpy_as_image_f32::<3>(py, &image)? };
            let (mut dst, out) = unsafe { alloc_output_pyarray_f32::<3>(py, src.size())? };
            py.detach(|| $func(&src, &mut dst)).map_err(to_pyerr)?;
            Ok(out)
        }
    };
}

py_f32_3to3!(hsv_from_rgb, color::hsv_from_rgb, "RGB f32 → HSV f32 (H,S,V in [0,255]).");
py_f32_3to3!(rgb_from_hsv, color::rgb_from_hsv, "HSV f32 → RGB f32.");
py_f32_3to3!(hls_from_rgb, color::hls_from_rgb, "RGB f32 → HLS f32.");
py_f32_3to3!(rgb_from_hls, color::rgb_from_hls, "HLS f32 → RGB f32.");
py_f32_3to3!(xyz_from_rgb, color::xyz_from_rgb, "RGB f32 → CIE XYZ f32 (D65).");
py_f32_3to3!(rgb_from_xyz, color::rgb_from_xyz, "CIE XYZ f32 → RGB f32.");
py_f32_3to3!(lab_from_rgb, color::lab_from_rgb, "RGB f32 → CIE L*a*b* f32.");
py_f32_3to3!(rgb_from_lab, color::rgb_from_lab, "CIE L*a*b* f32 → RGB f32.");
py_f32_3to3!(luv_from_rgb, color::luv_from_rgb, "RGB f32 → CIE L*u*v* f32.");
py_f32_3to3!(rgb_from_luv, color::rgb_from_luv, "CIE L*u*v* f32 → RGB f32.");
py_f32_3to3!(
    linear_rgb_from_rgb,
    color::linear_rgb_from_rgb,
    "sRGB f32 → linear-RGB f32 (gamma expand)."
);
py_f32_3to3!(
    rgb_from_linear_rgb,
    color::rgb_from_linear_rgb,
    "linear-RGB f32 → sRGB f32 (gamma compress)."
);
py_f32_3to3!(ycbcr_from_rgb, color::ycbcr_from_rgb, "RGB f32 → YCbCr f32 (full range).");
py_f32_3to3!(rgb_from_ycbcr, color::rgb_from_ycbcr, "YCbCr f32 → RGB f32.");
py_f32_3to3!(yuv_from_rgb, color::yuv_from_rgb, "RGB f32 → YUV f32 (planar, full range).");
py_f32_3to3!(rgb_from_yuv, color::rgb_from_yuv, "YUV f32 → RGB f32.");
py_f32_3to3!(sepia_from_rgb, color::sepia_from_rgb_f32, "RGB f32 → sepia-toned RGB f32.");

/// Demosaic a single-channel u8 Bayer mosaic to RGB (bilinear).
///
/// `pattern` is the sensor layout on kornia's *sensor-truth* convention:
/// `"rggb"`, `"bggr"`, `"grbg"`, `"gbrg"` (R at (0,0) for rggb, etc.). Note OpenCV's
/// naming offset: kornia `"rggb"` == `cv2.COLOR_BayerBG2RGB`.
#[pyfunction]
pub fn rgb_from_bayer(py: Python<'_>, image: PyImage, pattern: &str) -> PyResult<PyImage> {
    use kornia_imgproc::color::BayerPattern;
    let pat = match pattern.to_lowercase().as_str() {
        "rggb" => BayerPattern::Rggb,
        "bggr" => BayerPattern::Bggr,
        "grbg" => BayerPattern::Grbg,
        "gbrg" => BayerPattern::Gbrg,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown Bayer pattern '{other}'; valid: rggb, bggr, grbg, gbrg"
            )))
        }
    };
    let src = unsafe { numpy_as_image::<1>(py, &image)? };
    let (mut dst, out) = unsafe { alloc_output_pyarray::<3>(py, src.size())? };
    py.detach(|| color::rgb_from_bayer(&src, pat, &mut dst))
        .map_err(to_pyerr)?;
    Ok(out)
}
