use numpy::{PyArray, PyArray1, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;

use crate::dispatch::{cpu_op, try_dispatch_device};

use crate::image::{
    alloc_output_pyarray, alloc_output_pyarray_f32, numpy_as_image, numpy_as_image_f32, to_pyerr,
    PyImage, PyImageF32,
};
use kornia_image::ImageSize;
use kornia_imgproc::color;

#[pyfunction]
pub fn rgb_from_gray(py: Python<'_>, image: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    try_dispatch_device!(py, image, crate::cuda_ext::rgb_from_gray);
    cpu_op(py, image, |py, image| {
        let src = unsafe { numpy_as_image::<1>(py, &image)? };
        let (mut dst, out) = unsafe { alloc_output_pyarray::<3>(py, src.size())? };
        py.detach(|| color::rgb_from_gray(&src, &mut dst))
            .map_err(to_pyerr)?;
        Ok(out)
    })
}

#[pyfunction]
pub fn bgr_from_rgb(py: Python<'_>, image: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    try_dispatch_device!(py, image, crate::cuda_ext::bgr_from_rgb);
    cpu_op(py, image, |py, image| {
        let src = unsafe { numpy_as_image::<3>(py, &image)? };
        let (mut dst, out) = unsafe { alloc_output_pyarray::<3>(py, src.size())? };
        py.detach(|| color::bgr_from_rgb(&src, &mut dst))
            .map_err(to_pyerr)?;
        Ok(out)
    })
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
pub fn gray_from_rgb(py: Python<'_>, image: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    try_dispatch_device!(py, image, crate::cuda_ext::gray_from_rgb);
    cpu_op(py, image, |py, image| {
        let src = unsafe { numpy_as_image::<3>(py, &image)? };
        let (mut dst, out) = unsafe { alloc_output_pyarray::<1>(py, src.size())? };
        py.detach(|| color::gray_from_rgb_u8(&src, &mut dst))
            .map_err(to_pyerr)?;
        Ok(out)
    })
}

#[pyfunction]
pub fn rgba_from_rgb(py: Python<'_>, image: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    try_dispatch_device!(py, image, crate::cuda_ext::rgba_from_rgb);
    cpu_op(py, image, |py, image| {
        let src = unsafe { numpy_as_image::<3>(py, &image)? };
        let (mut dst, out) = unsafe { alloc_output_pyarray::<4>(py, src.size())? };
        py.detach(|| color::rgba_from_rgb(&src, &mut dst))
            .map_err(to_pyerr)?;
        Ok(out)
    })
}

#[pyfunction]
#[pyo3(signature = (image, background=None))]
pub fn rgb_from_rgba(
    py: Python<'_>,
    image: &Bound<'_, PyAny>,
    background: Option<[u8; 3]>,
) -> PyResult<Py<PyAny>> {
    // The GPU path honors `background` too (alpha composite when set, else
    // opaque drop) — same result as the CPU path below.
    try_dispatch_device!(py, image, |api| crate::cuda_ext::rgb_from_rgba_bg(
        api, background
    ));
    cpu_op(py, image, |py, image| {
        let src = unsafe { numpy_as_image::<4>(py, &image)? };
        let (mut dst, out) = unsafe { alloc_output_pyarray::<3>(py, src.size())? };
        py.detach(|| color::rgb_from_rgba(&src, &mut dst, background))
            .map_err(to_pyerr)?;
        Ok(out)
    })
}

#[pyfunction]
pub fn apply_colormap(
    py: Python<'_>,
    image: &Bound<'_, PyAny>,
    colormap: &str,
) -> PyResult<Py<PyAny>> {
    try_dispatch_device!(py, image, |api| crate::cuda_ext::apply_colormap(
        api, colormap
    ));
    // Validate name before touching the image array — fail fast on bad input.
    let cm = color::ColormapType::from_name(colormap).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "unknown colormap '{colormap}'; valid names: autumn, bone, jet, winter, rainbow, \
             ocean, summer, spring, cool, hsv, pink, hot, parula, magma, inferno, plasma, \
             viridis, cividis, twilight, turbo, deepgreen"
        ))
    })?;
    cpu_op(py, image, |py, image| {
        let src = unsafe { numpy_as_image::<1>(py, &image)? };
        let (mut dst, out) = unsafe { alloc_output_pyarray::<3>(py, src.size())? };
        py.detach(|| color::apply_colormap(&src, &mut dst, cm))
            .map_err(to_pyerr)?;
        Ok(out)
    })
}

#[pyfunction]
#[pyo3(signature = (image, background=None))]
pub fn rgb_from_bgra(
    py: Python<'_>,
    image: &Bound<'_, PyAny>,
    background: Option<[u8; 3]>,
) -> PyResult<Py<PyAny>> {
    // GPU path honors `background` (see `rgb_from_rgba`).
    try_dispatch_device!(py, image, |api| crate::cuda_ext::rgb_from_bgra_bg(
        api, background
    ));
    cpu_op(py, image, |py, image| {
        let src = unsafe { numpy_as_image::<4>(py, &image)? };
        let (mut dst, out) = unsafe { alloc_output_pyarray::<3>(py, src.size())? };
        py.detach(|| color::rgb_from_bgra(&src, &mut dst, background))
            .map_err(to_pyerr)?;
        Ok(out)
    })
}

/// Generates a residency-dispatching f32 3→3 channel color-conversion
/// `#[pyfunction]`: numpy / host `Image` run CPU (`$func`, GIL released for the
/// NEON/AVX2 kernel), a device `Image` runs the GPU op (`$dev`, if given —
/// returning a device `Image`). All perceptual/cylindrical conversions share
/// this (H, W, 3) float32 → (H, W, 3) float32 shape.
macro_rules! py_f32_3to3 {
    ($name:ident, $func:path, $doc:literal) => {
        #[doc = $doc]
        #[pyfunction]
        pub fn $name(py: Python<'_>, image: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
            cpu_op(py, image, |py, image| {
                let src = unsafe { numpy_as_image_f32::<3>(py, &image)? };
                let (mut dst, out) = unsafe { alloc_output_pyarray_f32::<3>(py, src.size())? };
                py.detach(|| $func(&src, &mut dst)).map_err(to_pyerr)?;
                Ok(out)
            })
        }
    };
    ($name:ident, $func:path, $dev:path, $doc:literal) => {
        #[doc = $doc]
        #[pyfunction]
        pub fn $name(py: Python<'_>, image: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
            try_dispatch_device!(py, image, $dev);
            cpu_op(py, image, |py, image| {
                let src = unsafe { numpy_as_image_f32::<3>(py, &image)? };
                let (mut dst, out) = unsafe { alloc_output_pyarray_f32::<3>(py, src.size())? };
                py.detach(|| $func(&src, &mut dst)).map_err(to_pyerr)?;
                Ok(out)
            })
        }
    };
}

py_f32_3to3!(
    hsv_from_rgb,
    color::hsv_from_rgb,
    crate::cuda_ext::hsv_from_rgb,
    "RGB f32 → HSV f32 (H,S,V in [0,255])."
);
py_f32_3to3!(
    rgb_from_hsv,
    color::rgb_from_hsv,
    crate::cuda_ext::rgb_from_hsv,
    "HSV f32 → RGB f32."
);
py_f32_3to3!(hls_from_rgb, color::hls_from_rgb, "RGB f32 → HLS f32.");
py_f32_3to3!(rgb_from_hls, color::rgb_from_hls, "HLS f32 → RGB f32.");
py_f32_3to3!(
    xyz_from_rgb,
    color::xyz_from_rgb,
    "RGB f32 → CIE XYZ f32 (D65)."
);
py_f32_3to3!(rgb_from_xyz, color::rgb_from_xyz, "CIE XYZ f32 → RGB f32.");
py_f32_3to3!(
    lab_from_rgb,
    color::lab_from_rgb,
    crate::cuda_ext::lab_from_rgb,
    "RGB f32 → CIE L*a*b* f32."
);
py_f32_3to3!(
    rgb_from_lab,
    color::rgb_from_lab,
    crate::cuda_ext::rgb_from_lab,
    "CIE L*a*b* f32 → RGB f32."
);
py_f32_3to3!(
    luv_from_rgb,
    color::luv_from_rgb,
    "RGB f32 → CIE L*u*v* f32."
);
py_f32_3to3!(
    rgb_from_luv,
    color::rgb_from_luv,
    "CIE L*u*v* f32 → RGB f32."
);
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
py_f32_3to3!(
    ycbcr_from_rgb,
    color::ycbcr_from_rgb,
    crate::cuda_ext::ycbcr_from_rgb,
    "RGB f32 → YCbCr f32 (full range)."
);
py_f32_3to3!(
    rgb_from_ycbcr,
    color::rgb_from_ycbcr,
    crate::cuda_ext::rgb_from_ycbcr,
    "YCbCr f32 → RGB f32."
);
py_f32_3to3!(
    yuv_from_rgb,
    color::yuv_from_rgb,
    "RGB f32 → YUV f32 (planar, full range)."
);
py_f32_3to3!(rgb_from_yuv, color::rgb_from_yuv, "YUV f32 → RGB f32.");
py_f32_3to3!(
    sepia_from_rgb,
    color::sepia_from_rgb_f32,
    crate::cuda_ext::sepia_from_rgb,
    "RGB f32 → sepia-toned RGB f32."
);

/// Demosaic a single-channel u8 Bayer mosaic to RGB (bilinear).
///
/// `pattern` is the sensor layout on kornia's *sensor-truth* convention:
/// `"rggb"`, `"bggr"`, `"grbg"`, `"gbrg"` (R at (0,0) for rggb, etc.). Note OpenCV's
/// naming offset: kornia `"rggb"` == `cv2.COLOR_BayerBG2RGB`.
#[pyfunction]
pub fn rgb_from_bayer(
    py: Python<'_>,
    image: &Bound<'_, PyAny>,
    pattern: &str,
) -> PyResult<Py<PyAny>> {
    try_dispatch_device!(py, image, |api| crate::cuda_ext::rgb_from_bayer(
        api, pattern
    ));
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
    cpu_op(py, image, |py, image| {
        let src = unsafe { numpy_as_image::<1>(py, &image)? };
        let (mut dst, out) = unsafe { alloc_output_pyarray::<3>(py, src.size())? };
        py.detach(|| color::rgb_from_bayer(&src, pat, &mut dst))
            .map_err(to_pyerr)?;
        Ok(out)
    })
}

/// Decode a raw packed/planar YUV byte buffer (1-D `uint8`) to an `(H, W, 3)` RGB image.
///
/// The buffer layout and required length are determined by the format:
/// packed 4:2:2 (`yuyv`/`uyvy`/`yvyu`) needs `W*H*2` bytes; planar 4:2:0
/// (`nv12`/`nv21`/`i420`/`yv12`) needs `W*H*3/2` bytes (Y plane followed by chroma).
/// BT.601 limited range, matching OpenCV's `COLOR_YUV2RGB_*`.
macro_rules! py_video_decode {
    ($name:ident, $func:path, $doc:expr) => {
        #[doc = $doc]
        #[pyfunction]
        pub fn $name(
            py: Python<'_>,
            data: Py<PyArray1<u8>>,
            width: usize,
            height: usize,
        ) -> PyResult<PyImage> {
            let arr = data.bind(py);
            if !arr.is_c_contiguous() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "YUV buffer must be a C-contiguous 1-D uint8 array",
                ));
            }
            // SAFETY: `data` is owned for the call, keeping the buffer alive; the slice is
            // only read inside `py.detach` while `arr` remains valid.
            let src = unsafe { std::slice::from_raw_parts(arr.data(), arr.len()) };
            let (mut dst, out) =
                unsafe { alloc_output_pyarray::<3>(py, ImageSize { width, height })? };
            // Length validation happens inside the kernel (returns InvalidImageSize).
            py.detach(|| $func(src, &mut dst)).map_err(to_pyerr)?;
            Ok(out)
        }
    };
}

py_video_decode!(
    rgb_from_yuyv,
    color::rgb_from_yuyv,
    "Decode packed 4:2:2 YUYV to RGB."
);
py_video_decode!(
    rgb_from_uyvy,
    color::rgb_from_uyvy,
    "Decode packed 4:2:2 UYVY to RGB."
);
py_video_decode!(
    rgb_from_yvyu,
    color::rgb_from_yvyu,
    "Decode packed 4:2:2 YVYU to RGB."
);
py_video_decode!(
    rgb_from_nv12,
    color::rgb_from_nv12,
    "Decode planar 4:2:0 NV12 to RGB."
);
py_video_decode!(
    rgb_from_nv21,
    color::rgb_from_nv21,
    "Decode planar 4:2:0 NV21 to RGB."
);
py_video_decode!(
    rgb_from_i420,
    color::rgb_from_i420,
    "Decode planar 4:2:0 I420 to RGB."
);
py_video_decode!(
    rgb_from_yv12,
    color::rgb_from_yv12,
    "Decode planar 4:2:0 YV12 to RGB."
);

macro_rules! py_video_encode {
    ($name:ident, $func:path, $len_expr:expr, $doc:literal) => {
        #[doc = $doc]
        #[pyfunction]
        pub fn $name(py: Python<'_>, image: PyImage) -> PyResult<Py<PyArray1<u8>>> {
            let src = unsafe { numpy_as_image::<3>(py, &image)? };
            let (w, h) = (src.width(), src.height());
            let len = $len_expr(w, h);
            let out = unsafe { PyArray::<u8, _>::new(py, [len], false) };
            // SAFETY: freshly-allocated 1-D uint8 array, not yet shared.
            let out_slice = unsafe { std::slice::from_raw_parts_mut(out.data(), len) };
            py.detach(|| $func(&src, out_slice)).map_err(to_pyerr)?;
            Ok(out.unbind())
        }
    };
}

py_video_encode!(
    yuyv_from_rgb,
    color::yuyv_from_rgb,
    |w: usize, h: usize| -> usize { w * h * 2 },
    "Encode an `(H, W, 3)` uint8 RGB image to a packed 4:2:2 **YUYV** byte buffer \
     (1-D `uint8`, length `W*H*2`), BT.601 limited range. Inverse of `rgb_from_yuyv`. \
     `width` must be even."
);
py_video_encode!(
    nv12_from_rgb,
    color::nv12_from_rgb,
    |w: usize, h: usize| -> usize { w * h * 3 / 2 },
    "Encode an `(H, W, 3)` uint8 RGB image to a planar 4:2:0 **NV12** byte buffer \
     (1-D `uint8`, length `W*H*3/2`: Y plane + interleaved `UV`), BT.601 limited range. \
     Inverse of `rgb_from_nv12`. `width` and `height` must be even."
);
