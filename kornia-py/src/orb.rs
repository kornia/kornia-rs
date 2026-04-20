use numpy::{PyArray, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;

use kornia_image::{allocator::CpuAllocator, Image, ImageSize};
use kornia_imgproc::{color::gray_from_rgb_u8, features::OrbDetector};

use crate::image::to_pyerr;

/// Detect ORB keypoints and compute descriptors on a u8 image.
///
/// Accepts a 2D `HxW` gray or a 3D `HxWx3` RGB numpy array. 3D RGB is
/// converted to gray first. Returns `(keypoints_xy, orientations, descriptors)`:
///
/// * `keypoints_xy` — `(N, 2)` float32 array, column/row in source pixels.
/// * `orientations` — `(N,)` float32 radians.
/// * `descriptors` — `(N, 32)` uint8 packed 256-bit descriptors.
#[pyfunction]
#[pyo3(signature = (image, n_keypoints=500, n_scales=8, downscale=1.2))]
pub fn orb_detect_and_compute(
    py: Python<'_>,
    image: Bound<'_, pyo3::types::PyAny>,
    n_keypoints: usize,
    n_scales: usize,
    downscale: f32,
) -> PyResult<(Py<PyArray2<f32>>, Py<PyArray1<f32>>, Py<PyArray2<u8>>)> {
    let gray = image_to_gray_u8(py, image)?;

    let detector = OrbDetector {
        n_keypoints,
        n_scales,
        downscale,
        ..Default::default()
    };

    let features = py
        .detach(|| detector.detect_and_extract_u8(&gray))
        .map_err(to_pyerr)?;

    let n = features.keypoints_xy.len();

    let (kps_arr, ori_arr, desc_arr) = unsafe {
        let kps_arr = PyArray::<f32, _>::new(py, [n, 2], false);
        let ori_arr = PyArray::<f32, _>::new(py, [n], false);
        let desc_arr = PyArray::<u8, _>::new(py, [n, 32], false);
        let kps_slice = std::slice::from_raw_parts_mut(kps_arr.data(), n * 2);
        for (i, xy) in features.keypoints_xy.iter().enumerate() {
            kps_slice[i * 2] = xy[0];
            kps_slice[i * 2 + 1] = xy[1];
        }
        let ori_slice = std::slice::from_raw_parts_mut(ori_arr.data(), n);
        ori_slice.copy_from_slice(&features.orientations);
        let desc_slice = std::slice::from_raw_parts_mut(desc_arr.data(), n * 32);
        for (i, d) in features.descriptors.iter().enumerate() {
            desc_slice[i * 32..(i + 1) * 32].copy_from_slice(d);
        }
        (kps_arr, ori_arr, desc_arr)
    };

    let kps_arr: Py<PyArray2<f32>> = kps_arr.unbind();
    let ori_arr: Py<PyArray1<f32>> = ori_arr.unbind();
    let desc_arr: Py<PyArray2<u8>> = desc_arr.unbind();
    Ok((kps_arr, ori_arr, desc_arr))
}

fn image_to_gray_u8(
    py: Python<'_>,
    image: Bound<'_, pyo3::types::PyAny>,
) -> PyResult<Image<u8, 1, CpuAllocator>> {
    if let Ok(arr) = image.cast::<PyArray2<u8>>() {
        if !arr.is_c_contiguous() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "gray array is not C-contiguous",
            ));
        }
        let shape = arr.shape();
        let (h, w) = (shape[0], shape[1]);
        let slice = unsafe { std::slice::from_raw_parts(arr.data(), h * w) };
        let size = ImageSize {
            width: w,
            height: h,
        };
        return Image::from_size_slice(size, slice, CpuAllocator).map_err(to_pyerr);
    }

    if let Ok(arr) = image.cast::<PyArray3<u8>>() {
        if !arr.is_c_contiguous() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "rgb array is not C-contiguous",
            ));
        }
        let shape = arr.shape();
        if shape[2] != 3 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "expected HxWx3 RGB u8 or HxW gray u8, got {}x{}x{}",
                shape[0], shape[1], shape[2]
            )));
        }
        let (h, w) = (shape[0], shape[1]);
        let size = ImageSize {
            width: w,
            height: h,
        };
        let rgb_slice = unsafe { std::slice::from_raw_parts(arr.data(), h * w * 3) };
        let rgb_img = Image::<u8, 3, _>::from_size_slice(size, rgb_slice, CpuAllocator)
            .map_err(to_pyerr)?;
        let mut gray = Image::from_size_val(size, 0u8, CpuAllocator).map_err(to_pyerr)?;
        py.detach(|| gray_from_rgb_u8(&rgb_img, &mut gray))
            .map_err(to_pyerr)?;
        return Ok(gray);
    }

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "expected HxW uint8 (gray) or HxWx3 uint8 (RGB) numpy array",
    ))
}
