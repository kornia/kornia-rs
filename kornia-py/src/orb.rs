use numpy::{PyArray, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;

use kornia_image::{allocator::CpuAllocator, Image, ImageSize};
use kornia_imgproc::{
    color::gray_from_rgb_u8,
    features::{fast_detect_rows_u8, OrbDetector},
};

use crate::image::to_pyerr;

/// ORB features extracted from a single frame.
///
/// Returned by [`orb_detect_and_compute`]. Attribute access keeps the call
/// site self-documenting and lets us add new per-keypoint fields (e.g.
/// responses, scales) later without breaking existing callers.
#[pyclass(name = "OrbFeatures", module = "kornia_rs.features", frozen)]
pub struct PyOrbFeatures {
    /// `(N, 2)` float32 keypoint positions (column, row) in source pixels.
    #[pyo3(get)]
    pub keypoints_xy: Py<PyArray2<f32>>,
    /// `(N,)` float32 orientation angles in radians.
    #[pyo3(get)]
    pub orientations: Py<PyArray1<f32>>,
    /// `(N, 32)` uint8 BRIEF descriptors (256-bit packed).
    #[pyo3(get)]
    pub descriptors: Py<PyArray2<u8>>,
    /// `(N,)` uint8 pyramid octave (0 = full resolution). Required by
    /// ORB-SLAM-style scale-aware matchers to reject cross-octave pairs and
    /// scale the search radius when projecting map points across frames.
    #[pyo3(get)]
    pub octaves: Py<PyArray1<u8>>,
    /// Cached feature count — cheap to expose so callers don't have to
    /// pay a numpy shape lookup.
    #[pyo3(get)]
    pub n: usize,
}

#[pymethods]
impl PyOrbFeatures {
    fn __len__(&self) -> usize {
        self.n
    }

    fn __repr__(&self) -> String {
        format!("OrbFeatures(n={})", self.n)
    }
}

/// Detect ORB keypoints and compute descriptors on a u8 image.
///
/// Accepts a 2D `HxW` gray or a 3D `HxWx3` RGB numpy array. 3D RGB is
/// converted to gray first. Returns an [`OrbFeatures`] object with
/// `.keypoints_xy`, `.orientations`, `.descriptors`, and `.octaves`
/// attributes (all numpy arrays of length `N`).
#[pyfunction]
#[pyo3(signature = (image, n_keypoints=500, n_scales=8, downscale=1.2))]
pub fn orb_detect_and_compute(
    py: Python<'_>,
    image: Bound<'_, pyo3::types::PyAny>,
    n_keypoints: usize,
    n_scales: usize,
    downscale: f32,
) -> PyResult<PyOrbFeatures> {
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

    let (kps_arr, ori_arr, desc_arr, oct_arr) = unsafe {
        let kps_arr = PyArray::<f32, _>::new(py, [n, 2], false);
        let ori_arr = PyArray::<f32, _>::new(py, [n], false);
        let desc_arr = PyArray::<u8, _>::new(py, [n, 32], false);
        let oct_arr = PyArray::<u8, _>::new(py, [n], false);
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
        let oct_slice = std::slice::from_raw_parts_mut(oct_arr.data(), n);
        oct_slice.copy_from_slice(&features.octaves);
        (kps_arr, ori_arr, desc_arr, oct_arr)
    };

    Ok(PyOrbFeatures {
        keypoints_xy: kps_arr.unbind(),
        orientations: ori_arr.unbind(),
        descriptors: desc_arr.unbind(),
        octaves: oct_arr.unbind(),
        n,
    })
}

/// Run just the FAST-9 corner detector on a u8 image (no Harris, no orientation,
/// no descriptor, no scale pyramid). The bare detector step, exposed separately
/// so callers can benchmark or compose it with other pipelines directly.
///
/// Args:
///     image: `HxW` uint8 gray or `HxWx3` uint8 RGB numpy array.
///     threshold: intensity threshold in u8 space 0..255 (default 20). Internally
///         normalized to [0, 1] for the Rust FastDetector.
///     arc_length: minimum arc length of contiguous bright/dark pixels (default 9 → FAST-9).
///     border: pixel border skipped at image edges (default 3, the FAST circle radius).
///
/// Returns:
///     `(keypoints_xy, responses)` where:
///     * `keypoints_xy` — `(N, 2)` float32 array of corner (x, y) positions.
///     * `responses` — `(N,)` float32 FAST score per corner.
#[pyfunction]
#[pyo3(signature = (image, threshold=20.0, arc_length=9, border=3))]
pub fn fast_detect(
    py: Python<'_>,
    image: Bound<'_, pyo3::types::PyAny>,
    threshold: f32,
    arc_length: usize,
    border: usize,
) -> PyResult<(Py<PyArray2<f32>>, Py<PyArray1<f32>>)> {
    let gray = image_to_gray_u8(py, image)?;
    let threshold_norm = (threshold / 255.0).clamp(0.0, 1.0);
    // Skip FastDetector::new — it allocates ~12 MB of scratch (response/mask/
    // taken buffers) per call that the fused single-pass path doesn't need.
    let margin = border.max(3);
    let h = gray.height();
    let rows = margin..h.saturating_sub(margin);
    let corners =
        py.detach(|| fast_detect_rows_u8(&gray, threshold_norm, arc_length, border, rows));

    let n = corners.len();
    let (kps_arr, resp_arr) = unsafe {
        let kps_arr = PyArray::<f32, _>::new(py, [n, 2], false);
        let resp_arr = PyArray::<f32, _>::new(py, [n], false);
        let kps_slice = std::slice::from_raw_parts_mut(kps_arr.data(), n * 2);
        let resp_slice = std::slice::from_raw_parts_mut(resp_arr.data(), n);
        for (i, ([row, col], score)) in corners.iter().enumerate() {
            kps_slice[i * 2] = *col as f32;
            kps_slice[i * 2 + 1] = *row as f32;
            resp_slice[i] = *score;
        }
        (kps_arr, resp_arr)
    };
    Ok((kps_arr.unbind(), resp_arr.unbind()))
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
