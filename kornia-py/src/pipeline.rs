//! Python bindings for fused preprocessing pipelines.
//!
//! Exposes the single-pass `resize + normalize + HWC→CHW` kernel from
//! `kornia_imgproc::resize` as the `kornia_rs.pipeline` submodule.

use numpy::{PyArray, PyArray3, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use kornia_imgproc::resize::{resize_normalize_to_tensor_u8_to_f32, NormalizeParams};

use crate::image::{to_pyerr, PyImage};

/// Fused resize (2× exact downscale) + per-channel normalize + HWC→CHW layout
/// convert, all in one NEON pass.
///
/// This is the stateless entry point — each call allocates a fresh output
/// array. For zero-allocation hot loops, use the `Preprocessor` class.
///
/// # Arguments
///
/// * `image` — `(H, W, 3)` uint8 numpy array (HWC, C-contiguous).
/// * `new_size` — `(dst_h, dst_w)`. Must equal `(H/2, W/2)`.
/// * `mean` — per-channel mean in `[0, 1]` range (PyTorch convention).
/// * `std`  — per-channel std in `[0, 1]` range.
///
/// # Returns
///
/// `(3, dst_h, dst_w)` float32 numpy array in NCHW layout, normalized as
/// `(x/255 - mean) / std`.
#[pyfunction]
pub fn resize_normalize_to_tensor(
    py: Python<'_>,
    image: PyImage,
    new_size: (usize, usize),
    mean: [f32; 3],
    std: [f32; 3],
) -> PyResult<Py<PyArray3<f32>>> {
    let (src_h, src_w, src_slice) = validate_and_borrow_src(py, &image)?;
    let (dst_h, dst_w) = new_size;
    validate_shapes(src_h, src_w, dst_h, dst_w)?;
    let params = NormalizeParams::<3>::from_mean_std(mean, std);

    let out_arr = unsafe { PyArray::<f32, _>::new(py, [3, dst_h, dst_w], false) };
    let out_len = 3 * dst_h * dst_w;
    // SAFETY: out_arr is a freshly-allocated C-contiguous f32 PyArray3.
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out_arr.data(), out_len) };

    py.detach(|| {
        resize_normalize_to_tensor_u8_to_f32(
            src_slice, src_w, src_h, out_slice, dst_w, dst_h, &params,
        );
    });

    Ok(out_arr.unbind())
}

/// Pre-allocated preprocessor for the fused resize+normalize+HWC→CHW pipeline.
///
/// Owns its output buffer — constructs the `(3, dst_h, dst_w)` float32 numpy
/// array **once** at init, then reuses it on every call. Ideal for tight
/// training loops where per-frame allocation shows up in profiles.
///
/// # Aliasing warning
///
/// `__call__` returns a view into the **shared internal buffer**. Subsequent
/// calls overwrite it in place. Call `.copy()` on the result if you need to
/// keep a snapshot (e.g. when batching into a list).
///
/// # Example
///
/// ```python
/// pp = kornia_rs.pipeline.Preprocessor(
///     src_size=(1080, 1920),
///     dst_size=(540, 960),
///     mean=(0.485, 0.456, 0.406),
///     std=(0.229, 0.224, 0.225),
/// )
/// for img in stream:
///     t = pp(img)           # zero allocation in hot loop
///     model.forward(t)      # consume before next call, or t.copy()
/// ```
#[pyclass(name = "Preprocessor", module = "kornia_rs.pipeline")]
pub struct Preprocessor {
    src_h: usize,
    src_w: usize,
    dst_h: usize,
    dst_w: usize,
    params: NormalizeParams<3>,
    /// Preallocated `(3, dst_h, dst_w)` f32 NCHW buffer, reused across calls.
    out: Py<PyArray3<f32>>,
}

#[pymethods]
impl Preprocessor {
    /// Construct a preprocessor fixed to the given source/destination shapes
    /// and normalization parameters. Allocates the output buffer once here.
    #[new]
    #[pyo3(signature = (src_size, dst_size, mean, std))]
    fn new(
        py: Python<'_>,
        src_size: (usize, usize),
        dst_size: (usize, usize),
        mean: [f32; 3],
        std: [f32; 3],
    ) -> PyResult<Self> {
        let (src_h, src_w) = src_size;
        let (dst_h, dst_w) = dst_size;
        validate_shapes(src_h, src_w, dst_h, dst_w)?;
        let params = NormalizeParams::<3>::from_mean_std(mean, std);
        let out = unsafe { PyArray::<f32, _>::new(py, [3, dst_h, dst_w], false) };
        Ok(Self {
            src_h,
            src_w,
            dst_h,
            dst_w,
            params,
            out: out.unbind(),
        })
    }

    /// Run the fused preprocess on `image`, writing into the internal buffer.
    ///
    /// `image` must be `(src_h, src_w, 3)` uint8, C-contiguous. Returns the
    /// internal `(3, dst_h, dst_w)` f32 buffer (shared — see class doc).
    fn __call__(&mut self, py: Python<'_>, image: PyImage) -> PyResult<Py<PyArray3<f32>>> {
        let arr = image.bind(py);
        if !arr.is_c_contiguous() {
            return Err(PyErr::new::<PyValueError, _>(
                "input numpy array must be C-contiguous",
            ));
        }
        let shape = arr.shape();
        let (h, w, c) = (shape[0], shape[1], shape[2]);
        if h != self.src_h || w != self.src_w || c != 3 {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "expected image shape ({}, {}, 3), got ({}, {}, {})",
                self.src_h, self.src_w, h, w, c
            )));
        }
        // SAFETY: PyImage is a u8 PyArray3; shape validated above.
        let src_slice = unsafe { std::slice::from_raw_parts(arr.data(), h * w * 3) };

        let out_bound = self.out.bind(py);
        let out_len = 3 * self.dst_h * self.dst_w;
        // SAFETY: out buffer was allocated at construction at this exact shape;
        // `&mut self` on __call__ prevents concurrent Python-level aliasing.
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out_bound.data(), out_len) };

        py.detach(|| {
            resize_normalize_to_tensor_u8_to_f32(
                src_slice,
                self.src_w,
                self.src_h,
                out_slice,
                self.dst_w,
                self.dst_h,
                &self.params,
            );
        });

        Ok(self.out.clone_ref(py))
    }

    /// `(3, dst_h, dst_w)` output shape of this preprocessor.
    #[getter]
    fn output_shape(&self) -> (usize, usize, usize) {
        (3, self.dst_h, self.dst_w)
    }

    /// `(src_h, src_w, 3)` expected input shape.
    #[getter]
    fn input_shape(&self) -> (usize, usize, usize) {
        (self.src_h, self.src_w, 3)
    }
}

fn validate_and_borrow_src<'py>(
    py: Python<'py>,
    image: &'py PyImage,
) -> PyResult<(usize, usize, &'py [u8])> {
    let arr = image.bind(py);
    if !arr.is_c_contiguous() {
        return Err(PyErr::new::<PyValueError, _>(
            "input numpy array must be C-contiguous",
        ));
    }
    let shape = arr.shape();
    let (h, w, c) = (shape[0], shape[1], shape[2]);
    if c != 3 {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "expected 3 channels, got {c}"
        )));
    }
    // SAFETY: PyImage is a u8 PyArray3; shape validated above.
    let slice = unsafe { std::slice::from_raw_parts(arr.data(), h * w * 3) };
    Ok((h, w, slice))
}

fn validate_shapes(src_h: usize, src_w: usize, dst_h: usize, dst_w: usize) -> PyResult<()> {
    if src_h == 0 || src_w == 0 || dst_h == 0 || dst_w == 0 {
        return Err(PyErr::new::<PyValueError, _>(
            "source/destination shape has zero extent",
        ));
    }
    if src_h != 2 * dst_h || src_w != 2 * dst_w {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "only exact 2× downscale is supported: src=({src_h}, {src_w}) → dst=({dst_h}, {dst_w}) requires src = 2·dst"
        )));
    }
    Ok(())
}

// Keep `to_pyerr` in scope so future additions (e.g. fallible variants) don't
// need to re-import it.
#[allow(dead_code)]
fn _keep_import_alive(e: impl std::fmt::Display) -> PyErr {
    to_pyerr(e)
}
