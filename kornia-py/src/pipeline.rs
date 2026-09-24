//! Python bindings for fused preprocessing pipelines.
//!
//! Exposes the single-pass `resize + normalize + HWC→CHW` kernel from
//! `kornia_imgproc::resize` as the `kornia_rs.pipeline` submodule.

use numpy::{PyArray, PyArray3, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use kornia_imgproc::resize::{resize_normalize_to_tensor_u8_to_f32_bilinear, NormalizeParams};

use crate::image::{to_pyerr, PyImage};
use crate::pyutils::{
    c_slice, checked_numel, ranges_overlap, require_c_contig_aligned, require_writeable,
};

/// Fused resize (general bilinear, any target size) + per-channel normalize +
/// HWC→CHW layout convert, all in one pass. Exact 2× downscale takes a faster
/// fused box path internally.
///
/// This is the stateless entry point — each call allocates a fresh output
/// array. For zero-allocation hot loops, use the `Preprocessor` class.
///
/// # Arguments
///
/// * `image` — `(H, W, 3)` uint8 numpy array (HWC, C-contiguous).
/// * `new_size` — `(dst_h, dst_w)`. Any positive size (up- or down-scale).
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

    let out_len = checked_numel(&[3, dst_h, dst_w], std::mem::size_of::<f32>())?;
    // SAFETY: the dimensions were validated by `checked_numel`; the fused kernel
    // writes every one of the `out_len` elements before the array is returned.
    let out_arr = unsafe { PyArray::<f32, _>::new(py, [3, dst_h, dst_w], false) };
    // SAFETY: out_arr is a freshly-allocated C-contiguous f32 PyArray3 of
    // exactly `out_len` elements, not yet shared with Python.
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out_arr.data(), out_len) };

    let result = py.detach(|| {
        resize_normalize_to_tensor_u8_to_f32_bilinear(
            src_slice, src_w, src_h, out_slice, dst_w, dst_h, &params,
        )
    });
    result.map_err(to_pyerr)?;

    Ok(out_arr.unbind())
}

/// Batched [`resize_normalize_to_tensor`]: preprocess a whole list of images
/// in one call — the GIL is released once and the images are processed in
/// parallel across the rayon pool (parallelism ACROSS images, which is what a
/// dataloader wants — per-image thread fan-out would fight the loader's own
/// worker parallelism).
///
/// Returns one `(3, dst_h, dst_w)` float32 NCHW array per input, in order.
#[pyfunction]
pub fn resize_normalize_to_tensor_batch(
    py: Python<'_>,
    images: Vec<PyImage>,
    new_size: (usize, usize),
    mean: [f32; 3],
    std: [f32; 3],
) -> PyResult<Vec<Py<PyArray3<f32>>>> {
    let (dst_h, dst_w) = new_size;
    let params = NormalizeParams::<3>::from_mean_std(mean, std);

    // Borrow every source and allocate every output under the GIL…
    let mut srcs = Vec::with_capacity(images.len());
    for image in &images {
        let (src_h, src_w, src_slice) = validate_and_borrow_src(py, image)?;
        validate_shapes(src_h, src_w, dst_h, dst_w)?;
        srcs.push((src_h, src_w, src_slice));
    }
    let out_len = checked_numel(&[3, dst_h, dst_w], std::mem::size_of::<f32>())?;
    let mut outs = Vec::with_capacity(images.len());
    let mut out_slices: Vec<&mut [f32]> = Vec::with_capacity(images.len());
    for _ in &images {
        let arr = unsafe { PyArray::<f32, _>::new(py, [3, dst_h, dst_w], false) };
        // SAFETY: freshly-allocated C-contiguous f32 PyArray3, kept alive by `outs`.
        out_slices.push(unsafe { std::slice::from_raw_parts_mut(arr.data(), out_len) });
        outs.push(arr.unbind());
    }

    // …then release it once and run the images back-to-back: each fused call
    // already parallelizes internally across the rayon pool, so fanning out
    // across images too would just oversubscribe the cores (measured slower).
    // The batch win is amortizing the GIL round-trip and Python call overhead.
    let result: Result<(), _> = py.detach(|| {
        srcs.iter()
            .zip(out_slices.iter_mut())
            .try_for_each(|((src_h, src_w, src_slice), out)| {
                resize_normalize_to_tensor_u8_to_f32_bilinear(
                    src_slice, *src_w, *src_h, out, dst_w, dst_h, &params,
                )
            })
    });
    result.map_err(to_pyerr)?;
    Ok(outs)
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
        checked_numel(&[3, dst_h, dst_w], std::mem::size_of::<f32>())?;
        checked_numel(&[src_h, src_w, 3], 1)?;
        let params = NormalizeParams::<3>::from_mean_std(mean, std);
        let out = PyArray::<f32, _>::zeros(py, [3, dst_h, dst_w], false);
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
        let shape = arr.shape();
        let (h, w, c) = (shape[0], shape[1], shape[2]);
        if h != self.src_h || w != self.src_w || c != 3 {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "expected image shape ({}, {}, 3), got ({}, {}, {})",
                self.src_h, self.src_w, h, w, c
            )));
        }
        let src_slice = c_slice(arr, "input numpy array")?;

        // The internal buffer is handed out to Python on every call, so Python
        // can change it behind our back (`out.resize(..., refcheck=False)`
        // reallocates it smaller, `out.setflags(write=False)` freezes it).
        // Re-validate it on EVERY call and fall back to a fresh buffer if it is
        // no longer exactly the (3, dst_h, dst_w) contiguous writeable array we
        // allocated — never trust the construction-time shape.
        let expected = [3, self.dst_h, self.dst_w];
        let reusable = {
            let out_bound = self.out.bind(py);
            out_bound.shape() == expected
                && require_c_contig_aligned(out_bound, "Preprocessor buffer").is_ok()
                && require_writeable(out_bound, "Preprocessor buffer").is_ok()
                && !ranges_overlap(
                    out_bound.data() as *const u8,
                    out_bound.len() * std::mem::size_of::<f32>(),
                    src_slice.as_ptr(),
                    src_slice.len(),
                )
        };
        if !reusable {
            self.out = PyArray::<f32, _>::zeros(py, expected, false).unbind();
        }
        let out_bound = self.out.bind(py);
        // Length comes from the live array (validated == 3*dst_h*dst_w above).
        let out_len = out_bound.len();
        // SAFETY: `out_bound` is C-contiguous, aligned, writeable and exactly
        // (3, dst_h, dst_w) (re-validated just above, or freshly allocated), and
        // does not overlap `src_slice`. `&mut self` prevents re-entrant use of
        // the buffer from this object while the kernel runs.
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out_bound.data(), out_len) };

        let result = py.detach(|| {
            resize_normalize_to_tensor_u8_to_f32_bilinear(
                src_slice,
                self.src_w,
                self.src_h,
                out_slice,
                self.dst_w,
                self.dst_h,
                &self.params,
            )
        });
        result.map_err(to_pyerr)?;

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
    let shape = arr.shape();
    let (h, w, c) = (shape[0], shape[1], shape[2]);
    if c != 3 {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "expected 3 channels, got {c}"
        )));
    }
    let slice = c_slice(arr, "input numpy array")?;
    Ok((h, w, slice))
}

fn validate_shapes(src_h: usize, src_w: usize, dst_h: usize, dst_w: usize) -> PyResult<()> {
    if src_h == 0 || src_w == 0 || dst_h == 0 || dst_w == 0 {
        return Err(PyErr::new::<PyValueError, _>(
            "source/destination shape has zero extent",
        ));
    }
    // Any src→dst ratio is supported (general bilinear); exact 2× downscale takes
    // a faster fused box path internally.
    Ok(())
}
