use numpy::PyUntypedArrayMethods;
use pyo3::prelude::*;

use crate::dispatch::cpu_op;
use crate::image::{alloc_output_pyarray, numpy_as_image, to_pyerr};
use kornia_imgproc::morphology::{self, Kernel, KernelShape};
use kornia_imgproc::padding::PaddingMode;

fn parse_kernel(shape: &str, size: (usize, usize)) -> PyResult<Kernel> {
    let (h, w) = size;
    match shape {
        "box" => {
            if h != w {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "box kernel requires a square size",
                ));
            }
            Ok(Kernel::new(KernelShape::Box { size: w }))
        }
        "cross" => {
            if h != w {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "cross kernel requires a square size",
                ));
            }
            Ok(Kernel::new(KernelShape::Cross { size: w }))
        }
        "ellipse" => Ok(Kernel::new(KernelShape::Ellipse {
            width: w,
            height: h,
        })),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown kernel shape '{other}' (expected 'box', 'cross', or 'ellipse')"
        ))),
    }
}

fn parse_border(border: &str) -> PyResult<PaddingMode> {
    Ok(match border {
        "constant" => PaddingMode::Constant,
        "replicate" => PaddingMode::Replicate,
        "reflect101" => PaddingMode::Reflect101,
        "reflect" => PaddingMode::Reflect,
        "wrap" => PaddingMode::Wrap,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown border mode '{other}' (expected 'constant', 'replicate', \
                 'reflect101', 'reflect', or 'wrap')"
            )))
        }
    })
}

macro_rules! morph_fn {
    ($name:ident, $op:path, $doc:literal) => {
        #[doc = $doc]
        ///
        /// Residency-dispatched: a u8 device `Image` (1/3/4-channel) runs the
        /// CUDA kernel — bit-identical to the CPU path — and a numpy u8 array
        /// runs the CPU path. `kernel` is `"box"`, `"cross"`, or `"ellipse"`;
        /// `size` is `(height, width)`; `border` selects the padding mode,
        /// with `constant_value` filling `"constant"` borders.
        #[pyfunction]
        #[pyo3(signature = (image, kernel="box", size=(3, 3), border="replicate", constant_value=0))]
        pub fn $name(
            py: Python<'_>,
            image: &Bound<'_, PyAny>,
            kernel: &str,
            size: (usize, usize),
            border: &str,
            constant_value: u8,
        ) -> PyResult<Py<PyAny>> {
            let se = parse_kernel(kernel, size)?;
            let mode = parse_border(border)?;

            #[cfg(feature = "cuda")]
            if let Ok(api) = image.cast::<crate::image::PyImageApi>() {
                let img = api.borrow();
                if img.is_device() {
                    return crate::cuda_ext::morphology::$name(py, &img, &se, mode, constant_value)?
                        .into_py(py);
                }
            }

            cpu_op(py, image, move |py, arr: Py<numpy::PyArray3<u8>>| {
                let c = arr.bind(py).shape()[2];
                match c {
                    1 => {
                        let src = unsafe { numpy_as_image::<1>(py, &arr)? };
                        let (mut dst, out) = unsafe { alloc_output_pyarray::<1>(py, src.size())? };
                        py.detach(|| $op(&src, &mut dst, &se, mode, [constant_value; 1]))
                            .map_err(to_pyerr)?;
                        Ok(out)
                    }
                    3 => {
                        let src = unsafe { numpy_as_image::<3>(py, &arr)? };
                        let (mut dst, out) = unsafe { alloc_output_pyarray::<3>(py, src.size())? };
                        py.detach(|| $op(&src, &mut dst, &se, mode, [constant_value; 3]))
                            .map_err(to_pyerr)?;
                        Ok(out)
                    }
                    4 => {
                        let src = unsafe { numpy_as_image::<4>(py, &arr)? };
                        let (mut dst, out) = unsafe { alloc_output_pyarray::<4>(py, src.size())? };
                        py.detach(|| $op(&src, &mut dst, &se, mode, [constant_value; 4]))
                            .map_err(to_pyerr)?;
                        Ok(out)
                    }
                    c => Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "morphology supports 1, 3, or 4 channels; got {c}"
                    ))),
                }
            })
        }
    };
}

morph_fn!(
    dilate,
    morphology::dilate,
    "Dilate an image (neighborhood maximum over the structuring element)."
);
morph_fn!(
    erode,
    morphology::erode,
    "Erode an image (neighborhood minimum over the structuring element)."
);
