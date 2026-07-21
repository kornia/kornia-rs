use pyo3::prelude::*;
use numpy::PyArrayMethods;

use crate::dispatch::cpu_op;
use crate::image::{
    alloc_output_pyarray, alloc_output_pyarray_f32, numpy_as_image, numpy_as_image_f32, to_pyerr,
    PyImageApi,
};
use kornia_image::{Image, ImageError, ImageSize};
use kornia_imgproc::pyramid;

#[cfg(feature = "cuda")]
use crate::cuda_ext::{device_mode, source_stream};
#[cfg(feature = "cuda")]
use crate::device::DeviceImage;

fn pyrdown_size(s: (usize, usize, usize)) -> (usize, usize, usize) {
    ((s.0 + 1) / 2, (s.1 + 1) / 2, s.2)
}

fn pyrup_size(s: (usize, usize, usize)) -> (usize, usize, usize) {
    (s.0 * 2, s.1 * 2, s.2)
}

#[cfg(feature = "cuda")]
fn run_dev<T, const C: usize>(
    img: &PyImageApi,
    src: &Image<T, C>,
    wrap: fn(Image<T, C>) -> DeviceImage,
    op: fn(&Image<T, C>, &mut Image<T, C>) -> Result<(), ImageError>,
    out_shape: (usize, usize, usize),
) -> PyResult<Py<PyAny>>
where
    T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Clone + Default + 'static,
{
    let stream = source_stream(src)?;
    let out_size = ImageSize {
        width: out_shape.1,
        height: out_shape.0,
    };
    let mut dst = unsafe { Image::<T, C>::uninit_cuda(out_size, &stream) }.map_err(to_pyerr)?;
    op(src, &mut dst).map_err(to_pyerr)?;
    let api = PyImageApi::from_device(wrap(dst), img.color_space, device_mode::<T>(C));
    Python::with_gil(|py| Ok(Py::new(py, api)?.into_any()))
}

fn run_cpu_u8<const C: usize>(
    py: Python<'_>,
    arr: &Py<numpy::PyArray3<u8>>,
    op: fn(&Image<u8, C>, &mut Image<u8, C>) -> Result<(), ImageError>,
    size_fn: fn((usize, usize, usize)) -> (usize, usize, usize),
) -> PyResult<Py<numpy::PyArray3<u8>>> {
    let shape = arr.bind(py).shape();
    let src = unsafe { numpy_as_image::<C>(py, arr)? };
    let out_shape = size_fn((shape[0], shape[1], shape[2]));
    let out_size = ImageSize {
        width: out_shape.1,
        height: out_shape.0,
    };
    let (mut dst, out) = unsafe { alloc_output_pyarray::<C>(py, out_size)? };
    py.detach(|| op(&src, &mut dst)).map_err(to_pyerr)?;
    Ok(out)
}

fn run_cpu_f32<const C: usize>(
    py: Python<'_>,
    arr: &Py<numpy::PyArray3<f32>>,
    op: fn(&Image<f32, C>, &mut Image<f32, C>) -> Result<(), ImageError>,
    size_fn: fn((usize, usize, usize)) -> (usize, usize, usize),
) -> PyResult<Py<numpy::PyArray3<f32>>> {
    let shape = arr.bind(py).shape();
    let src = unsafe { numpy_as_image_f32::<C>(py, arr)? };
    let out_shape = size_fn((shape[0], shape[1], shape[2]));
    let out_size = ImageSize {
        width: out_shape.1,
        height: out_shape.0,
    };
    let (mut dst, out) = unsafe { alloc_output_pyarray_f32::<C>(py, out_size)? };
    py.detach(|| op(&src, &mut dst)).map_err(to_pyerr)?;
    Ok(out)
}

macro_rules! py_pyramid_op {
    ($name:ident, $op_u8:path, $op_f32:path, $doc:literal, $size_fn:ident) => {
        #[doc = $doc]
        #[pyfunction]
        pub fn $name(py: Python<'_>, image: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
            #[cfg(feature = "cuda")]
            if let Ok(api) = image.cast::<PyImageApi>() {
                let img = api.borrow();
                if img.is_device() {
                    let dev = img.as_device().ok_or_else(|| {
                        pyo3::exceptions::PyRuntimeError::new_err("is_device was true but as_device returned None")
                    })?;
                    let out_shape = $size_fn(img.shape_hwc());
                    return match dev {
                        DeviceImage::U8C1(src) => run_dev(&img, src, DeviceImage::U8C1, $op_u8, out_shape),
                        DeviceImage::U8C3(src) => run_dev(&img, src, DeviceImage::U8C3, $op_u8, out_shape),
                        DeviceImage::U8C4(src) => run_dev(&img, src, DeviceImage::U8C4, $op_u8, out_shape),
                        DeviceImage::F32C1(src) => run_dev(&img, src, DeviceImage::F32C1, $op_f32, out_shape),
                        DeviceImage::F32C3(src) => run_dev(&img, src, DeviceImage::F32C3, $op_f32, out_shape),
                        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                            concat!(stringify!($name), ": the GPU path supports u8 1/3/4-channel and f32 1/3-channel device images, got dtype={:?}, channels={}"),
                            other.dtype_enum(),
                            other.channels()
                        ))),
                    };
                }
            }

            let view = if let Ok(api) = image.cast::<PyImageApi>() {
                crate::dispatch::no_gpu_kernel_if_device(&api.borrow())?;
                api.call_method0("numpy")?
            } else {
                image.clone()
            };

            use pyo3::types::PyAnyMethods;
            let dtype = view.getattr("dtype")?.getattr("name")?.extract::<String>()?;
            match dtype.as_str() {
                "uint8" => {
                    cpu_op(py, image, |py, image| {
                        let c = image.bind(py).shape()[2];
                        match c {
                            1 => run_cpu_u8::<1>(py, &image, $op_u8, $size_fn),
                            3 => run_cpu_u8::<3>(py, &image, $op_u8, $size_fn),
                            4 => run_cpu_u8::<4>(py, &image, $op_u8, $size_fn),
                            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                                concat!(stringify!($name), ": CPU path supports u8 1/3/4 channels, got {}"), c
                            )))
                        }
                    })
                }
                "float32" => {
                    cpu_op(py, image, |py, image| {
                        let c = image.bind(py).shape()[2];
                        match c {
                            1 => run_cpu_f32::<1>(py, &image, $op_f32, $size_fn),
                            3 => run_cpu_f32::<3>(py, &image, $op_f32, $size_fn),
                            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                                concat!(stringify!($name), ": CPU path supports f32 1/3 channels, got {}"), c
                            )))
                        }
                    })
                }
                other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                    concat!(stringify!($name), ": supports uint8 or float32, got dtype={}"), other
                )))
            }
        }
    };
}

py_pyramid_op!(
    pyrdown,
    pyramid::pyrdown_u8,
    pyramid::pyrdown_f32,
    "Downsample an image by 2x using a Gaussian filter.",
    pyrdown_size
);

py_pyramid_op!(
    pyrup,
    pyramid::pyrup_u8,
    pyramid::pyrup_f32,
    "Upsample an image by 2x using a Gaussian filter.",
    pyrup_size
);

#[pyfunction]
#[pyo3(signature = (image, max_level))]
pub fn build_pyramid(
    py: Python<'_>,
    image: &Bound<'_, PyAny>,
    max_level: usize,
) -> PyResult<Vec<Py<PyAny>>> {
    let mut out = Vec::with_capacity(max_level + 1);
    out.push(image.clone().unbind());
    let mut current = image.clone();
    for _ in 0..max_level {
        current = pyrdown(py, &current)?.into_bound(py);
        out.push(current.clone().unbind());
    }
    Ok(out)
}
