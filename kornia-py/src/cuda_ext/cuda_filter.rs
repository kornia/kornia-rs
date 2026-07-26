//! Device-resident GPU separable filters (`kornia_rs.imgproc.gaussian_blur` /
//! `box_blur` / `sobel` device paths). Same shape as `cuda_morphology`:
//! allocate the destination on the source's stream and call the public
//! residency-dispatched op — routing and byte-exactness live in the crate.

use super::cuda_geometry::PyOut;
use super::*;

fn run<T, const C: usize>(
    img: &PyImageApi,
    src: &Image<T, C>,
    wrap: fn(Image<T, C>) -> Inner,
    op: impl Fn(&Image<T, C>, &mut Image<T, C>) -> Result<(), kornia_image::ImageError>,
) -> PyResult<PyOut>
where
    T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Clone + Default + 'static,
{
    let stream = source_stream(src)?;
    // SAFETY: the filter kernels write every output pixel (bounds-guarded
    // grid, all channels), so the uninitialized destination is fully
    // overwritten.
    let mut dst = unsafe { Image::<T, C>::uninit_cuda(src.size(), &stream) }.map_err(err)?;
    op(src, &mut dst).map_err(err)?;
    Ok(PyOut::New(PyImageApi::from_device(
        wrap(dst),
        img.color_space,
        device_mode::<T>(C),
    )))
}

macro_rules! blur_dev {
    ($name:ident, $op_u8:path, $op_f32:path, ($($arg:ident: $ty:ty),*)) => {
        pub(crate) fn $name(img: &PyImageApi, $($arg: $ty),*) -> PyResult<PyOut> {
            let dev = img.as_device().ok_or_else(|| {
                PyValueError::new_err(concat!(
                    stringify!($name),
                    ": expected a device Image; for a host image pass its numpy array"
                ))
            })?;
            match dev {
                Inner::U8C1(src) => run::<u8, 1>(img, src, Inner::U8C1, |s, d| $op_u8(s, d, $($arg),*)),
                Inner::U8C3(src) => run::<u8, 3>(img, src, Inner::U8C3, |s, d| $op_u8(s, d, $($arg),*)),
                Inner::U8C4(src) => run::<u8, 4>(img, src, Inner::U8C4, |s, d| $op_u8(s, d, $($arg),*)),
                Inner::F32C1(src) => run::<f32, 1>(img, src, Inner::F32C1, |s, d| $op_f32(s, d, $($arg),*)),
                Inner::F32C3(src) => run::<f32, 3>(img, src, Inner::F32C3, |s, d| $op_f32(s, d, $($arg),*)),
                other => Err(PyValueError::new_err(format!(
                    concat!(
                        stringify!($name),
                        ": the GPU path supports u8 1/3/4-channel and f32 1/3-channel \
                         device images, got {:?} with {} channel(s)"
                    ),
                    other.dtype_enum(),
                    other.channels()
                ))),
            }
        }
    };
}

blur_dev!(
    gaussian_blur,
    kornia_imgproc::filter::gaussian_blur_u8,
    kornia_imgproc::filter::gaussian_blur,
    (kernel_size: (usize, usize), sigma: (f32, f32))
);
blur_dev!(
    box_blur,
    kornia_imgproc::filter::box_blur_u8,
    kornia_imgproc::filter::box_blur,
    (kernel_size: (usize, usize))
);

/// Sobel magnitude — f32 device images only (the crate op is f32).
pub(crate) fn sobel(img: &PyImageApi, kernel_size: usize) -> PyResult<PyOut> {
    let dev = img.as_device().ok_or_else(|| {
        PyValueError::new_err(
            "sobel: expected a device Image; for a host image pass its numpy array",
        )
    })?;
    match dev {
        Inner::F32C1(src) => run::<f32, 1>(img, src, Inner::F32C1, |s, d| {
            kornia_imgproc::filter::sobel(s, d, kernel_size)
        }),
        Inner::F32C3(src) => run::<f32, 3>(img, src, Inner::F32C3, |s, d| {
            kornia_imgproc::filter::sobel(s, d, kernel_size)
        }),
        other => Err(PyValueError::new_err(format!(
            "sobel: the GPU path supports f32 1/3-channel device images, got {:?} with {} channel(s)",
            other.dtype_enum(),
            other.channels()
        ))),
    }
}

pub(crate) fn median_blur(img: &PyImageApi, kernel_size: usize) -> PyResult<PyOut> {
    let dev = img.as_device().ok_or_else(|| {
        PyValueError::new_err(
            "median_blur: expected a device Image; for a host image pass its numpy array",
        )
    })?;
    fn run<const C: usize>(
        img: &PyImageApi,
        src: &Image<u8, C>,
        ksize: usize,
        wrap: fn(Image<u8, C>) -> Inner,
    ) -> PyResult<PyOut> {
        let stream = source_stream(src)?;
        // SAFETY: the median kernel writes every output pixel (bounds-
        // guarded grid, all channels), so the uninitialized destination is
        // fully overwritten.
        let mut dst = unsafe { Image::<u8, C>::uninit_cuda(src.size(), &stream) }.map_err(err)?;
        kornia_imgproc::filter::median_blur(src, &mut dst, ksize).map_err(err)?;
        Ok(PyOut::New(PyImageApi::from_device(
            wrap(dst),
            img.color_space,
            device_mode::<u8>(C),
        )))
    }
    match dev {
        Inner::U8C1(src) => run::<1>(img, src, kernel_size, Inner::U8C1),
        Inner::U8C3(src) => run::<3>(img, src, kernel_size, Inner::U8C3),
        Inner::U8C4(src) => run::<4>(img, src, kernel_size, Inner::U8C4),
        other => Err(PyValueError::new_err(format!(
            "median_blur: the GPU path supports u8 1/3/4-channel device images, \
             got {:?} with {} channel(s)",
            other.dtype_enum(),
            other.channels(),
        ))),
    }
}

pub(crate) fn bilateral_filter(
    img: &PyImageApi,
    d: i32,
    sigma_color: f64,
    sigma_space: f64,
) -> PyResult<PyOut> {
    let dev = img.as_device().ok_or_else(|| {
        PyValueError::new_err(
            "bilateral_filter: expected a device Image; for a host image pass its numpy array",
        )
    })?;
    let Inner::U8C1(src) = dev else {
        return Err(PyValueError::new_err(format!(
            "bilateral_filter: the GPU path supports u8 single-channel device images, \
             got {:?} with {} channel(s)",
            dev.dtype_enum(),
            dev.channels(),
        )));
    };
    let stream = source_stream(src)?;
    // SAFETY: the bilateral kernel writes every output pixel (bounds-
    // guarded grid), so the uninitialized destination is fully overwritten.
    let mut dst = unsafe { Image::<u8, 1>::uninit_cuda(src.size(), &stream) }.map_err(err)?;
    kornia_imgproc::filter::bilateral_filter(src, &mut dst, d, sigma_color, sigma_space)
        .map_err(err)?;
    Ok(PyOut::New(PyImageApi::from_device(
        Inner::U8C1(dst),
        img.color_space,
        device_mode::<u8>(1),
    )))
}
