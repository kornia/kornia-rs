//! Device-resident GPU morphology (`kornia_rs.imgproc.dilate` / `erode`
//! device path). Same shape as `cuda_geometry`: allocate the destination on
//! the source's stream, call the public residency-dispatched op.

use super::cuda_geometry::PyOut;
use super::*;
use kornia_imgproc::morphology::Kernel;
use kornia_imgproc::padding::PaddingMode;

macro_rules! morph_dev {
    ($name:ident, $op:path) => {
        pub(crate) fn $name(
            _py: Python<'_>,
            img: &PyImageApi,
            se: &Kernel,
            mode: PaddingMode,
            constant_value: u8,
        ) -> PyResult<PyOut> {
            fn run<const C: usize>(
                img: &PyImageApi,
                src: &Image<u8, C>,
                se: &Kernel,
                mode: PaddingMode,
                cval: u8,
                wrap: fn(Image<u8, C>) -> Inner,
                op: impl Fn(
                    &Image<u8, C>,
                    &mut Image<u8, C>,
                ) -> Result<(), kornia_image::ImageError>,
            ) -> PyResult<PyOut> {
                let stream = source_stream(src)?;
                // SAFETY: the morphology kernel writes every output pixel
                // (bounds-guarded grid, all channels), so the uninitialized
                // destination is fully overwritten.
                let mut dst =
                    unsafe { Image::<u8, C>::uninit_cuda(src.size(), &stream) }.map_err(err)?;
                op(src, &mut dst).map_err(err)?;
                let _ = mode;
                let _ = se;
                let _ = cval;
                Ok(PyOut::New(PyImageApi::from_device(
                    wrap(dst),
                    img.color_space,
                    device_mode::<u8>(C),
                )))
            }

            let dev = img.as_device().ok_or_else(|| {
                PyValueError::new_err(concat!(
                    stringify!($name),
                    ": expected a device Image; for a host image pass its numpy array"
                ))
            })?;
            match dev {
                Inner::U8C1(src) => run::<1>(img, src, se, mode, constant_value, Inner::U8C1, |s, d| {
                    $op(s, d, se, mode, [constant_value; 1])
                }),
                Inner::U8C3(src) => run::<3>(img, src, se, mode, constant_value, Inner::U8C3, |s, d| {
                    $op(s, d, se, mode, [constant_value; 3])
                }),
                Inner::U8C4(src) => run::<4>(img, src, se, mode, constant_value, Inner::U8C4, |s, d| {
                    $op(s, d, se, mode, [constant_value; 4])
                }),
                other => Err(PyValueError::new_err(format!(
                    concat!(
                        stringify!($name),
                        ": the GPU path supports u8 1/3/4-channel device images, \
                         got {:?} with {} channel(s)"
                    ),
                    other.dtype_enum(),
                    other.channels()
                ))),
            }
        }
    };
}

morph_dev!(dilate, kornia_imgproc::morphology::dilate);
morph_dev!(erode, kornia_imgproc::morphology::erode);
