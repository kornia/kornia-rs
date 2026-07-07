//! Shared device-resident image representation for the Python bindings.
//!
//! [`DeviceImage`] monomorphizes the supported `(dtype, channels)` combinations
//! of a device-resident [`kornia_image::Image<T, C>`] — each backed by a typed
//! `CudaResource<T>` that carries its own `Arc<CudaStream>` internally. It is the
//! single device representation shared by the unified `Image` (via
//! [`crate::backing::Backing::Device`]) and the legacy `kornia_rs.cuda.CudaImage`.
//!
//! Enabled only under the `cuda` feature.

use std::fmt::Display;

use cudarc::driver::{DeviceRepr, ValidAsZeroBits};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use kornia_image::{Image, ImageSize};

use crate::backing::{AlignedBytes, Dtype};

fn err<E: Display>(e: E) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// Device-resident pixels (HWC), one variant per supported `(dtype, channels)`.
pub enum DeviceImage {
    U8C1(Image<u8, 1>),
    U8C3(Image<u8, 3>),
    U8C4(Image<u8, 4>),
    F32C1(Image<f32, 1>),
    F32C3(Image<f32, 3>),
}

impl DeviceImage {
    /// Pixel dimensions.
    pub fn size(&self) -> ImageSize {
        match self {
            DeviceImage::U8C1(i) => i.size(),
            DeviceImage::U8C3(i) => i.size(),
            DeviceImage::U8C4(i) => i.size(),
            DeviceImage::F32C1(i) => i.size(),
            DeviceImage::F32C3(i) => i.size(),
        }
    }

    /// Channel count (1, 3, or 4).
    pub fn channels(&self) -> usize {
        match self {
            DeviceImage::U8C1(_) | DeviceImage::F32C1(_) => 1,
            DeviceImage::U8C3(_) | DeviceImage::F32C3(_) => 3,
            DeviceImage::U8C4(_) => 4,
        }
    }

    /// Element dtype as the backing [`Dtype`] enum.
    pub fn dtype_enum(&self) -> Dtype {
        match self {
            DeviceImage::U8C1(_) | DeviceImage::U8C3(_) | DeviceImage::U8C4(_) => Dtype::U8,
            DeviceImage::F32C1(_) | DeviceImage::F32C3(_) => Dtype::F32,
        }
    }

    /// HWC shape `[height, width, channels]`.
    pub fn shape_hwc(&self) -> [usize; 3] {
        let s = self.size();
        [s.height, s.width, self.channels()]
    }

    /// CUDA device ordinal this image is resident on.
    pub fn device_id(&self) -> i32 {
        match self {
            DeviceImage::U8C1(i) => i.0.storage.device_id(),
            DeviceImage::U8C3(i) => i.0.storage.device_id(),
            DeviceImage::U8C4(i) => i.0.storage.device_id(),
            DeviceImage::F32C1(i) => i.0.storage.device_id(),
            DeviceImage::F32C3(i) => i.0.storage.device_id(),
        }
    }

    /// Raw `CUstream` handle (address) of the stream this image was produced on,
    /// for the CUDA Array Interface / cuda-python stream handshake. `None` if the
    /// backing does not carry a stream.
    pub fn stream_ptr(&self) -> Option<usize> {
        let s = match self {
            DeviceImage::U8C1(i) => i.0.cuda_stream(),
            DeviceImage::U8C3(i) => i.0.cuda_stream(),
            DeviceImage::U8C4(i) => i.0.cuda_stream(),
            DeviceImage::F32C1(i) => i.0.cuda_stream(),
            DeviceImage::F32C3(i) => i.0.cuda_stream(),
        };
        s.map(|s| s.cu_stream() as usize)
    }

    /// Block until this image's producing stream has completed, so a consumer
    /// (DLPack export) sees fully-written pixels.
    pub fn synchronize(&self) -> PyResult<()> {
        let s = match self {
            DeviceImage::U8C1(i) => i.0.cuda_stream(),
            DeviceImage::U8C3(i) => i.0.cuda_stream(),
            DeviceImage::U8C4(i) => i.0.cuda_stream(),
            DeviceImage::F32C1(i) => i.0.cuda_stream(),
            DeviceImage::F32C3(i) => i.0.cuda_stream(),
        };
        if let Some(s) = s {
            s.synchronize().map_err(err)?;
        }
        Ok(())
    }

    /// Raw device pointer (`CUdeviceptr` as `*mut u8`) — for DLPack export and
    /// the `data_ptr` getter. Never dereferenced on the host.
    pub fn as_ptr(&self) -> *mut u8 {
        match self {
            DeviceImage::U8C1(i) => i.0.as_ptr() as *mut u8,
            DeviceImage::U8C3(i) => i.0.as_ptr() as *mut u8,
            DeviceImage::U8C4(i) => i.0.as_ptr() as *mut u8,
            DeviceImage::F32C1(i) => i.0.as_ptr() as *mut u8,
            DeviceImage::F32C3(i) => i.0.as_ptr() as *mut u8,
        }
    }

    /// Copy this device image back to host (D2H) into an owned, 64-byte-aligned
    /// buffer, returning `(bytes, hwc_shape, dtype)`. The GIL is released for the
    /// duration of the copy + synchronize.
    pub fn download_to_owned(
        &self,
        py: Python<'_>,
    ) -> PyResult<(AlignedBytes, [usize; 3], Dtype)> {
        py.detach(|| match self {
            DeviceImage::U8C1(i) => dl_owned(i, Dtype::U8),
            DeviceImage::U8C3(i) => dl_owned(i, Dtype::U8),
            DeviceImage::U8C4(i) => dl_owned(i, Dtype::U8),
            DeviceImage::F32C1(i) => dl_owned(i, Dtype::F32),
            DeviceImage::F32C3(i) => dl_owned(i, Dtype::F32),
        })
    }
}

/// D2H a single typed device image into an owned aligned host byte buffer.
fn dl_owned<T, const C: usize>(
    img: &Image<T, C>,
    dt: Dtype,
) -> PyResult<(AlignedBytes, [usize; 3], Dtype)>
where
    T: DeviceRepr + ValidAsZeroBits + Clone + Default + 'static,
{
    let host = img.download().map_err(err)?;
    let (h, w) = (host.height(), host.width());
    let s = host.as_slice();
    // SAFETY: `s` is a contiguous host slice of `T`; reinterpret as its bytes.
    let bytes =
        unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) };
    Ok((AlignedBytes::from_slice(bytes), [h, w, C], dt))
}
