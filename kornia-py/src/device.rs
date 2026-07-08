//! Shared device-resident image representation for the Python bindings.
//!
//! [`DeviceImage`] monomorphizes the supported `(dtype, channels)` combinations
//! of a device-resident [`kornia_image::Image<T, C>`] — each backed by a typed
//! `CudaResource<T>` that carries its own `Arc<CudaStream>` internally. It is the
//! single device representation behind the unified `Image`
//! ([`crate::backing::Backing::Device`]): a device-resident `Image` is the same
//! Python type as a host one, distinguished only by `.device`, and the
//! `kornia_rs.cuda` module's color/preprocessing ops read and produce it too.
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

/// Bind the inner typed `Image<T, C>` of any variant as `$i` and evaluate
/// `$body` — collapses the otherwise-identical 5-arm variant match that every
/// type-agnostic accessor would spell out by hand.
macro_rules! dispatch {
    ($self:expr, $i:ident => $body:expr) => {
        match $self {
            DeviceImage::U8C1($i) => $body,
            DeviceImage::U8C3($i) => $body,
            DeviceImage::U8C4($i) => $body,
            DeviceImage::F32C1($i) => $body,
            DeviceImage::F32C3($i) => $body,
        }
    };
}

impl DeviceImage {
    /// Pixel dimensions.
    pub fn size(&self) -> ImageSize {
        dispatch!(self, i => i.size())
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
        dispatch!(self, i => i.0.storage.device_id())
    }

    /// The `Arc<CudaStream>` this image was produced on, if the backing carries
    /// one. Shared by [`Self::stream_ptr`], [`Self::synchronize`], and the DLPack
    /// export's non-blocking consumer-stream fence.
    pub(crate) fn cuda_stream(&self) -> Option<&std::sync::Arc<cudarc::driver::CudaStream>> {
        dispatch!(self, i => i.0.cuda_stream())
    }

    /// Raw `CUstream` handle (address) of the stream this image was produced on,
    /// for the CUDA Array Interface / cuda-python stream handshake. `None` if the
    /// backing does not carry a stream.
    pub fn stream_ptr(&self) -> Option<usize> {
        self.cuda_stream().map(|s| s.cu_stream() as usize)
    }

    /// Block until this image's producing stream has completed, so a consumer
    /// (DLPack export) sees fully-written pixels.
    pub fn synchronize(&self) -> PyResult<()> {
        if let Some(s) = self.cuda_stream() {
            s.synchronize().map_err(err)?;
        }
        Ok(())
    }

    /// Raw device pointer (`CUdeviceptr` as `*mut u8`) — for DLPack export and
    /// the `data_ptr` getter. Never dereferenced on the host.
    pub fn as_ptr(&self) -> *mut u8 {
        dispatch!(self, i => i.0.as_ptr() as *mut u8)
    }

    /// Copy this device image back to host (D2H) into an owned, 64-byte-aligned
    /// buffer, returning `(bytes, hwc_shape, dtype)`. The GIL is released for the
    /// duration of the copy + synchronize.
    pub fn download_to_owned(
        &self,
        py: Python<'_>,
    ) -> PyResult<(AlignedBytes, [usize; 3], Dtype)> {
        let dt = self.dtype_enum();
        py.detach(|| dispatch!(self, i => dl_owned(i, dt)))
    }
}

/// D2H a single typed device image into an owned aligned host byte buffer.
///
/// The copy targets the final aligned buffer directly (via
/// [`Image::to_host_into`]), so there is no intermediate `Vec<T>` + second host
/// memcpy — a device→host round-trip is a single DMA into the numpy-backing
/// storage.
fn dl_owned<T, const C: usize>(
    img: &Image<T, C>,
    dt: Dtype,
) -> PyResult<(AlignedBytes, [usize; 3], Dtype)>
where
    T: DeviceRepr + ValidAsZeroBits + Clone + Default + 'static,
{
    let s = img.size();
    let (h, w) = (s.height, s.width);
    let numel = h * w * C;
    let mut bytes = AlignedBytes::uninit(numel * std::mem::size_of::<T>());
    // SAFETY: `bytes` owns `numel * size_of::<T>()` bytes, 64-byte aligned (>=
    // align_of::<T>() for u8/f32); reinterpret as a `&mut [T]` of exactly `numel`
    // elements. `to_host_into` writes every element (a full D2H copy) before any
    // read, satisfying the uninit-buffer contract.
    let dst = unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut T, numel) };
    img.to_host_into(dst).map_err(err)?;
    Ok((bytes, [h, w, C], dt))
}
