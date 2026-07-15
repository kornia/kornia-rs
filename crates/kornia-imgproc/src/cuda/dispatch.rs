//! Residency-aware CUDA dispatch shared by every device-capable op.
//!
//! Enabled by the `cuda` feature. High-level entry points (`ConvertColor`,
//! and the geometry ops as they gain device paths) call [`pair_residency`]
//! on their operands: host pairs run the existing CPU (NEON/AVX2) path
//! unchanged, device pairs are routed to the native CUDA launchers, and
//! mixed pairs are a typed error — there is **no implicit transfer** in
//! either direction.
//!
//! The stream used for the launch is recovered from the *source* image
//! ([`kornia_tensor::Tensor::cuda_stream`]); device images therefore must be
//! created via the typed helpers (`to_cuda` / `zeros_cuda` on `Image` or the
//! color-space newtypes) so the storage is a typed `CudaResource<T>`.

use std::sync::Arc;

use cudarc::driver::{CudaStream, DeviceRepr, ValidAsZeroBits};
use kornia_image::{Image, ImageError};
use kornia_tensor::MemoryDomain;

/// Where a (src, dst) operand pair lives.
pub(crate) enum Residency {
    /// Both operands host-resident → run the CPU path.
    Host,
    /// Both operands device-resident → launch through the [`DeviceExec`].
    Device(DeviceExec),
}

/// A checked device execution context: the stream to launch on plus the
/// cross-stream fence obligations when src and dst live on different streams.
///
/// CUDA gives no implicit ordering between streams, so for a cross-stream
/// pair [`pair_residency`] records an event on the destination's stream and
/// makes the launch stream wait on it (the destination's pending writes —
/// e.g. `zeros_cuda`'s async memset — complete first), and [`finish`]
/// records the launch on an event the destination's stream then waits on
/// (subsequent destination-stream reads see the converted pixels).
pub(crate) struct DeviceExec {
    stream: Arc<CudaStream>,
    /// Destination stream to fence back to, when different from `stream`.
    fence_back: Option<Arc<CudaStream>>,
}

impl DeviceExec {
    /// Build the execution context for a (launch, destination) stream pair —
    /// the single home of the cross-stream fence protocol. Same device
    /// required; same stream = no fence; different streams = pre-fence now
    /// (launch stream waits for the destination's pending work) and a
    /// post-fence obligation discharged by [`run`](Self::run).
    fn for_streams(launch: &Arc<CudaStream>, dst: &Arc<CudaStream>) -> Result<Self, ImageError> {
        if launch.context().ordinal() != dst.context().ordinal() {
            return Err(ImageError::DeviceMismatch);
        }
        if launch.cu_stream() == dst.cu_stream() {
            return Ok(DeviceExec {
                stream: launch.clone(),
                fence_back: None,
            });
        }
        let ev = dst.record_event(None).map_err(driver_err)?;
        launch.wait(&ev).map_err(driver_err)?;
        Ok(DeviceExec {
            stream: launch.clone(),
            fence_back: Some(dst.clone()),
        })
    }

    /// Launch through `f` and complete the cross-stream ordering — fusing the
    /// launch and the post-fence so forgetting the fence is unrepresentable.
    pub(crate) fn run(
        self,
        f: impl FnOnce(&Arc<CudaStream>) -> Result<(), ImageError>,
    ) -> Result<(), ImageError> {
        f(&self.stream)?;
        if let Some(dst_stream) = self.fence_back {
            let ev = self.stream.record_event(None).map_err(driver_err)?;
            dst_stream.wait(&ev).map_err(driver_err)?;
        }
        Ok(())
    }
}

fn driver_err(e: cudarc::driver::DriverError) -> ImageError {
    ImageError::Cuda(e.to_string())
}

/// True if the image's backing memory is device- (or unified-) resident.
///
/// Uses the storage's [`MemoryDomain`], so it is accurate even when the
/// element type does not match the stored `CudaResource<T>`.
pub(crate) fn is_device<T, const C: usize>(img: &Image<T, C>) -> bool {
    matches!(
        img.0.storage.domain(),
        MemoryDomain::Device { .. } | MemoryDomain::Unified { .. }
    )
}

/// Classify a (src, dst) pair: both-host, both-device, or error on a mix.
///
/// Device pairs must live on the **same device** (cross-device errors with
/// [`ImageError::DeviceMismatch`]). Different **streams** on the same device
/// are supported via event fences — see [`DeviceExec`]. Streams are compared
/// by raw `CUstream` handle (every `ctx.default_stream()` call returns a
/// fresh `Arc` over the same null handle, so `Arc` identity is wrong).
pub(crate) fn pair_residency<T, const C: usize, const D: usize>(
    src: &Image<T, C>,
    dst: &Image<T, D>,
) -> Result<Residency, ImageError>
where
    T: DeviceRepr + ValidAsZeroBits + 'static,
{
    match (is_device(src), is_device(dst)) {
        (false, false) => Ok(Residency::Host),
        (true, true) => {
            let s_stream = src
                .0
                .cuda_stream()
                .ok_or_else(|| untyped_device_err("source"))?;
            let d_stream = dst
                .0
                .cuda_stream()
                .ok_or_else(|| untyped_device_err("destination"))?;
            Ok(Residency::Device(DeviceExec::for_streams(
                s_stream, d_stream,
            )?))
        }
        _ => Err(ImageError::MixedResidency),
    }
}

/// Build a [`DeviceExec`] for a known-device source stream and a destination
/// image (used by non-`Image` sources like `DeviceVideoFrame`). Same
/// same-device + event-fence rules as [`pair_residency`].
pub(crate) fn device_exec_for<T>(
    src_stream: &Arc<CudaStream>,
    dst: &kornia_tensor::Tensor<T, 3>,
) -> Result<DeviceExec, ImageError>
where
    T: DeviceRepr + ValidAsZeroBits + 'static,
{
    let d_stream = dst
        .cuda_stream()
        .ok_or_else(|| untyped_device_err("destination"))?;
    DeviceExec::for_streams(src_stream, d_stream)
}

pub(crate) fn untyped_device_err(what: &str) -> ImageError {
    ImageError::Cuda(format!(
        "{what} image is device-resident but not backed by a typed CudaResource; \
         create device images via Image::to_cuda / zeros_cuda (or the \
         color-space newtype to_cuda / zeros_cuda helpers)"
    ))
}

/// Extract the typed device slices from a checked device pair.
#[macro_export]
#[doc(hidden)]
macro_rules! __device_slices {
    ($src:expr, $dst:expr) => {{
        let s = $src
            .0
            .as_cudaslice()
            .ok_or_else(|| untyped_device_err("source"))?;
        let d = $dst
            .0
            .as_cudaslice_mut()
            .ok_or_else(|| untyped_device_err("destination"))?;
        (s, d)
    }};
}

/// Extract the typed device slices from a checked device pair.
pub(crate) use crate::__device_slices as device_slices;
