/// A module for capturing video streams from v4l2 cameras.
pub mod camera;

/// A module for capturing video streams from different sources.
pub mod capture;

/// Error types for the stream module.
pub mod error;

/// A module for capturing video streams from rtsp sources.
pub mod rtsp;

/// A module for capturing video streams from v4l cameras.
pub mod v4l2;

/// A module for capturing video streams from video files.
pub mod video;

pub use crate::stream::camera::{CameraCapture, CameraCaptureConfig};
pub use crate::stream::capture::StreamCapture;
pub use crate::stream::error::StreamCaptureError;
pub use crate::stream::rtsp::RTSPCameraConfig;
pub use crate::stream::v4l2::V4L2CameraConfig;
pub use crate::stream::video::VideoWriter;

use kornia_image::allocator::ImageAllocator;
use kornia_tensor::{
    allocator::TensorAllocatorError,
    resource::MemoryResource,
    TensorAllocator,
};

#[derive(Clone)]
/// A [TensorAllocator] used for those images, whose memory is managed by gstreamer.
///
/// NOTE: This is a transitional shim (Tasks 7-8 will replace it with `GstResource`).
/// `allocate` always returns [`TensorAllocatorError::CannotAllocateForeign`] because
/// GStreamer manages the buffer memory; use `Image::from_raw_parts` with the mapped
/// buffer pointer directly.
pub struct GstAllocator(pub gstreamer::Buffer);

impl Default for GstAllocator {
    fn default() -> Self {
        Self(gstreamer::Buffer::new())
    }
}

impl TensorAllocator for GstAllocator {
    fn allocate(
        &self,
        _layout: std::alloc::Layout,
    ) -> Result<Box<dyn MemoryResource>, TensorAllocatorError> {
        // GStreamer manages the buffer memory — allocation never happens here.
        // For more info, check https://github.com/kornia/kornia-rs/pull/338
        // TODO(Task 7): replace with GstResource that keeps the MappedBuffer alive.
        Err(TensorAllocatorError::CannotAllocateForeign)
    }
}

impl ImageAllocator for GstAllocator {}
