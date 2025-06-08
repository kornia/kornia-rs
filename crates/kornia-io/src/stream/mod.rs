/// A module for capturing video streams from v4l2 cameras.
pub mod camera;

/// A module for capturing video streams from different sources.
pub mod capture;

/// Error types for the stream module.
pub mod error;

/// A module for capturing video streams from rtsp sources.
pub mod rtsp;

/// A module for capturing video streams from v4l2 cameras.
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
use kornia_tensor::{allocator::TensorAllocatorError, TensorAllocator};

#[allow(dead_code)]
#[derive(Clone)]
/// A [TensorAllocator] used for those images, whose memory is managed by gstreamer.
pub struct GstAllocator(gstreamer::Buffer);

impl Default for GstAllocator {
    fn default() -> Self {
        Self(gstreamer::Buffer::new())
    }
}

impl TensorAllocator for GstAllocator {
    fn alloc(
        &self,
        layout: std::alloc::Layout,
    ) -> Result<*mut u8, kornia_tensor::allocator::TensorAllocatorError> {
        let ptr = unsafe { std::alloc::alloc(layout) };

        if ptr.is_null() {
            Err(TensorAllocatorError::NullPointer)?
        }

        Ok(ptr)
    }

    fn dealloc(&self, _ptr: *mut u8, _layout: std::alloc::Layout) {
        // Do nothing as the memory is managed by Gstreamer
        // For more info, check https://github.com/kornia/kornia-rs/pull/338
    }
}

impl ImageAllocator for GstAllocator {}
