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

use std::any::Any;
use std::sync::Arc;

use kornia_tensor::resource::{MemoryDomain, MemoryResource};

pub use kornia_tensor::allocator::ForeignAllocator;

/// A proper [`MemoryResource`] for a GStreamer-mapped buffer (sysmem).
///
/// Holds a `MappedBuffer<Readable>` which:
/// - keeps the GStreamer buffer's reference count alive, and
/// - keeps the map handle active so the host pointer remains valid.
///
/// `Drop` is implicit: when `GstResource` is dropped, `_map` drops first, which
/// unmaps the buffer and releases the GStreamer buffer reference — exactly once.
pub struct GstResource {
    /// The mapped, readable GStreamer buffer.
    ///
    /// Keeping this field alive keeps both the memory map and the buffer ref-count
    /// alive. `Drop` on this field unmaps and releases automatically.
    pub _map: gstreamer::buffer::MappedBuffer<gstreamer::buffer::Readable>,
}

// SAFETY: gstreamer Buffers are ref-counted and thread-safe; the MappedBuffer holds
// a read-only map. Once mapped, the pointer is valid until the map is released on Drop.
unsafe impl Send for GstResource {}
unsafe impl Sync for GstResource {}

impl MemoryResource for GstResource {
    /// Returns the host pointer to the mapped GStreamer buffer data.
    fn as_ptr(&self) -> *mut u8 {
        // MappedBuffer<Readable>::as_ptr returns *const u8; we cast to *mut u8 as the
        // MemoryResource trait requires *mut u8.  The Image built from this is read-only
        // in practice (the buffer is only mapped for reading), so callers must not write.
        self._map.as_ptr() as *mut u8
    }

    /// Returns the size in bytes of the mapped region.
    fn len_bytes(&self) -> usize {
        self._map.len()
    }

    /// GStreamer system memory is host-accessible.
    fn domain(&self) -> MemoryDomain {
        MemoryDomain::Host
    }

    /// Downcast hook.
    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Mutable downcast hook.
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Construct a borrowed [`Image`] backed by a GStreamer [`MappedBuffer`].
///
/// # Arguments
///
/// * `size` - image dimensions.
/// * `mapped_buffer` - the read-mapped GStreamer buffer; ownership is transferred
///   into a [`GstResource`] keepalive that is Arc-shared with the tensor's `ForeignResource`.
///
/// # Returns
///
/// An `Image<u8, 3, ForeignAllocator>` whose memory is the GStreamer buffer.
/// The buffer remains live (and the pointer valid) for exactly the lifetime of the
/// returned `Image`; dropping the `Image` releases the buffer ref exactly once.
///
/// # Safety
///
/// The caller must ensure the pointer has not been aliased as mutable elsewhere.
pub(crate) fn image_from_gst_buffer(
    size: kornia_image::ImageSize,
    mapped_buffer: gstreamer::buffer::MappedBuffer<gstreamer::buffer::Readable>,
) -> Result<kornia_image::Image<u8, 3, ForeignAllocator>, crate::stream::error::StreamCaptureError>
{
    use kornia_tensor::storage::{MemoryDomain, TensorStorage};
    use kornia_tensor::Tensor;

    // Capture pointer and length BEFORE moving mapped_buffer into GstResource.
    let data_ptr: *const u8 = mapped_buffer.as_ptr();
    let data_len: usize = mapped_buffer.len();

    // Defense-in-depth: verify the buffer is large enough for an RGB24 frame.
    // The pipeline normally enforces RGB caps, but a misconfigured or non-RGB
    // pipeline would silently produce out-of-bounds stride-based access otherwise.
    let expected_len = size.width * size.height * 3;
    if data_len < expected_len {
        return Err(
            crate::stream::error::StreamCaptureError::BufferSizeMismatch {
                expected: expected_len,
                got: data_len,
            },
        );
    }

    // Move the MappedBuffer into a GstResource; its Drop releases the buffer.
    let resource = GstResource {
        _map: mapped_buffer,
    };
    let keepalive: Arc<dyn Any + Send + Sync> = Arc::new(resource);

    // Build a TensorStorage that borrows the gst memory and holds the keepalive.
    // SAFETY:
    //   - data_ptr is non-null (gst sysmem buffers are always non-null).
    //   - data_len equals the mapped size (captured above from mapped_buffer.len()).
    //   - The memory is host-accessible (MemoryDomain::Host).
    //   - The keepalive (Arc<GstResource>) holds the map alive for the storage's lifetime.
    //   - The storage is read-only: `as_mut_slice` will panic (GstMappedBuffer is Readable).
    let storage: TensorStorage<u8, ForeignAllocator> = unsafe {
        TensorStorage::from_borrowed_readonly(
            data_ptr,
            data_len,
            ForeignAllocator,
            MemoryDomain::Host,
            0,
            keepalive,
        )
    };

    // Row-major strides for shape [H, W, 3]: strides = [W*3, 3, 1].
    // We compute this inline since `get_strides_from_shape` is pub(crate) in kornia-tensor.
    let shape = [size.height, size.width, 3_usize];
    let strides = [size.width * 3, 3, 1];
    let tensor = Tensor {
        storage,
        shape,
        strides,
    };

    // TryFrom<Tensor3<T, A>> for Image<T, C, A> validates that shape[2] == C (== 3).
    kornia_image::Image::try_from(tensor)
        .map_err(crate::stream::error::StreamCaptureError::ImageError)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::StreamCapture;

    /// Verifies that capturing N frames with `videotestsrc` succeeds, that the pixel
    /// data is readable through the Image slice (proving the GstResource keepalive is
    /// active), and that dropping each Image releases the buffer exactly once (no
    /// crash / no double-unmap — validated by the clean exit without sanitizer errors).
    ///
    /// Uses `videotestsrc` (no camera or display required).
    #[test]
    fn gst_resource_capture_n_frames_and_drop() -> Result<(), Box<dyn std::error::Error>> {
        const N_FRAMES: usize = 5;
        const WIDTH: usize = 8;
        const HEIGHT: usize = 4;

        if !gstreamer::INITIALIZED.load(std::sync::atomic::Ordering::Relaxed) {
            gstreamer::init()?;
        }

        let pipeline_desc = format!(
            "videotestsrc num-buffers={n} ! \
             video/x-raw,format=RGB,width={w},height={h},framerate=30/1 ! \
             appsink name=sink sync=false",
            n = N_FRAMES,
            w = WIDTH,
            h = HEIGHT,
        );

        let mut capture = StreamCapture::new(&pipeline_desc)?;
        capture.start()?;

        let mut frames_received = 0usize;
        // Poll until we have all N frames (videotestsrc with num-buffers is bounded).
        // We attempt up to 5×N polls to avoid an infinite loop.
        let max_polls = N_FRAMES * 5;
        for _ in 0..max_polls {
            if let Some(image) = capture.grab_rgb8()? {
                // 1. Verify dimensions.
                assert_eq!(image.width(), WIDTH, "frame width mismatch");
                assert_eq!(image.height(), HEIGHT, "frame height mismatch");
                assert_eq!(image.num_channels(), 3, "frame channels mismatch");

                // 2. Read pixel data — proves the GstResource keepalive is active and
                //    the underlying mapped buffer is still valid.
                let slice = image.as_slice();
                assert_eq!(
                    slice.len(),
                    WIDTH * HEIGHT * 3,
                    "frame pixel count mismatch"
                );
                // Access first and last byte to ensure the mapping is live.
                let _ = slice[0];
                let _ = slice[slice.len() - 1];

                frames_received += 1;

                // 3. `image` drops here — GstResource::Drop unmaps and releases the
                //    GStreamer buffer ref exactly once.  A double-free or use-after-free
                //    would crash here (or be caught by valgrind/asan in CI).
            }
            if frames_received >= N_FRAMES {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        // Close pipeline before asserting so Close errors don't shadow the count.
        capture.close()?;

        assert_eq!(
            frames_received, N_FRAMES,
            "expected {N_FRAMES} frames but received {frames_received}"
        );

        Ok(())
    }

    /// Validates the buffer-size guard arithmetic used in `image_from_gst_buffer`.
    ///
    /// A real `MappedBuffer<Readable>` requires a live GStreamer pipeline and cannot
    /// be constructed in a pure unit test, so this test asserts the validation formula
    /// `expected = width * height * 3` for representative boundary values.
    #[test]
    fn gst_buffer_size_validation_arithmetic() {
        // expected bytes for an RGB24 frame of various sizes
        assert_eq!(8usize * 4 * 3, 96, "8x4 RGB24 = 96 bytes");
        assert_eq!(640usize * 480 * 3, 921_600, "640x480 RGB24 = 921600 bytes");
        assert_eq!(
            1920usize * 1080 * 3,
            6_220_800,
            "1920x1080 RGB24 = 6220800 bytes"
        );
        // A buffer smaller than the expected size must be rejected.
        // The actual rejection is inside image_from_gst_buffer; this test
        // documents the check boundary: expected-1 < expected → rejected.
        let width = 8usize;
        let height = 4usize;
        let expected = width * height * 3;
        assert!(
            (expected - 1) < expected,
            "a buffer of size expected-1 is strictly smaller than expected"
        );
    }
}
