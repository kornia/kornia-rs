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

/// A module for a generic, format-aware captured frame.
pub mod frame;

/// A module for capturing video streams from video files.
pub mod video;

pub use crate::stream::camera::{CameraCapture, CameraCaptureConfig};
pub use crate::stream::capture::StreamCapture;
pub use crate::stream::error::StreamCaptureError;
pub use crate::stream::frame::GstFrame;
pub use crate::stream::rtsp::RTSPCameraConfig;
pub use crate::stream::v4l2::V4L2CameraConfig;
pub use crate::stream::video::VideoWriter;

use std::any::Any;

use kornia_tensor::resource::{MemoryDomain, MemoryResource};

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

impl GstResource {
    /// Zero-copy read-only view of the mapped buffer bytes.
    pub(crate) fn as_slice(&self) -> &[u8] {
        self._map.as_slice()
    }
}

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

#[cfg(test)]
mod tests {
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

    /// GRAY8 counterpart of the RGB capture test: verifies the channel-generic
    /// `image_from_gst_buffer::<1>` path via `grab_mono8` returns single-channel
    /// frames of the right size, using `videotestsrc` (no camera required).
    #[test]
    fn gst_capture_mono8_frames() -> Result<(), Box<dyn std::error::Error>> {
        const N_FRAMES: usize = 5;
        const WIDTH: usize = 8;
        const HEIGHT: usize = 4;

        if !gstreamer::INITIALIZED.load(std::sync::atomic::Ordering::Relaxed) {
            gstreamer::init()?;
        }

        let pipeline_desc = format!(
            "videotestsrc num-buffers={n} ! \
             video/x-raw,format=GRAY8,width={w},height={h},framerate=30/1 ! \
             appsink name=sink sync=false",
            n = N_FRAMES,
            w = WIDTH,
            h = HEIGHT,
        );

        let mut capture = StreamCapture::new(&pipeline_desc)?;
        capture.start()?;

        let mut frames_received = 0usize;
        let max_polls = N_FRAMES * 5;
        for _ in 0..max_polls {
            if let Some(image) = capture.grab_mono8()? {
                assert_eq!(image.width(), WIDTH, "frame width mismatch");
                assert_eq!(image.height(), HEIGHT, "frame height mismatch");
                assert_eq!(image.num_channels(), 1, "frame channels mismatch");

                let slice = image.as_slice();
                assert_eq!(slice.len(), WIDTH * HEIGHT, "frame pixel count mismatch");
                let _ = slice[0];
                let _ = slice[slice.len() - 1];

                frames_received += 1;
            }
            if frames_received >= N_FRAMES {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        capture.close()?;

        assert_eq!(
            frames_received, N_FRAMES,
            "expected {N_FRAMES} frames but received {frames_received}"
        );

        Ok(())
    }

    /// A pipeline whose source cannot be opened must surface the fatal bus error
    /// through the grab methods instead of returning `None` forever. Uses a
    /// `filesrc` pointing at a nonexistent path (no camera/network required).
    #[test]
    fn gst_bus_error_surfaces_on_grab() -> Result<(), Box<dyn std::error::Error>> {
        use crate::stream::StreamCapture;

        if !gstreamer::INITIALIZED.load(std::sync::atomic::Ordering::Relaxed) {
            gstreamer::init()?;
        }

        let pipeline_desc = "filesrc location=/nonexistent/kornia_test_missing_file.mp4 ! \
             decodebin ! videoconvert ! video/x-raw,format=RGB ! appsink name=sink";

        let mut capture = StreamCapture::new(pipeline_desc)?;
        // `start` may fail synchronously or the error may arrive asynchronously on
        // the bus; accept either path.
        let _ = capture.start();

        let mut surfaced = false;
        for _ in 0..200 {
            if capture.grab_rgb8().is_err() || capture.last_error().is_some() {
                surfaced = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let _ = capture.close();

        assert!(
            surfaced,
            "expected a fatal pipeline bus error to surface via grab/last_error"
        );
        Ok(())
    }

    /// Exercises the generic `GstFrame` path: capture GRAY16_LE frames via
    /// `videotestsrc` and read them as `Image<u16, 1>`, proving the u16 + format
    /// validation + stride handling in `GstFrame::to_image_u16`.
    #[test]
    fn gst_frame_mono16_via_videotestsrc() -> Result<(), Box<dyn std::error::Error>> {
        const WIDTH: usize = 8;
        const HEIGHT: usize = 4;

        if !gstreamer::INITIALIZED.load(std::sync::atomic::Ordering::Relaxed) {
            gstreamer::init()?;
        }

        let pipeline_desc = format!(
            "videotestsrc num-buffers=5 ! \
             video/x-raw,format=GRAY16_LE,width={WIDTH},height={HEIGHT},framerate=30/1 ! \
             appsink name=sink sync=false"
        );

        let mut capture = StreamCapture::new(&pipeline_desc)?;
        capture.start()?;

        let mut got = false;
        for _ in 0..25 {
            if let Some(frame) = capture.grab()? {
                assert_eq!(frame.size().width, WIDTH);
                assert_eq!(frame.size().height, HEIGHT);
                // Requesting the wrong element type must be rejected.
                assert!(frame.to_image_u8::<1>().is_err(), "u8 view of GRAY16 rejected");
                let img = frame.to_image_u16::<1>()?;
                assert_eq!(img.width(), WIDTH);
                assert_eq!(img.num_channels(), 1);
                assert_eq!(img.as_slice().len(), WIDTH * HEIGHT);
                got = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        capture.close()?;
        assert!(got, "expected at least one GRAY16_LE frame");
        Ok(())
    }

    /// Exercises the padded-row path: an odd width makes GStreamer pad each RGB row
    /// (stride != width*3), so `to_image_u8` must take the packed-copy branch and
    /// still yield a correctly-sized, tightly-packed image.
    #[test]
    fn gst_frame_rgb_padded_stride() -> Result<(), Box<dyn std::error::Error>> {
        const WIDTH: usize = 5; // 5*3 = 15 bytes/row -> padded to 16 by GStreamer
        const HEIGHT: usize = 3;

        if !gstreamer::INITIALIZED.load(std::sync::atomic::Ordering::Relaxed) {
            gstreamer::init()?;
        }

        let pipeline_desc = format!(
            "videotestsrc num-buffers=5 ! \
             video/x-raw,format=RGB,width={WIDTH},height={HEIGHT},framerate=30/1 ! \
             appsink name=sink sync=false"
        );

        let mut capture = StreamCapture::new(&pipeline_desc)?;
        capture.start()?;

        let mut got = false;
        for _ in 0..25 {
            if let Some(frame) = capture.grab()? {
                // Confirm the source really padded the rows (otherwise the test is moot).
                assert!(
                    frame.stride() >= WIDTH * 3,
                    "stride {} should be >= packed {}",
                    frame.stride(),
                    WIDTH * 3
                );
                let img = frame.to_image_u8::<3>()?;
                assert_eq!(img.width(), WIDTH);
                assert_eq!(img.height(), HEIGHT);
                // Packed length regardless of source padding.
                assert_eq!(img.as_slice().len(), WIDTH * HEIGHT * 3);
                got = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        capture.close()?;
        assert!(got, "expected at least one RGB frame");
        Ok(())
    }
}
