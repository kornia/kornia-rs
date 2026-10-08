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

/// Quotes a user-supplied value for interpolation into a `gst_parse_launch` pipeline string.
///
/// Values are wrapped in double quotes so whitespace, `!` and `=` cannot introduce new elements or
/// properties. Characters that would terminate or escape the quoted string (`"`, `\`) and control
/// characters are rejected outright.
///
/// # Errors
///
/// Returns [`StreamCaptureError::InvalidConfig`] if `value` contains `"`, `\` or a control
/// character.
pub(crate) fn quote_pipeline_value(value: &str) -> Result<String, StreamCaptureError> {
    if value
        .chars()
        .any(|c| c == '"' || c == '\\' || c.is_control())
    {
        return Err(StreamCaptureError::InvalidConfig(format!(
            "value {value:?} contains characters not allowed in a pipeline description"
        )));
    }
    Ok(format!("\"{value}\""))
}

/// Sets the `location` property of the element called `name` in `pipeline` to `path`.
///
/// File paths are set as a property after parsing instead of being interpolated into the
/// `gst_parse_launch` description, so any path (spaces, `!`, `"`, Windows `\`
/// separators, ...) works verbatim and can never inject pipeline syntax.
///
/// # Errors
///
/// Returns [`StreamCaptureError::GetElementByNameError`] if the pipeline has no element
/// called `name`, or [`StreamCaptureError::InvalidConfig`] if that element has no
/// writable string `location` property.
pub(crate) fn set_location_property(
    pipeline: &gstreamer::Pipeline,
    name: &str,
    path: &std::path::Path,
) -> Result<(), StreamCaptureError> {
    use gstreamer::prelude::*;
    let element = pipeline
        .by_name(name)
        .ok_or(StreamCaptureError::GetElementByNameError)?;
    // `set_property` panics on a missing, read-only or non-string property.
    let writable_string = element.find_property("location").is_some_and(|p| {
        p.value_type() == String::static_type()
            && p.flags().contains(gstreamer::glib::ParamFlags::WRITABLE)
    });
    if !writable_string {
        return Err(StreamCaptureError::InvalidConfig(format!(
            "element {name:?} has no writable string `location` property"
        )));
    }
    element.set_property("location", path.to_string_lossy().as_ref());
    Ok(())
}

use kornia_image::Image;
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

/// Construct an RGB [`Image`] from a GStreamer [`MappedBuffer`].
///
/// GStreamer may pad every row so that it starts on an aligned address. For RGB, a row of
/// `width * 3` bytes is rounded up to a multiple of 4, so an 854-pixel row is 2564 bytes in the
/// buffer instead of 2562. `stride` is the real row length in bytes and `offset` is where the
/// first row starts, both taken from the frame's `VideoMeta` or `VideoInfo`.
///
/// When the rows are already tightly packed (`offset == 0` and `stride == width * 3`) the
/// image borrows the GStreamer buffer without copying. Otherwise the pixels of each row are
/// copied into a new packed image and the padding bytes are skipped.
///
/// # Arguments
///
/// * `size` - image dimensions.
/// * `stride` - number of bytes between the start of two consecutive rows in the buffer.
/// * `offset` - byte offset of the first row in the buffer.
/// * `mapped_buffer` - the read-mapped GStreamer buffer. In the zero-copy case its ownership
///   moves into a [`GstResource`] keepalive that is Arc-shared with the tensor's
///   `ForeignResource`.
///
/// # Returns
///
/// An `Image<u8, 3>` with tightly packed rows.
///
/// # Errors
///
/// Returns [`StreamCaptureError::InvalidImageFormat`] if `stride` is smaller than one row of
/// pixels or the size overflows, and [`StreamCaptureError::BufferSizeMismatch`] if the buffer
/// is too small to hold `size.height` rows with the given `stride` and `offset`.
pub(crate) fn image_from_gst_buffer(
    size: kornia_image::ImageSize,
    stride: usize,
    offset: usize,
    mapped_buffer: gstreamer::buffer::MappedBuffer<gstreamer::buffer::Readable>,
) -> Result<kornia_image::Image<u8, 3>, StreamCaptureError> {
    let row_bytes = size.width.checked_mul(3).ok_or_else(|| {
        StreamCaptureError::InvalidImageFormat(format!(
            "frame dimensions overflow: {}x{}",
            size.width, size.height
        ))
    })?;
    if stride < row_bytes {
        return Err(StreamCaptureError::InvalidImageFormat(format!(
            "row stride {stride} is smaller than width * 3 = {row_bytes}"
        )));
    }

    // Bytes needed to read every row: all rows but the last are a full stride, the last row
    // only needs its pixels (GStreamer may omit the trailing padding).
    let required_len = match size.height.checked_sub(1) {
        None => 0,
        Some(last_row) => last_row
            .checked_mul(stride)
            .and_then(|n| n.checked_add(row_bytes))
            .and_then(|n| n.checked_add(offset))
            .ok_or_else(|| {
                StreamCaptureError::InvalidImageFormat(format!(
                    "frame layout overflows: {}x{} with stride {stride}",
                    size.width, size.height
                ))
            })?,
    };
    let data_len = mapped_buffer.len();
    if data_len < required_len {
        return Err(StreamCaptureError::BufferSizeMismatch {
            expected: required_len,
            got: data_len,
        });
    }

    if offset != 0 || stride != row_bytes {
        let packed = pack_rows(
            mapped_buffer.as_slice(),
            offset,
            stride,
            row_bytes,
            size.height,
        );
        return Image::<u8, 3>::new(size, packed).map_err(StreamCaptureError::ImageError);
    }

    // Capture the pointer BEFORE moving mapped_buffer into GstResource.
    let data_ptr: *const u8 = mapped_buffer.as_ptr();

    // Move the MappedBuffer into a GstResource; its Drop releases the buffer.
    let resource = GstResource {
        _map: mapped_buffer,
    };
    let keepalive: Arc<dyn Any + Send + Sync> = Arc::new(resource);

    // SAFETY:
    // - `data_ptr` is non-null as GStreamer sysmem buffers are always non-null.
    // - The rows are tightly packed (offset 0, stride == width * 3) and we verified
    //   `data_len >= width * height * 3` above, preventing out-of-bounds reads.
    // - `keepalive` (GstResource) holds the map alive for the lifetime of the Image.
    let image = unsafe {
        Image::<u8, 3>::from_borrowed_host_readonly(size, data_ptr, keepalive)
            .map_err(StreamCaptureError::ImageError)?
    };

    Ok(image)
}

/// Copies `height` rows of `row_bytes` bytes out of a padded buffer into a packed `Vec`.
///
/// Row `i` starts at `offset + i * stride` in `data`; the `stride - row_bytes` padding bytes
/// after each row are skipped. The caller must have checked that `data` holds every row.
fn pack_rows(
    data: &[u8],
    offset: usize,
    stride: usize,
    row_bytes: usize,
    height: usize,
) -> Vec<u8> {
    if height == 0 || row_bytes == 0 {
        return Vec::new();
    }
    let mut packed = Vec::with_capacity(row_bytes * height);
    for row in data[offset..].chunks(stride).take(height) {
        packed.extend_from_slice(&row[..row_bytes]);
    }
    packed
}

#[cfg(test)]
mod tests {
    use crate::stream::StreamCapture;

    /// A quoted value containing pipeline syntax must be parsed as a single property value,
    /// not as additional elements.
    #[test]
    fn quoted_value_cannot_inject_elements() -> Result<(), Box<dyn std::error::Error>> {
        use gstreamer::prelude::*;
        gstreamer::init()?;
        let evil = "/tmp/a b ! fakesink name=injected location=x";
        let desc = format!(
            "filesrc name=src location={}",
            super::quote_pipeline_value(evil)?
        );
        let bin = gstreamer::parse::launch(&desc)?
            .dynamic_cast::<gstreamer::Bin>()
            .ok();
        // A single element is returned as-is rather than wrapped in a bin.
        assert!(bin.is_none(), "value was split into multiple elements");
        let src = gstreamer::parse::launch(&desc)?;
        assert_eq!(
            src.property::<Option<String>>("location").as_deref(),
            Some(evil)
        );

        assert!(super::quote_pipeline_value("a\" ! fakesink").is_err());
        assert!(super::quote_pipeline_value("a\\").is_err());
        assert!(super::quote_pipeline_value("a\nb").is_err());
        Ok(())
    }

    /// File paths are set as a property, so characters that `quote_pipeline_value`
    /// rejects (e.g. Windows `\` separators) or that are pipeline syntax round-trip.
    #[test]
    fn location_property_accepts_any_path() -> Result<(), Box<dyn std::error::Error>> {
        use gstreamer::prelude::*;
        gstreamer::init()?;
        let path = std::path::Path::new(r#"C:\videos\my clip ! fakesink name=x "q".mp4"#);
        let pipeline = gstreamer::parse::launch("filesrc name=src ! fakesink")?
            .dynamic_cast::<gstreamer::Pipeline>()
            .map_err(|_| "not a pipeline")?;
        super::set_location_property(&pipeline, "src", path)?;
        let src = pipeline.by_name("src").ok_or("missing src")?;
        assert_eq!(
            src.property::<Option<String>>("location").as_deref(),
            path.to_str()
        );
        // Only the two parsed elements exist; nothing was injected.
        assert_eq!(pipeline.children().len(), 2);
        assert!(super::set_location_property(&pipeline, "missing", path).is_err());
        // An element without a `location` property is an error, not a panic.
        let fakesink = pipeline
            .children()
            .into_iter()
            .find(|e| e.name() != "src")
            .ok_or("missing fakesink")?;
        assert!(super::set_location_property(&pipeline, &fakesink.name(), path).is_err());
        Ok(())
    }

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

    /// `pack_rows` must drop the padding after every row and keep the pixel bytes in order.
    #[test]
    fn pack_rows_skips_row_padding() {
        // 2x3 RGB image: 6 pixel bytes per row, padded to a stride of 8, starting at offset 1.
        let data: Vec<u8> = vec![
            99, // offset byte before the first row
            1, 2, 3, 4, 5, 6, 0, 0, // row 0 + 2 padding bytes
            7, 8, 9, 10, 11, 12, 0, 0, // row 1 + 2 padding bytes
            13, 14, 15, 16, 17, 18, // row 2, no trailing padding
        ];
        let packed = super::pack_rows(&data, 1, 8, 6, 3);
        assert_eq!(packed, (1..=18).collect::<Vec<u8>>());

        // Tightly packed rows come back unchanged.
        let tight: Vec<u8> = (0..12).collect();
        assert_eq!(super::pack_rows(&tight, 0, 6, 6, 2), tight);

        // Zero rows or zero-width rows produce an empty buffer instead of panicking.
        assert!(super::pack_rows(&[], 5, 0, 0, 0).is_empty());
    }

    /// Regression test for #1160: for widths where `width * 3` is not a multiple of 4,
    /// GStreamer pads every RGB row. Reading the frame as if it were tightly packed shears
    /// it, so each row ends up shifted further than the one above it.
    ///
    /// `videotestsrc` paints the default SMPTE pattern: the top part of the frame is vertical
    /// colour bars, so every row in the top half must be identical to row 0.
    #[test]
    fn capture_unaligned_width_is_not_sheared() -> Result<(), Box<dyn std::error::Error>> {
        const WIDTH: usize = 854; // 854 * 3 = 2562 bytes, padded to a 2564-byte stride
        const HEIGHT: usize = 480;

        if !gstreamer::INITIALIZED.load(std::sync::atomic::Ordering::Relaxed) {
            gstreamer::init()?;
        }

        let pipeline_desc = format!(
            "videotestsrc num-buffers=1 pattern=smpte ! \
             video/x-raw,format=RGB,width={WIDTH},height={HEIGHT},framerate=30/1 ! \
             appsink name=sink sync=false"
        );
        let mut capture = StreamCapture::new(&pipeline_desc)?;
        capture.start()?;

        let mut frame = None;
        for _ in 0..200 {
            if let Some(image) = capture.grab_rgb8()? {
                frame = Some(image);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        capture.close()?;
        let image = frame.ok_or("no frame received from videotestsrc")?;

        assert_eq!(image.width(), WIDTH);
        assert_eq!(image.height(), HEIGHT);
        let data = image.as_slice();
        assert_eq!(
            data.len(),
            WIDTH * HEIGHT * 3,
            "rows must be tightly packed"
        );

        let row_bytes = WIDTH * 3;
        let row0 = &data[..row_bytes];
        // Sanity check: the bars make row 0 non-uniform, so a shifted row cannot match it.
        assert!(
            row0.iter().any(|&b| b != row0[0]),
            "row 0 has no colour bars"
        );
        for y in 1..HEIGHT / 2 {
            let row = &data[y * row_bytes..(y + 1) * row_bytes];
            assert!(row == row0, "row {y} differs from row 0: frame is sheared");
        }
        Ok(())
    }
}
