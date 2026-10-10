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

/// Construct an RGB [`Image`] from a GStreamer buffer and the [`VideoInfo`] of its caps.
///
/// The frame is mapped with [`VideoFrame::from_buffer_readable`] (`gst_video_frame_map`), which
/// applies the buffer's `VideoMeta` (its map function, plane offset and stride) when it has
/// one and the default layout of `info` otherwise. GStreamer may pad every row so that it
/// starts on an aligned address: in the default layout an RGB row of `width * 3` bytes is
/// rounded up to a multiple of 4, so an 854-pixel row takes 2564 bytes instead of 2562.
///
/// When the rows are tightly packed (`stride == width * 3`) the image borrows the mapped frame
/// without copying, whatever the plane offset. Otherwise the pixels of each row are copied into
/// a new packed image and the padding bytes are skipped.
///
/// Two layouts that `gst_video_frame_map` cannot handle are read from the whole buffer
/// instead, with the same bounds checks:
/// - a `VideoMeta` plane that does not fit in the memory holding its offset (e.g. a frame
///   split across several memories), since GStreamer maps only that memory;
/// - a buffer without a `VideoMeta` that is smaller than the default layout. If it still
///   holds every row's pixels at the default stride (only the padding after the last row is
///   missing), it is read with that stride; otherwise, if it holds `width * height * 3`
///   bytes, it is read as tightly packed rows, as pushed by `appsrc` producers that copy a
///   packed image. A buffer without a `VideoMeta` that is at least as large as the default
///   layout is always read with the default stride, even if its producer packed the rows.
///
/// [`VideoInfo`]: gstreamer_video::VideoInfo
/// [`VideoFrame::from_buffer_readable`]: gstreamer_video::VideoFrame::from_buffer_readable
///
/// # Arguments
///
/// * `buffer` - the frame buffer. In the zero-copy case it stays mapped for as long as the
///   returned image lives.
/// * `info` - the video info parsed from the sample caps. Its format must be `RGB`.
///
/// # Returns
///
/// An `Image<u8, 3>` of `info.width() x info.height()` pixels with tightly packed rows.
///
/// # Errors
///
/// Returns [`StreamCaptureError::InvalidImageFormat`] if the caps describe an empty or invalid
/// frame, or the frame has no planes, a negative row stride (bottom-up rows), a stride smaller
/// than `width * 3`, or a `VideoMeta` that does not match the caps;
/// [`StreamCaptureError::BufferSizeMismatch`] if the buffer is too small for the frame layout;
/// and [`StreamCaptureError::GetBufferError`] if it cannot be mapped.
pub(crate) fn image_from_gst_buffer(
    buffer: gstreamer::Buffer,
    info: &gstreamer_video::VideoInfo,
) -> Result<Image<u8, 3>, StreamCaptureError> {
    use gstreamer_video::prelude::*;

    // `VideoFrame::from_buffer_readable` asserts that `info` is valid, and `VideoInfo::from_caps`
    // accepts zero-sized caps that are not, so reject them here instead of panicking.
    if !info.is_valid() || info.width() == 0 || info.height() == 0 {
        return Err(StreamCaptureError::InvalidImageFormat(format!(
            "invalid frame size {}x{}",
            info.width(),
            info.height()
        )));
    }
    let size = kornia_image::ImageSize {
        width: info.width() as usize,
        height: info.height() as usize,
    };
    let row_bytes = size.width.checked_mul(3).ok_or_else(|| {
        StreamCaptureError::InvalidImageFormat(format!(
            "frame dimensions overflow: {}x{}",
            size.width, size.height
        ))
    })?;
    let has_meta = buffer.meta::<gstreamer_video::VideoMeta>().is_some();

    let frame = match gstreamer_video::VideoFrame::from_buffer_readable(buffer, info) {
        Ok(frame) => frame,
        // Without a meta, mapping fails only when the buffer is smaller than `info.size()`.
        Err(buffer) if !has_meta => {
            return image_from_undersized_buffer(buffer, info, size, row_bytes)
        }
        Err(_) => {
            return Err(StreamCaptureError::InvalidImageFormat(
                "cannot map the frame described by the buffer's VideoMeta".to_string(),
            ))
        }
    };

    // `plane_data` reads the stride as unsigned to size the plane, so a negative stride must be
    // rejected before it is called.
    let stride = frame.plane_stride().first().copied().ok_or_else(|| {
        StreamCaptureError::InvalidImageFormat("video frame has no planes".to_string())
    })?;
    let stride = usize::try_from(stride).map_err(|_| {
        StreamCaptureError::InvalidImageFormat(format!(
            "negative row stride {stride} is not supported"
        ))
    })?;
    if stride < row_bytes {
        return Err(StreamCaptureError::InvalidImageFormat(format!(
            "row stride {stride} is smaller than width * 3 = {row_bytes}"
        )));
    }
    // `gst_video_frame_map` only checks this with `g_return_val_if_fail`, which builds without
    // checks compile out, and the reads below rely on it.
    let plane_height = frame.plane_height(0) as usize;
    if (frame.width() as usize) < size.width || plane_height < size.height {
        return Err(StreamCaptureError::InvalidImageFormat(format!(
            "VideoMeta size {}x{plane_height} is smaller than the caps size {}x{}",
            frame.width(),
            size.width,
            size.height
        )));
    }

    // `plane_data(0)` is a slice of `stride * plane_height` bytes from the plane start, sized in
    // `u32` arithmetic. With a `VideoMeta`, GStreamer maps only the memory holding the plane
    // offset and does not check that the plane fits in it, so `plane_data` is used only when the
    // plane provably fits; any other layout is read from the whole buffer.
    let offset = frame.plane_offset()[0];
    let plane_fits = stride
        .checked_mul(plane_height)
        .filter(|&plane_len| plane_len <= u32::MAX as usize)
        .is_some_and(|plane_len| mapped_plane_len(frame.buffer(), offset, has_meta) >= plane_len);
    if !plane_fits {
        return image_from_strided_buffer(frame.into_buffer(), size, row_bytes, offset, stride);
    }
    let plane = frame
        .plane_data(0)
        .map_err(|e| StreamCaptureError::InvalidImageFormat(e.to_string()))?;

    if stride != row_bytes {
        let packed = pack_rows(plane, stride, row_bytes, size.height);
        return Image::<u8, 3>::new(size, packed).map_err(StreamCaptureError::ImageError);
    }

    let data_ptr: *const u8 = plane.as_ptr();
    let keepalive: Arc<dyn Any + Send + Sync> = Arc::new(frame);

    // SAFETY:
    // - `data_ptr` is the start of plane 0 of a frame that `keepalive` keeps mapped for
    //   reading for the lifetime of the Image. Moving the `VideoFrame` into the `Arc` does not
    //   move the mapped memory.
    // - The plane holds `stride * plane_height >= width * height * 3` bytes (stride equals
    //   `width * 3` here and `plane_height >= height` was checked above), and we checked that
    //   the mapping covers all of them, so reads stay in bounds.
    let image = unsafe {
        Image::<u8, 3>::from_borrowed_host_readonly(size, data_ptr, keepalive)
            .map_err(StreamCaptureError::ImageError)?
    };

    Ok(image)
}

/// Returns how many bytes of `buffer` are mapped from the plane at byte `offset` onwards.
///
/// With a `VideoMeta`, the default map function maps only the memories that hold `offset`
/// (`gst_buffer_find_memory(buffer, offset, 1)`). Without one, the whole buffer is mapped as a
/// single block.
fn mapped_plane_len(buffer: &gstreamer::BufferRef, offset: usize, has_meta: bool) -> usize {
    if !has_meta {
        return buffer.size().saturating_sub(offset);
    }
    let Some((memories, skip)) = buffer.find_memory(offset..offset.saturating_add(1)) else {
        return 0;
    };
    let mapped: usize = memories.map(|idx| buffer.peek_memory(idx).size()).sum();
    mapped.saturating_sub(skip)
}

/// Reads a buffer without a `VideoMeta` that is smaller than the default layout of `info`.
///
/// It is read with the default stride if it holds every row's pixels at that stride (only the
/// padding after the last row is missing), and as tightly packed rows if it holds
/// `width * height * 3` bytes. For two or more rows the first condition needs more bytes than
/// the second whenever the default layout pads rows, so the two cannot be confused.
///
/// # Errors
///
/// Returns [`StreamCaptureError::BufferSizeMismatch`] if the buffer holds fewer than
/// `width * height * 3` bytes, and the errors of [`image_from_strided_buffer`].
fn image_from_undersized_buffer(
    buffer: gstreamer::Buffer,
    info: &gstreamer_video::VideoInfo,
    size: kornia_image::ImageSize,
    row_bytes: usize,
) -> Result<Image<u8, 3>, StreamCaptureError> {
    let default_stride = info
        .stride()
        .first()
        .and_then(|&stride| usize::try_from(stride).ok())
        .unwrap_or(row_bytes);
    let default_offset = info.offset().first().copied().unwrap_or(0);
    if strided_len(size.height, default_stride, row_bytes, default_offset)
        .is_some_and(|len| buffer.size() >= len)
    {
        return image_from_strided_buffer(buffer, size, row_bytes, default_offset, default_stride);
    }
    image_from_strided_buffer(buffer, size, row_bytes, 0, row_bytes)
}

/// Bytes needed to read `height` rows of `row_bytes` pixels, `stride` apart, from `offset`: all
/// rows but the last are a full stride, the last needs only its pixels. `None` on overflow.
fn strided_len(height: usize, stride: usize, row_bytes: usize, offset: usize) -> Option<usize> {
    match height.checked_sub(1) {
        None => Some(offset),
        Some(last_row) => last_row
            .checked_mul(stride)?
            .checked_add(row_bytes)?
            .checked_add(offset),
    }
}

/// Reads `size.height` RGB rows, `stride` bytes apart and starting at byte `offset`, from the
/// whole buffer mapped as one block. The rows are borrowed when they are contiguous and copied
/// into a packed image otherwise.
///
/// # Errors
///
/// Returns [`StreamCaptureError::GetBufferError`] if the buffer cannot be mapped,
/// [`StreamCaptureError::InvalidImageFormat`] if the layout overflows, and
/// [`StreamCaptureError::BufferSizeMismatch`] if the buffer does not hold every row.
fn image_from_strided_buffer(
    buffer: gstreamer::Buffer,
    size: kornia_image::ImageSize,
    row_bytes: usize,
    offset: usize,
    stride: usize,
) -> Result<Image<u8, 3>, StreamCaptureError> {
    let mapped_buffer = buffer
        .into_mapped_buffer_readable()
        .map_err(|_| StreamCaptureError::GetBufferError)?;
    let required_len = strided_len(size.height, stride, row_bytes, offset).ok_or_else(|| {
        StreamCaptureError::InvalidImageFormat(format!(
            "frame layout overflows: {}x{} with stride {stride} at offset {offset}",
            size.width, size.height
        ))
    })?;
    if mapped_buffer.len() < required_len {
        return Err(StreamCaptureError::BufferSizeMismatch {
            expected: required_len,
            got: mapped_buffer.len(),
        });
    }

    let rows = &mapped_buffer.as_slice()[offset..];
    if stride != row_bytes && size.height > 1 {
        let packed = pack_rows(rows, stride, row_bytes, size.height);
        return Image::<u8, 3>::new(size, packed).map_err(StreamCaptureError::ImageError);
    }

    // Capture the pointer BEFORE moving mapped_buffer into GstResource.
    let data_ptr: *const u8 = rows.as_ptr();
    let keepalive: Arc<dyn Any + Send + Sync> = Arc::new(GstResource {
        _map: mapped_buffer,
    });

    // SAFETY:
    // - `data_ptr` points `offset` bytes into the mapped buffer, which `keepalive`
    //   (GstResource) keeps mapped for the lifetime of the Image.
    // - The rows are contiguous here (`stride == width * 3`, or a single row), and we checked
    //   that the buffer holds `offset + width * height * 3` bytes, so reads stay in bounds.
    let image = unsafe {
        Image::<u8, 3>::from_borrowed_host_readonly(size, data_ptr, keepalive)
            .map_err(StreamCaptureError::ImageError)?
    };

    Ok(image)
}

/// Copies `height` rows of `row_bytes` bytes out of a padded plane into a packed `Vec`.
///
/// Row `i` starts at `i * stride` in `plane`; the `stride - row_bytes` padding bytes after each
/// row are skipped. The caller must have checked that `plane` holds every row.
fn pack_rows(plane: &[u8], stride: usize, row_bytes: usize, height: usize) -> Vec<u8> {
    if height == 0 || row_bytes == 0 {
        return Vec::new();
    }
    let mut packed = Vec::with_capacity(row_bytes * height);
    for row in plane.chunks(stride).take(height) {
        packed.extend_from_slice(&row[..row_bytes]);
    }
    packed
}

#[cfg(test)]
mod tests {
    use crate::stream::{StreamCapture, StreamCaptureError};
    use gstreamer_video::{VideoFormat, VideoFrameFlags, VideoInfo, VideoMeta};
    use kornia_image::Image;

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
        // 2x3 RGB image: 6 pixel bytes per row, padded to a stride of 8.
        let plane: Vec<u8> = vec![
            1, 2, 3, 4, 5, 6, 0, 0, // row 0 + 2 padding bytes
            7, 8, 9, 10, 11, 12, 0, 0, // row 1 + 2 padding bytes
            13, 14, 15, 16, 17, 18, // row 2, no trailing padding
        ];
        let packed = super::pack_rows(&plane, 8, 6, 3);
        assert_eq!(packed, (1..=18).collect::<Vec<u8>>());

        // Tightly packed rows come back unchanged.
        let tight: Vec<u8> = (0..12).collect();
        assert_eq!(super::pack_rows(&tight, 6, 6, 2), tight);

        // Zero rows or zero-width rows produce an empty buffer instead of panicking.
        assert!(super::pack_rows(&[], 0, 0, 0).is_empty());
    }

    /// Builds the bytes of an RGB frame of `height` rows of `width` pixels, each row padded to
    /// `stride` bytes and the first row starting at `offset`. Pixel bytes count up (mod 251, so
    /// no two rows are equal) and padding bytes are 0xAA. Returns the frame bytes and the
    /// expected tightly packed pixels.
    fn padded_rgb(width: usize, height: usize, stride: usize, offset: usize) -> (Vec<u8>, Vec<u8>) {
        let row_bytes = width * 3;
        let packed: Vec<u8> = (0..row_bytes * height).map(|i| (i % 251) as u8).collect();
        let mut data = vec![0xAA; offset + stride * height];
        for (y, row) in packed.chunks(row_bytes).enumerate() {
            let start = offset + y * stride;
            data[start..start + row_bytes].copy_from_slice(row);
        }
        (data, packed)
    }

    /// The conversion result and the address of the source buffer's first byte.
    type Converted = (Result<Image<u8, 3>, StreamCaptureError>, usize);

    /// A `VideoMeta` to attach to a test buffer.
    struct Meta {
        width: u32,
        height: u32,
        offset: usize,
        stride: i32,
    }

    /// Wraps `parts` in a buffer with one memory per part, attaching `meta` if given, and
    /// converts it with RGB caps of `width`x`height`. Also returns the address of the first
    /// byte, so callers can tell a borrowed image from a copied one.
    fn convert_parts(
        parts: Vec<Vec<u8>>,
        width: u32,
        height: u32,
        meta: Option<Meta>,
    ) -> Result<Converted, Box<dyn std::error::Error>> {
        gstreamer::init()?;
        let info = VideoInfo::builder(VideoFormat::Rgb, width, height).build()?;
        let base = parts.first().map_or(0, |part| part.as_ptr() as usize);
        let mut buffer = gstreamer::Buffer::new();
        {
            let buffer = buffer.get_mut().ok_or("buffer is not writable")?;
            for part in parts {
                buffer.append_memory(gstreamer::Memory::from_slice(part));
            }
            if let Some(meta) = meta {
                VideoMeta::add_full(
                    buffer,
                    VideoFrameFlags::empty(),
                    VideoFormat::Rgb,
                    meta.width,
                    meta.height,
                    &[meta.offset],
                    &[meta.stride],
                )?;
            }
        }
        Ok((super::image_from_gst_buffer(buffer, &info), base))
    }

    /// Single-memory variant of [`convert_parts`]; `meta` is `(offset, stride)` at the caps size.
    fn convert(
        data: Vec<u8>,
        width: u32,
        height: u32,
        meta: Option<(usize, i32)>,
    ) -> Result<Converted, Box<dyn std::error::Error>> {
        let meta = meta.map(|(offset, stride)| Meta {
            width,
            height,
            offset,
            stride,
        });
        convert_parts(vec![data], width, height, meta)
    }

    /// Tightly packed rows (an aligned width, default layout) are borrowed, not copied.
    #[test]
    fn tight_rows_are_borrowed_without_copy() -> Result<(), Box<dyn std::error::Error>> {
        let (data, packed) = padded_rgb(640, 4, 640 * 3, 0);
        let (image, base) = convert(data, 640, 4, None)?;
        let image = image?;
        assert_eq!(image.as_slice().as_ptr() as usize, base, "image was copied");
        assert_eq!(image.as_slice(), packed.as_slice());
        Ok(())
    }

    /// Regression test for #1160 at the buffer level: the default layout pads an 854-pixel RGB
    /// row to 2564 bytes, and the padding must be dropped.
    #[test]
    fn padded_default_layout_is_repacked() -> Result<(), Box<dyn std::error::Error>> {
        let (data, packed) = padded_rgb(854, 4, 2564, 0);
        let (image, base) = convert(data, 854, 4, None)?;
        let image = image?;
        assert_ne!(image.as_slice().as_ptr() as usize, base);
        assert_eq!(image.as_slice(), packed.as_slice());
        Ok(())
    }

    /// Buffers without a `VideoMeta` that are smaller than the default layout: a padded frame
    /// missing only the padding after its last row is still read with the default stride, and
    /// exactly `width * height * 3` bytes (as pushed by `appsrc` from a packed image) are read
    /// as tight rows without copying, also for a single row.
    #[test]
    fn undersized_buffers_without_meta_are_read() -> Result<(), Box<dyn std::error::Error>> {
        let (mut data, packed) = padded_rgb(854, 4, 2564, 0);
        data.truncate(3 * 2564 + 854 * 3);
        let (image, _) = convert(data, 854, 4, None)?;
        assert_eq!(
            image?.as_slice(),
            packed.as_slice(),
            "last row padding missing"
        );

        for height in [4, 1] {
            let (data, packed) = padded_rgb(854, height, 854 * 3, 0);
            let (image, base) = convert(data, 854, height as u32, None)?;
            let image = image?;
            assert_eq!(image.as_slice().as_ptr() as usize, base, "image was copied");
            assert_eq!(
                image.as_slice(),
                packed.as_slice(),
                "tight rows, height {height}"
            );
        }
        Ok(())
    }

    /// The stride and offset of a `VideoMeta` win over the default layout. A meta with an
    /// offset but no row padding is still borrowed, starting at the offset.
    #[test]
    fn video_meta_layout_is_applied() -> Result<(), Box<dyn std::error::Error>> {
        let (data, packed) = padded_rgb(854, 4, 2564, 0);
        let (image, _) = convert(data, 854, 4, Some((0, 2564)))?;
        assert_eq!(image?.as_slice(), packed.as_slice());

        let (data, packed) = padded_rgb(854, 4, 2562, 16);
        let (image, base) = convert(data, 854, 4, Some((16, 2562)))?;
        let image = image?;
        assert_eq!(
            image.as_slice().as_ptr() as usize,
            base + 16,
            "image was copied"
        );
        assert_eq!(image.as_slice(), packed.as_slice());
        Ok(())
    }

    /// A `VideoMeta` frame split across several memories does not fit in the memory GStreamer
    /// maps for the plane, so it is read from the whole buffer instead of being rejected.
    #[test]
    fn video_meta_frame_split_across_memories_is_read() -> Result<(), Box<dyn std::error::Error>> {
        let (data, packed) = padded_rgb(854, 4, 2562, 0);
        let parts = data.chunks(5124).map(<[u8]>::to_vec).collect();
        let meta = Meta {
            width: 854,
            height: 4,
            offset: 0,
            stride: 2562,
        };
        let (image, _) = convert_parts(parts, 854, 4, Some(meta))?;
        assert_eq!(
            image?.as_slice(),
            packed.as_slice(),
            "tight rows in 2 memories"
        );

        let (data, packed) = padded_rgb(854, 4, 2564, 0);
        let parts = data.chunks(2564).map(<[u8]>::to_vec).collect();
        let meta = Meta {
            width: 854,
            height: 4,
            offset: 0,
            stride: 2564,
        };
        let (image, _) = convert_parts(parts, 854, 4, Some(meta))?;
        assert_eq!(
            image?.as_slice(),
            packed.as_slice(),
            "one memory per padded row"
        );

        // A 16-byte header shares the first memory with the start of the frame.
        let (data, packed) = padded_rgb(854, 4, 2562, 16);
        let parts = vec![data[..16 + 5124].to_vec(), data[16 + 5124..].to_vec()];
        let meta = Meta {
            width: 854,
            height: 4,
            offset: 16,
            stride: 2562,
        };
        let (image, _) = convert_parts(parts, 854, 4, Some(meta))?;
        assert_eq!(
            image?.as_slice(),
            packed.as_slice(),
            "offset 16 in 2 memories"
        );
        Ok(())
    }

    /// `mapped_plane_len` must match what GStreamer maps: the whole buffer without a meta, and
    /// only the memory holding the offset (minus the bytes before it) with one.
    #[test]
    fn mapped_plane_len_matches_the_gstreamer_mapping() -> Result<(), Box<dyn std::error::Error>> {
        gstreamer::init()?;
        let mut buffer = gstreamer::Buffer::new();
        {
            let buffer = buffer.get_mut().ok_or("buffer is not writable")?;
            buffer.append_memory(gstreamer::Memory::from_slice(vec![0u8; 100]));
            buffer.append_memory(gstreamer::Memory::from_slice(vec![0u8; 200]));
        }
        assert_eq!(super::mapped_plane_len(&buffer, 10, false), 290);
        assert_eq!(super::mapped_plane_len(&buffer, 10, true), 90);
        assert_eq!(super::mapped_plane_len(&buffer, 150, true), 150);
        assert_eq!(super::mapped_plane_len(&buffer, 300, true), 0);
        Ok(())
    }

    /// Buffers too small for their layout are rejected before any row is read, reporting the
    /// minimum size their layout needs. This includes a `VideoMeta` whose plane runs past the
    /// end of the buffer, which `gst_video_frame_map` itself accepts.
    #[test]
    fn undersized_buffers_are_rejected() -> Result<(), Box<dyn std::error::Error>> {
        let (mut data, _) = padded_rgb(854, 4, 854 * 3, 0);
        data.pop();
        let (image, _) = convert(data, 854, 4, None)?;
        assert!(
            matches!(
                image,
                Err(StreamCaptureError::BufferSizeMismatch {
                    expected: 10248,
                    got: 10247
                })
            ),
            "short buffer: {:?}",
            image.map(|_| ())
        );

        let (data, _) = padded_rgb(854, 4, 2564, 0);
        let (image, _) = convert(data, 854, 4, Some((0, 4096)))?;
        assert!(
            matches!(
                image,
                Err(StreamCaptureError::BufferSizeMismatch {
                    expected: 14850,
                    got: 10256
                })
            ),
            "plane past the end of the buffer: {:?}",
            image.map(|_| ())
        );
        Ok(())
    }

    /// Negative strides (bottom-up rows), strides shorter than a row, a `VideoMeta` smaller
    /// than the caps and zero-sized caps are rejected with `InvalidImageFormat`, without
    /// panicking.
    #[test]
    fn invalid_layouts_are_rejected() -> Result<(), Box<dyn std::error::Error>> {
        let invalid = |image: Result<Image<u8, 3>, StreamCaptureError>, needle: &str| matches!(image, Err(StreamCaptureError::InvalidImageFormat(ref msg)) if msg.contains(needle));

        let (data, _) = padded_rgb(854, 4, 2562, 0);
        let (image, _) = convert(data, 854, 4, Some((2562 * 3, -2562)))?;
        assert!(invalid(image, "negative"), "negative stride");

        let (data, _) = padded_rgb(854, 4, 2564, 0);
        let (image, _) = convert(data, 854, 4, Some((0, 100)))?;
        assert!(
            invalid(image, "smaller than width"),
            "stride shorter than a row"
        );

        // A tight buffer that the no-meta fallback would accept, so this only fails if the
        // mismatching meta is honoured. GStreamer logs a CRITICAL for the mismatch.
        let (data, _) = padded_rgb(854, 4, 854 * 3, 0);
        let meta = Meta {
            width: 800,
            height: 4,
            offset: 0,
            stride: 2562,
        };
        let (image, _) = convert_parts(vec![data], 854, 4, Some(meta))?;
        assert!(invalid(image, "VideoMeta"), "meta smaller than the caps");

        for (width, height) in [(0, 4), (4, 0)] {
            let caps = gstreamer::Caps::builder("video/x-raw")
                .field("format", "RGB")
                .field("width", width)
                .field("height", height)
                .field("framerate", gstreamer::Fraction::new(30, 1))
                .build();
            let info = VideoInfo::from_caps(&caps)?;
            let image =
                super::image_from_gst_buffer(gstreamer::Buffer::from_slice([0u8; 64]), &info);
            assert!(
                invalid(image, "invalid frame size"),
                "{width}x{height} caps"
            );
        }
        Ok(())
    }

    /// Starts `pipeline_desc`, pushes `data` into its `appsrc` named `src` if given, polls
    /// `grab_rgb8` until it returns a frame or an error, then closes the pipeline. Returns
    /// `Ok(None)` if nothing arrives within about 2 seconds, and the fps seen after the grab.
    fn grab_first_with(
        pipeline_desc: &str,
        data: Option<Vec<u8>>,
    ) -> Result<(Option<Image<u8, 3>>, Option<f64>), StreamCaptureError> {
        use gstreamer::prelude::*;
        let mut capture = StreamCapture::new(pipeline_desc)?;
        capture.start()?;
        if let Some(data) = data {
            let appsrc = capture
                .pipeline
                .by_name("src")
                .ok_or(StreamCaptureError::GetElementByNameError)?
                .dynamic_cast::<gstreamer_app::AppSrc>()
                .map_err(StreamCaptureError::DowncastPipelineError)?;
            appsrc.push_buffer(gstreamer::Buffer::from_slice(data))?;
        }
        let mut result = Ok(None);
        for _ in 0..200 {
            result = capture.grab_rgb8();
            if !matches!(result, Ok(None)) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let fps = capture.get_fps();
        capture.close()?;
        Ok((result?, fps))
    }

    /// [`grab_first_with`] without an `appsrc` push.
    fn grab_first(pipeline_desc: &str) -> Result<Option<Image<u8, 3>>, StreamCaptureError> {
        Ok(grab_first_with(pipeline_desc, None)?.0)
    }

    /// Whether the first sample `pipeline_desc` delivers to its appsink carries a `VideoMeta`.
    fn first_sample_has_video_meta(
        pipeline_desc: &str,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        use gstreamer::prelude::*;
        gstreamer::init()?;
        let pipeline = gstreamer::parse::launch(pipeline_desc)?
            .dynamic_cast::<gstreamer::Pipeline>()
            .map_err(|_| "not a pipeline")?;
        let appsink = pipeline
            .by_name("sink")
            .ok_or("no appsink named sink")?
            .dynamic_cast::<gstreamer_app::AppSink>()
            .map_err(|_| "not an appsink")?;
        pipeline.set_state(gstreamer::State::Playing)?;
        let sample = appsink.pull_sample();
        pipeline.set_state(gstreamer::State::Null)?;
        let sample = sample?;
        let buffer = sample.buffer().ok_or("sample has no buffer")?;
        Ok(buffer.meta::<VideoMeta>().is_some())
    }

    /// Checks that every row in the top half of a packed frame equals row 0. `videotestsrc`
    /// paints the SMPTE pattern with vertical colour bars at the top, so a sheared frame (rows
    /// read at the wrong stride) fails this check.
    fn assert_top_half_rows_match(data: &[u8], row_bytes: usize, height: usize) {
        let row0 = &data[..row_bytes];
        // Sanity check: the bars make row 0 non-uniform, so a shifted row cannot match it.
        assert!(
            row0.iter().any(|&b| b != row0[0]),
            "row 0 has no colour bars"
        );
        for y in 1..height / 2 {
            let row = &data[y * row_bytes..(y + 1) * row_bytes];
            assert!(row == row0, "row {y} differs from row 0: frame is sheared");
        }
    }

    /// Regression test for #1160: for widths where `width * 3` is not a multiple of 4,
    /// GStreamer pads every RGB row. Reading the frame as if it were tightly packed shears
    /// it, so each row ends up shifted further than the one above it.
    ///
    /// The first pipeline feeds RGB straight from `videotestsrc`, which attaches no
    /// `VideoMeta` (default layout). The second converts from I420 like `VideoReader`, V4L2 and
    /// RTSP do, where `videoconvert` describes the layout with a `VideoMeta`; that is asserted,
    /// so the test notices if a GStreamer version stops covering the meta path.
    #[test]
    fn capture_unaligned_width_is_not_sheared() -> Result<(), Box<dyn std::error::Error>> {
        const WIDTH: usize = 854; // 854 * 3 = 2562 bytes, padded to a 2564-byte stride
        const HEIGHT: usize = 480;

        for (upstream, has_meta) in [
            ("", false),
            ("video/x-raw,format=I420 ! videoconvert ! ", true),
        ] {
            let pipeline_desc = format!(
                "videotestsrc num-buffers=1 pattern=smpte ! {upstream}\
                 video/x-raw,format=RGB,width={WIDTH},height={HEIGHT},framerate=30/1 ! \
                 appsink name=sink sync=false"
            );
            assert_eq!(
                first_sample_has_video_meta(&pipeline_desc)?,
                has_meta,
                "VideoMeta presence for {upstream:?}"
            );

            let image = grab_first(&pipeline_desc)?.ok_or("no frame received from videotestsrc")?;
            assert_eq!(image.width(), WIDTH);
            assert_eq!(image.height(), HEIGHT);
            assert_top_half_rows_match(image.as_slice(), WIDTH * 3, HEIGHT);
        }
        Ok(())
    }

    /// `grab_rgb8` reports frames it cannot read as an error instead of returning a misread
    /// image or silently ending the stream: non-RGB raw formats and encoded caps give
    /// `InvalidImageFormat`, caps that are not video caps give `GetCapsError`.
    #[test]
    fn grab_rgb8_rejects_other_formats() {
        for caps in [
            "video/x-raw,format=RGBx,width=64,height=48,framerate=30/1",
            "video/x-raw,format=BGR,width=64,height=48,framerate=30/1",
            "video/x-raw,width=64,height=48,framerate=30/1 ! jpegenc",
        ] {
            let result = grab_first(&format!(
                "videotestsrc num-buffers=1 ! {caps} ! appsink name=sink sync=false"
            ));
            assert!(
                matches!(result, Err(StreamCaptureError::InvalidImageFormat(_))),
                "{caps}: {:?}",
                result.map(|frame| frame.is_some())
            );
        }

        let result = grab_first_with(
            "appsrc name=src caps=application/x-unknown ! appsink name=sink sync=false",
            Some(vec![0u8; 16]),
        );
        assert!(
            matches!(result, Err(StreamCaptureError::GetCapsError(_))),
            "non-video caps: {:?}",
            result.map(|(frame, _)| frame.is_some())
        );
    }

    /// Caps without a framerate used to fail on the streaming thread, which ended the stream
    /// without an error. The frame is now delivered and the fps reads as 0 (unknown).
    #[test]
    fn caps_without_framerate_still_deliver_frames() -> Result<(), Box<dyn std::error::Error>> {
        let (data, packed) = padded_rgb(4, 4, 12, 0);
        let (image, fps) = grab_first_with(
            "appsrc name=src caps=video/x-raw,format=RGB,width=4,height=4 ! \
             appsink name=sink sync=false",
            Some(data),
        )?;
        let image = image.ok_or("no frame delivered")?;
        assert_eq!(image.as_slice(), packed.as_slice());
        assert_eq!(fps, Some(0.0));
        Ok(())
    }
}
