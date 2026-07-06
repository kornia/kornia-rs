//! A generic, format-aware captured GStreamer frame.
//!
//! [`GstFrame`] is the GStreamer analogue of the V4L `EncodedFrame`: it holds a
//! **zero-copy** view of the mapped buffer plus the negotiated [`VideoInfo`]
//! (format, dimensions, strides) and timing metadata (PTS/duration). It decodes to
//! a typed [`Image`] on demand, validating the requested element type / channel
//! count against the frame's actual format and honouring real row strides.

use std::any::Any;
use std::sync::Arc;
use std::time::Duration;

use gstreamer_video::{VideoFormat, VideoInfo};

use kornia_image::{ColorSpace, Image, ImageSize};
use kornia_tensor::resource::MemoryDomain;
use kornia_tensor::storage::TensorStorage;
use kornia_tensor::Tensor;

use super::GstResource;
use crate::stream::error::StreamCaptureError;

/// A single captured frame: a zero-copy view of the mapped GStreamer buffer plus
/// the negotiated video info and timing metadata.
///
/// The raw bytes are always available zero-copy via [`as_bytes`](Self::as_bytes).
/// Typed images are produced on demand via [`to_image_u8`](Self::to_image_u8) /
/// [`to_image_u16`](Self::to_image_u16), which validate the requested element type
/// and channel count against the frame's actual format.
pub struct GstFrame {
    /// Arc-shared keepalive over the mapped buffer (holds the map + buffer ref).
    resource: Arc<GstResource>,
    info: VideoInfo,
    pts: Option<gstreamer::ClockTime>,
    duration: Option<gstreamer::ClockTime>,
}

/// Channel count + colour space for a `u8` [`VideoFormat`].
fn u8_layout(format: VideoFormat) -> Option<(usize, ColorSpace)> {
    match format {
        VideoFormat::Rgb => Some((3, ColorSpace::Rgb)),
        VideoFormat::Bgr => Some((3, ColorSpace::Bgr)),
        VideoFormat::Rgba => Some((4, ColorSpace::Rgba)),
        VideoFormat::Gray8 => Some((1, ColorSpace::Gray)),
        _ => None,
    }
}

/// Channel count + colour space for a native-endian `u16` [`VideoFormat`].
fn u16_layout(format: VideoFormat) -> Option<(usize, ColorSpace)> {
    // Only the native-endian 16-bit gray format is exposed zero-copy; the
    // opposite endianness would require a byte-swapping copy.
    #[cfg(target_endian = "little")]
    let native_gray16 = VideoFormat::Gray16Le;
    #[cfg(target_endian = "big")]
    let native_gray16 = VideoFormat::Gray16Be;

    if format == native_gray16 {
        Some((1, ColorSpace::Gray))
    } else {
        None
    }
}

impl GstFrame {
    /// Build a frame from a mapped buffer, its negotiated info and timing.
    pub(crate) fn new(
        map: gstreamer::buffer::MappedBuffer<gstreamer::buffer::Readable>,
        info: VideoInfo,
        pts: Option<gstreamer::ClockTime>,
        duration: Option<gstreamer::ClockTime>,
    ) -> Self {
        Self {
            resource: Arc::new(GstResource { _map: map }),
            info,
            pts,
            duration,
        }
    }

    /// The negotiated pixel format (e.g. `RGB`, `GRAY16_LE`).
    pub fn format(&self) -> VideoFormat {
        self.info.format()
    }

    /// Frame dimensions.
    pub fn size(&self) -> ImageSize {
        ImageSize {
            width: self.info.width() as usize,
            height: self.info.height() as usize,
        }
    }

    /// Row stride (in bytes) of plane 0 — may exceed `width * channels * elem` if
    /// the rows are padded.
    pub fn stride(&self) -> usize {
        self.info.stride()[0] as usize
    }

    /// The colour space implied by the negotiated format, if known.
    pub fn color_space(&self) -> Option<ColorSpace> {
        u8_layout(self.info.format())
            .or_else(|| u16_layout(self.info.format()))
            .map(|(_, cs)| cs)
    }

    /// Presentation timestamp, if set.
    pub fn pts(&self) -> Option<Duration> {
        self.pts.map(|t| Duration::from_nanos(t.nseconds()))
    }

    /// Frame duration, if set.
    pub fn duration(&self) -> Option<Duration> {
        self.duration.map(|t| Duration::from_nanos(t.nseconds()))
    }

    /// Zero-copy view of the raw mapped bytes (may include row padding).
    pub fn as_bytes(&self) -> &[u8] {
        self.resource.as_slice()
    }

    /// Decode as an interleaved `u8` image with `C` channels.
    ///
    /// Errors if the frame's format is not a `u8` format with exactly `C`
    /// channels (e.g. calling `to_image_u8::<3>()` on a `GRAY8` frame).
    pub fn to_image_u8<const C: usize>(&self) -> Result<Image<u8, C>, StreamCaptureError> {
        self.to_image::<u8, C>(u8_layout(self.info.format()))
    }

    /// Decode as an interleaved `u16` image with `C` channels (native endian).
    pub fn to_image_u16<const C: usize>(&self) -> Result<Image<u16, C>, StreamCaptureError> {
        self.to_image::<u16, C>(u16_layout(self.info.format()))
    }

    /// Shared conversion: validate `(T, C)` against the negotiated format, then
    /// build a zero-copy [`Image`] for tightly-packed rows, or a packed copy when
    /// the rows are padded.
    fn to_image<T, const C: usize>(
        &self,
        layout: Option<(usize, ColorSpace)>,
    ) -> Result<Image<T, C>, StreamCaptureError>
    where
        T: Copy + Default + Send + Sync + 'static,
        Image<T, C>: TryFrom<Tensor<T, 3>, Error = kornia_image::ImageError>,
    {
        let format = self.info.format();
        let Some((channels, _cs)) = layout else {
            return Err(StreamCaptureError::InvalidImageFormat(format!(
                "frame format {format:?} cannot be viewed as a {}-bit image",
                std::mem::size_of::<T>() * 8
            )));
        };
        if channels != C {
            return Err(StreamCaptureError::InvalidImageFormat(format!(
                "frame format {format:?} has {channels} channels, but {C} were requested"
            )));
        }

        let (w, h) = (self.size().width, self.size().height);
        let elem = std::mem::size_of::<T>();
        let row_bytes = self.stride();
        let packed_row = w * C * elem;

        // Guard: the mapped buffer must actually contain the pixel data.
        let needed = h.saturating_sub(1) * row_bytes + packed_row;
        let available = self.resource.as_slice().len();
        if available < needed {
            return Err(StreamCaptureError::BufferSizeMismatch {
                expected: needed,
                got: available,
            });
        }

        if row_bytes == packed_row {
            // Fast path: rows are tightly packed → zero-copy, sharing the mapped
            // buffer keepalive with the tensor storage.
            let data_ptr = self.resource.as_slice().as_ptr() as *const T;
            let keepalive: Arc<dyn Any + Send + Sync> = self.resource.clone();
            // SAFETY: data_ptr points to `h * packed_row` valid bytes (checked
            // above); the keepalive holds the map alive; storage is read-only.
            let storage: TensorStorage<T> = unsafe {
                TensorStorage::from_borrowed_readonly(
                    data_ptr,
                    h * packed_row,
                    kornia_tensor::host_alloc(),
                    MemoryDomain::Host,
                    keepalive,
                )
            };
            let tensor = Tensor {
                storage,
                shape: [h, w, C],
                strides: [w * C, C, 1],
            };
            Image::try_from(tensor).map_err(StreamCaptureError::ImageError)
        } else {
            // Padded rows: copy each row into a packed, owned image so `as_slice`
            // stays correct for downstream consumers.
            let mut img = Image::<T, C>::from_size_val(ImageSize { width: w, height: h }, T::default())
                .map_err(StreamCaptureError::ImageError)?;
            let src = self.resource.as_slice();
            let dst = img.as_slice_mut();
            // SAFETY: dst is `h * w * C` contiguous `T` values; reinterpret as bytes
            // to copy row-by-row from the (byte-strided) source.
            let dst_bytes =
                unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u8, h * packed_row) };
            for row in 0..h {
                let src_off = row * row_bytes;
                let dst_off = row * packed_row;
                dst_bytes[dst_off..dst_off + packed_row]
                    .copy_from_slice(&src[src_off..src_off + packed_row]);
            }
            Ok(img)
        }
    }
}
