//! Video frame extraction for the SfM pipeline.
//!
//! Reads an MP4 (or any GStreamer-decodable video) into a sequence of frames.
//! The video reader always decodes to RGB, so this module returns both the
//! owned RGB frames (used later to colour the reconstructed point cloud) and
//! matching grayscale frames (used for feature detection).

use std::error::Error;
use std::path::Path;
use std::time::Duration;

use kornia_image::Image;
use kornia_imgproc::color::gray_from_rgb_u8;
use kornia_io::gstreamer::video::{ImageFormat, VideoReader};

/// Owned frames decoded from a video: parallel RGB and grayscale buffers.
pub type VideoFrames = (Vec<Image<u8, 3>>, Vec<Image<u8, 1>>);

/// Read a video file into a sequence of RGB and grayscale frames.
///
/// Returns `(rgb_frames, gray_frames)`, both subsampled by `frame_step`:
/// only frames whose index is a multiple of `frame_step` are kept. A
/// `frame_step` of `1` keeps every frame.
///
/// # Arguments
///
/// * `path` - Path to the input video file (any format GStreamer can decode,
///   e.g. MP4, MKV, WebM).
/// * `frame_step` - Keep every Nth frame. Must be `>= 1`.
///
/// # Returns
///
/// A [`VideoFrames`] tuple where the first vector holds the owned RGB frames
/// and the second holds their grayscale conversions. The two vectors always
/// have the same length.
///
/// # Errors
///
/// Returns an error if the video cannot be opened or decoded, or if any frame
/// cannot be converted to grayscale.
pub fn read_frames(path: &Path, frame_step: usize) -> Result<VideoFrames, Box<dyn Error>> {
    assert!(frame_step >= 1, "frame_step must be >= 1");

    // NOTE: `VideoReader` only decodes to RGB via `grab_rgb8()` regardless of the
    // requested `ImageFormat`; requesting `Mono8` makes GStreamer produce 1-byte
    // frames that the RGB validator rejects at runtime. So we always request RGB
    // and convert to grayscale ourselves.
    let mut reader = VideoReader::new(path, ImageFormat::Rgb8)?;
    reader.start()?;

    let mut rgb_frames: Vec<Image<u8, 3>> = Vec::new();
    let mut gray_frames: Vec<Image<u8, 1>> = Vec::new();
    let mut frame_idx: usize = 0;

    loop {
        match reader.grab_rgb8()? {
            Some(frame) => {
                if frame_idx.is_multiple_of(frame_step) {
                    // `frame` is zero-copy, backed by a read-only GStreamer buffer.
                    // Copy it so the frame data outlives the reader.
                    let owned_rgb =
                        Image::<u8, 3>::from_size_slice(frame.size(), frame.as_slice())?;
                    let mut gray = Image::<u8, 1>::from_size_val(frame.size(), 0)?;
                    gray_from_rgb_u8(&owned_rgb, &mut gray)?;
                    rgb_frames.push(owned_rgb);
                    gray_frames.push(gray);
                }
                frame_idx += 1;
            }
            None => {
                // `grab_rgb8()` returns `None` while the pipeline is still
                // buffering; it does not mean end-of-stream. Detect the end by
                // comparing playback position against the total duration.
                if let (Some(pos), Some(dur)) = (reader.get_pos(), reader.get_duration()) {
                    if pos >= dur {
                        break;
                    }
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        }
    }

    reader.close()?;

    Ok((rgb_frames, gray_frames))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_frames_errors_on_missing_file() {
        let result = read_frames(Path::new("/nonexistent/video.mp4"), 1);
        match result {
            Ok(_) => panic!("reading a missing video must fail"),
            Err(e) => {
                // The exact error variant is GStreamer-specific; just assert we got one.
                assert!(!e.to_string().is_empty());
            }
        }
    }

    #[test]
    #[should_panic(expected = "frame_step must be >= 1")]
    fn read_frames_rejects_zero_frame_step() {
        let _ = read_frames(Path::new("x.mp4"), 0);
    }
}
