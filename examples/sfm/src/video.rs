//! Video frame extraction for the SfM pipeline.
//!
//! Reads an MP4 (or any GStreamer-decodable video) into a sequence of frames.
//! The video reader always decodes to RGB, so this module returns both the
//! owned RGB frames (used later to colour the reconstructed point cloud) and
//! matching grayscale frames (used for feature detection).
//!
//! Reading emits live progress to stderr (updated in place) plus a timing
//! breakdown when it finishes, so slow videos can be diagnosed.

use std::error::Error;
use std::io::Write;
use std::path::Path;
use std::time::{Duration, Instant};

use kornia_image::Image;
use kornia_imgproc::color::gray_from_rgb_u8;
use kornia_io::gstreamer::video::{ImageFormat, VideoReader};

/// Owned frames decoded from a video: parallel RGB and grayscale buffers.
pub type VideoFrames = (Vec<Image<u8, 3>>, Vec<Image<u8, 1>>);

/// How often (in frames) the live progress line is refreshed.
const PROGRESS_INTERVAL: usize = 10;

/// Cumulative timing and progress counters for one `read_frames` call.
struct ReadStats {
    /// Time to construct and start the GStreamer pipeline.
    init: Duration,
    /// Cumulative time inside `grab_rgb8()` (buffer pop + map).
    grab: Duration,
    /// Cumulative time allocating the owned RGB + gray frames.
    alloc: Duration,
    /// Cumulative time converting RGB → gray.
    convert: Duration,
    /// Frames returned by the pipeline (including skipped ones).
    frames_grabbed: usize,
    /// Frames actually kept after `frame_step` subsampling.
    frames_kept: usize,
    /// Start of the whole read.
    start: Instant,
}

impl ReadStats {
    fn new() -> Self {
        Self {
            init: Duration::ZERO,
            grab: Duration::ZERO,
            alloc: Duration::ZERO,
            convert: Duration::ZERO,
            frames_grabbed: 0,
            frames_kept: 0,
            start: Instant::now(),
        }
    }

    /// Refresh the single-line progress indicator (every `PROGRESS_INTERVAL`).
    fn progress(&self) {
        if self.frames_grabbed.is_multiple_of(PROGRESS_INTERVAL) {
            let elapsed = self.start.elapsed();
            let fps = self.frames_grabbed as f64 / elapsed.as_secs_f64();
            eprint!(
                "\r[video] decoded {} frames, kept {} | {fps:.1} fps | elapsed {:.1}s",
                self.frames_grabbed,
                self.frames_kept,
                elapsed.as_secs_f64()
            );
            let _ = std::io::stderr().flush();
        }
    }

    /// Print a timing breakdown once the read finishes.
    fn summary(&self) {
        let total = self.start.elapsed();
        let fps = self.frames_grabbed as f64 / total.as_secs_f64().max(1e-9);
        let pct = |d: Duration| 100.0 * d.as_secs_f64() / total.as_secs_f64().max(1e-9);
        eprintln!();
        eprintln!("[video] reading complete:");
        eprintln!(
            "  total {:.1}s | {fps:.1} fps | kept {kept} / {grabbed} decoded",
            total.as_secs_f64(),
            kept = self.frames_kept,
            grabbed = self.frames_grabbed,
        );
        eprintln!(
            "  init    {:.3}s ({:.1}%)",
            self.init.as_secs_f64(),
            pct(self.init)
        );
        eprintln!(
            "  grab    {:.3}s ({:.1}%)",
            self.grab.as_secs_f64(),
            pct(self.grab)
        );
        eprintln!(
            "  alloc   {:.3}s ({:.1}%)",
            self.alloc.as_secs_f64(),
            pct(self.alloc)
        );
        eprintln!(
            "  convert {:.3}s ({:.1}%)",
            self.convert.as_secs_f64(),
            pct(self.convert)
        );
    }
}

/// Read a video file into a sequence of RGB and grayscale frames.
///
/// Returns `(rgb_frames, gray_frames)`, both subsampled by `frame_step`:
/// only frames whose index is a multiple of `frame_step` are kept. A
/// `frame_step` of `1` keeps every frame.
///
/// Live progress and a timing breakdown are printed to stderr.
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

    let mut stats = ReadStats::new();

    // NOTE: `VideoReader` only decodes to RGB via `grab_rgb8()` regardless of
    // the requested `ImageFormat`; requesting `Mono8` makes GStreamer produce
    // 1-byte frames that the RGB validator rejects at runtime. So we always
    // request RGB and convert to grayscale ourselves.
    let init_start = Instant::now();
    let mut reader = VideoReader::new(path, ImageFormat::Rgb8)?;
    reader.start()?;
    stats.init = init_start.elapsed();

    let mut rgb_frames: Vec<Image<u8, 3>> = Vec::new();
    let mut gray_frames: Vec<Image<u8, 1>> = Vec::new();
    let mut frame_idx: usize = 0;
    let mut seen_any = false;
    let mut consecutive_none: usize = 0;

    loop {
        let grab_start = Instant::now();
        let grabbed = reader.grab_rgb8()?;
        stats.grab += grab_start.elapsed();

        match grabbed {
            Some(frame) => {
                seen_any = true;
                consecutive_none = 0;
                stats.frames_grabbed += 1;
                if frame_idx.is_multiple_of(frame_step) {
                    // `frame` is zero-copy, backed by a read-only GStreamer buffer.
                    // Copy it so the frame data outlives the reader.
                    let alloc_start = Instant::now();
                    let owned_rgb =
                        Image::<u8, 3>::from_size_slice(frame.size(), frame.as_slice())?;
                    let mut gray = Image::<u8, 1>::from_size_val(frame.size(), 0)?;
                    stats.alloc += alloc_start.elapsed();

                    let convert_start = Instant::now();
                    gray_from_rgb_u8(&owned_rgb, &mut gray)?;
                    stats.convert += convert_start.elapsed();

                    rgb_frames.push(owned_rgb);
                    gray_frames.push(gray);
                    stats.frames_kept += 1;
                }
                frame_idx += 1;
                stats.progress();
            }
            None => {
                consecutive_none += 1;
                // `grab_rgb8()` returns `None` while the pipeline is still
                // buffering; it does not mean end-of-stream. Prefer detecting
                // the end via playback position, but some files report a bad
                // duration (e.g. a malformed moov atom), so fall back to a
                // sustained silence after having seen frames.
                if let (Some(pos), Some(dur)) = (reader.get_pos(), reader.get_duration()) {
                    if pos >= dur {
                        break;
                    }
                }
                if seen_any && consecutive_none > 30 {
                    break;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        }
    }

    reader.close()?;

    stats.summary();

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
