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
use kornia_io::gstreamer::Grab;

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
/// Returns an error if the video cannot be opened, its pipeline fails (an
/// undecodable file, a container with no video stream such as audio-only, a
/// missing decoder), or a frame cannot be converted to grayscale. Reaching the
/// end of the stream is not an error: the frames decoded so far are returned.
pub fn read_frames(path: &Path, frame_step: usize) -> Result<VideoFrames, Box<dyn Error>> {
    assert!(frame_step >= 1, "frame_step must be >= 1");

    let mut stats = ReadStats::new();

    // Request RGB because point colours are sampled from RGB frames; the
    // grayscale images used for features are converted here from the same frame.
    // (`try_grab_gray8` also exists, but grabbing one stream twice would step
    // the pipeline twice.)
    let init_start = Instant::now();
    let mut reader = VideoReader::new(path, ImageFormat::Rgb8)?;
    reader.start()?;
    stats.init = init_start.elapsed();

    let mut rgb_frames: Vec<Image<u8, 3>> = Vec::new();
    let mut gray_frames: Vec<Image<u8, 1>> = Vec::new();
    let mut frame_idx: usize = 0;

    loop {
        let grab_start = Instant::now();
        let grabbed = reader.try_grab_rgb8()?;
        stats.grab += grab_start.elapsed();

        match grabbed {
            Grab::Frame(frame) => {
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
            // Still running, nothing buffered yet: back off and retry. The
            // reader reports the terminal states itself — a failed pipeline
            // (undecodable file, audio-only container, missing decoder)
            // surfaces as `Err(StreamError)` from `try_grab_rgb8`, and the
            // end of the stream as `Grab::Eos` — so this arm never has to
            // guess that the stream is dead.
            Grab::Pending => std::thread::sleep(Duration::from_millis(10)),
            // End of stream: every decoded frame has been grabbed.
            Grab::Eos => break,
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

    use kornia_image::ImageSize;
    use kornia_io::gstreamer::video::{VideoCodec, VideoWriter};

    const TEST_SIZE: ImageSize = ImageSize {
        width: 64,
        height: 48,
    };

    /// Runs `f` on its own thread, panicking if it has not returned within
    /// `timeout`. The regressions this module guards against are infinite read
    /// loops, which a plain assertion can never fail — only a deadline can.
    fn finishes_within<T: Send + 'static>(
        timeout: Duration,
        f: impl FnOnce() -> T + Send + 'static,
    ) -> T {
        let handle = std::thread::spawn(f);
        let deadline = Instant::now() + timeout;
        loop {
            if handle.is_finished() {
                return handle
                    .join()
                    .unwrap_or_else(|e| std::panic::resume_unwind(e));
            }
            assert!(
                Instant::now() < deadline,
                "operation did not finish within {timeout:?}"
            );
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// Encodes one solid-colour frame per `colours` entry into an H.264 MP4.
    ///
    /// Returns `Ok(None)` when `x264enc` is not installed (the caller skips the
    /// round-trip test), `Ok(Some(()))` once the file is closed, and `Err` for
    /// any other writer failure, so encoding problems cannot silently skip the
    /// test.
    fn write_solid_mp4(
        path: &Path,
        colours: &[(u8, u8, u8)],
    ) -> Result<Option<()>, Box<dyn std::error::Error>> {
        let mut writer =
            match VideoWriter::new(path, VideoCodec::H264, ImageFormat::Rgb8, 30, TEST_SIZE) {
                Ok(writer) => writer,
                // The only expected skip: the encoder plugin is not installed.
                Err(e) if e.to_string().contains("x264enc") => {
                    eprintln!("skipping video round-trip test: {e}");
                    return Ok(None);
                }
                Err(e) => return Err(e.into()),
            };
        writer.start()?;
        for &(r, g, b) in colours {
            let bytes: Vec<u8> = std::iter::repeat_n([r, g, b], TEST_SIZE.width * TEST_SIZE.height)
                .flatten()
                .collect();
            let frame = Image::<u8, 3>::from_size_slice(TEST_SIZE, &bytes)?;
            writer.write(&frame)?;
        }
        writer.close()?;
        Ok(Some(()))
    }

    /// A pipeline that cannot decode must surface an error, not spin forever.
    #[test]
    fn read_frames_errors_on_non_media_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("not-a-video.bin");
        std::fs::write(&path, b"this is not a media file").unwrap();

        // decodebin rejects the file on the bus; try_grab_rgb8 must report it.
        let result = finishes_within(Duration::from_secs(30), move || {
            read_frames(&path, 1).map_err(|e| e.to_string())
        });
        match result {
            Ok((rgb, _)) => panic!("a non-media file must fail, got {} frames", rgb.len()),
            Err(e) => assert!(!e.is_empty(), "error must carry a message"),
        }
    }

    /// End-of-stream must deliver every decoded frame, with no truncation.
    #[test]
    fn read_frames_reads_every_frame_and_stops_at_eos() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("frames.mp4");
        // Solid colours with r < g < b keep channel order observable even after
        // the lossy H.264 round-trip; rising r per frame keeps order observable.
        let colours: Vec<(u8, u8, u8)> = (0..7)
            .map(|i| (i * 20 + 10, i * 20 + 26, i * 20 + 42))
            .collect();
        if write_solid_mp4(&path, &colours).unwrap().is_none() {
            return;
        }

        let (rgb, gray) = finishes_within(Duration::from_secs(30), move || {
            read_frames(&path, 1).map_err(|e| e.to_string())
        })
        .unwrap();

        assert_eq!(rgb.len(), colours.len(), "every decoded frame is returned");
        assert_eq!(gray.len(), colours.len(), "RGB and gray stay in step");
        for (i, (frame, gray_frame)) in rgb.iter().zip(&gray).enumerate() {
            let (er, eg, eb) = colours[i];
            assert_eq!(
                (frame.width(), frame.height()),
                (TEST_SIZE.width, TEST_SIZE.height)
            );
            for px in frame.as_slice().chunks_exact(3) {
                // Uniform source must decode uniformly (no stride shearing).
                let (r, g, b) = (px[0], px[1], px[2]);
                // H.264 is lossy: allow a few levels of quantization error.
                // The `r < g < b` check below is what catches colour rotation.
                assert!(
                    r.abs_diff(er) <= 8 && g.abs_diff(eg) <= 8 && b.abs_diff(eb) <= 8,
                    "expected solid ({er}, {eg}, {eb}), got ({r}, {g}, {b})"
                );
                // Channel order survives the encode/decode (no colour rotation).
                assert!(r < g && g < b, "channel order changed: ({r}, {g}, {b})");
            }
            // Every gray pixel must equal the OpenCV Q14 oracle applied to the
            // reconstructed RGB pixel (kornia-imgproc's fixed-point weights).
            let (r, g, b) = (
                frame.as_slice()[0],
                frame.as_slice()[1],
                frame.as_slice()[2],
            );
            let expected_gray =
                ((4899 * r as u32 + 9617 * g as u32 + 1868 * b as u32 + 8192) >> 14) as u8;
            for &v in gray_frame.as_slice() {
                assert!(
                    v.abs_diff(expected_gray) <= 1,
                    "gray {v} != expected {expected_gray} for RGB ({r}, {g}, {b})"
                );
            }
        }
    }

    /// `frame_step` keeps exactly the indexed frames, in order.
    #[test]
    fn read_frames_subsamples_with_frame_step() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("subsample.mp4");
        let colours: Vec<(u8, u8, u8)> = (0..6)
            .map(|i| (i * 20 + 10, i * 20 + 26, i * 20 + 42))
            .collect();
        if write_solid_mp4(&path, &colours).unwrap().is_none() {
            return;
        }

        let (rgb, gray) = finishes_within(Duration::from_secs(30), move || {
            read_frames(&path, 3).map_err(|e| e.to_string())
        })
        .unwrap();
        assert_eq!(rgb.len(), 2, "indices 0 and 3 are kept");
        assert_eq!(gray.len(), 2);
        for (frame, expected) in rgb.iter().zip([&colours[0], &colours[3]]) {
            let (er, _eg, _eb) = *expected;
            let mean_r = frame
                .as_slice()
                .chunks_exact(3)
                .map(|px| px[0] as u32)
                .sum::<u32>()
                / (TEST_SIZE.width * TEST_SIZE.height) as u32;
            assert!(
                mean_r.abs_diff(er as u32) <= 4,
                "frame mean r {mean_r} != expected {er}"
            );
        }
    }
}
