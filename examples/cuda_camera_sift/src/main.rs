//! Live SIFT tracking on CUDA — camera to matched correspondences, on device.
//!
//! Captures V4L2 frames and, for each one, detects SIFT keypoints, describes
//! them, and matches against the previous frame — **without the descriptors ever
//! touching host memory**. `detect_and_compute` leaves the descriptor
//! block on the GPU and `descriptors` hands it straight to the matcher,
//! so the only per-frame download is the keypoint list and the final pair
//! indices. That is the whole point of the device API: a host round trip of two
//! 2515x128 descriptor blocks costs more than the detection that produced them.
//!
//! The configuration matters as much as the code. Two knobs turn a 19 ms
//! detector into an 8 ms one on a Jetson Orin at 752x480:
//!
//! * `n_features` — a keypoint budget. Descriptors are computed *after* the
//!   budget is applied (as `cv::SIFT` does), so cost scales with the budget
//!   rather than with however many extrema the frame happens to contain. A
//!   frontend tracking 500 features does not pay for 2515.
//! * `fast_descriptor` — a rotated-frame descriptor and shared-atomic
//!   orientation. **Not bit-exact** against OpenCV, and validated geometrically
//!   instead; measured epipolar inlier ratio 63.1% against the exact path's
//!   65.3% on mh01. Leave it off if you need byte-identical output.
//!
//! Measured on an Orin Nano Super at 752x480, medians after warm-up:
//!
//! ```text
//! exact, no budget          18.6 ms      exact + 500 budget   12.4 ms
//! fast,  no budget          10.2 ms      fast  + 500 budget    8.1 ms
//! ```
//!
//! ```text
//! cargo run -p cuda-camera-sift --release -- [--device /dev/video0] [--frames 200]
//!                                            [--features 500] [--fast] [--ratio 0.8]
//! ```

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    linux::demo()
}

#[cfg(not(target_os = "linux"))]
fn main() {
    panic!("V4L2 capture is Linux-only.");
}

#[cfg(target_os = "linux")]
mod linux {
    use std::time::Instant;

    use argh::FromArgs;
    use cudarc::driver::CudaContext;
    use kornia_image::{Image, ImageSize};
    use kornia_imgproc::cuda::sift::{
        FirstOctave, SiftCuda, SiftCudaConfig, SiftCudaFeatures, SiftMatcher, DESCR_LEN,
    };
    use kornia_io::v4l::{PixelFormat, V4LCameraConfig, V4lVideoCapture};

    #[derive(FromArgs)]
    /// Live SIFT detection and frame-to-frame matching on CUDA.
    struct Args {
        /// V4L2 device path
        #[argh(option, default = "String::from(\"/dev/video0\")")]
        device: String,
        /// number of frames to process
        #[argh(option, default = "200")]
        frames: usize,
        /// keypoint budget; 0 keeps every keypoint
        #[argh(option, default = "500")]
        features: usize,
        /// use the faster, non-bit-exact descriptor and orientation
        #[argh(switch)]
        fast: bool,
        /// lowe ratio for the match test
        #[argh(option, default = "0.8")]
        ratio: f32,
    }

    pub fn demo() -> Result<(), Box<dyn std::error::Error>> {
        let args: Args = argh::from_env();

        // ── Camera ──
        // YUYV so the luma plane can be strided out directly; SIFT wants a
        // single grayscale channel and YUYV already carries one.
        let mut cam = match V4lVideoCapture::new(V4LCameraConfig {
            device_path: args.device.clone(),
            size: ImageSize {
                width: 752,
                height: 480,
            },
            fps: 30,
            format: PixelFormat::YUYV,
            buffer_size: 4,
        }) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Cannot open {} ({e}).", args.device);
                eprintln!("Connect a V4L2 camera or pass --device /dev/videoN.");
                return Ok(());
            }
        };
        if cam.pixel_format() != PixelFormat::YUYV {
            eprintln!(
                "Camera negotiated {} instead of YUYV; this example strides the \
                 luma plane out of a raw YUYV buffer.",
                cam.pixel_format()
            );
            return Ok(());
        }
        let size = cam.size();
        let (w, h) = (size.width, size.height);
        println!("Camera: {} {w}x{h} YUYV", args.device);

        // ── CUDA: one plan, reused for every frame ──
        // The plan owns every scratch buffer the pipeline needs — the whole
        // Gaussian pyramid, the DoG stack, the keypoint and descriptor slabs.
        // Building it per frame would allocate tens of megabytes each time and
        // recompile nothing, which is exactly what a streaming caller must not
        // do. Sized once for this camera's geometry.
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let cfg = SiftCudaConfig {
            n_features: args.features,
            ..SiftCudaConfig::default()
        };
        let max_kp = cfg.max_keypoints;
        let mut sift = SiftCuda::new(&ctx, &stream, w, h, cfg, FirstOctave::Double, 8)?;
        sift.set_fast_descriptor(args.fast);
        let mut matcher = SiftMatcher::new(&stream, max_kp)?;

        // Host staging for the grayscale plane, allocated once; each frame
        // goes to the device through the `Image::to_cuda` API.
        let mut gray = vec![0.0f32; w * h];
        let size = ImageSize {
            width: w,
            height: h,
        };

        // Each frame's result OWNS its device descriptors, so keeping the
        // previous frame alive for matching is just holding the struct — no
        // copy, no invalidation.
        let mut prev: Option<SiftCudaFeatures> = None;

        println!(
            "SIFT: budget {} | {} descriptor | ratio {}",
            if args.features == 0 {
                "none".to_string()
            } else {
                args.features.to_string()
            },
            if args.fast { "fast" } else { "exact" },
            args.ratio
        );

        // Warm up: the first call JIT-compiles every kernel (~1.2 s). Timing it
        // would attribute the compile to whichever stage ran first, a mistake
        // this module has made before.
        if let Some(frame) = cam.grab_frame()? {
            luma_to_f32(frame.buffer.as_slice(), &mut gray, w, h);
            let dev = Image::<f32, 1>::from_size_slice(size, &gray)?.to_cuda(&stream)?;
            sift.detect_and_compute(&ctx, &stream, &dev)?;
            stream.synchronize()?;
            println!("Kernels compiled.\n");
        }

        // ── Frame loop ──
        let (mut t_grab, mut t_det, mut t_match) = (0.0f64, 0.0f64, 0.0f64);
        let (mut n_kp, mut n_pairs, mut processed) = (0usize, 0usize, 0usize);
        let t_total = Instant::now();

        while processed < args.frames {
            let t0 = Instant::now();
            let Some(frame) = cam.grab_frame()? else {
                continue;
            };
            let buf = frame.buffer.as_slice();
            if buf.len() < w * h * 2 {
                continue; // short or corrupt buffer
            }
            luma_to_f32(buf, &mut gray, w, h);
            let dev = Image::<f32, 1>::from_size_slice(size, &gray)?.to_cuda(&stream)?;
            let t1 = Instant::now();

            // Detect + describe. Descriptors stay on device, owned by `cur`.
            let cur = sift.detect_and_compute(&ctx, &stream, &dev)?;
            stream.synchronize()?;
            let t2 = Instant::now();

            // Match against the previous frame, device to device.
            let n_cur = cur.len();
            let pairs = match &prev {
                Some(p) if !p.is_empty() && n_cur > 0 => matcher.match_descriptors(
                    &ctx,
                    &stream,
                    &cur.descriptors.slice(0..n_cur * DESCR_LEN),
                    n_cur,
                    &p.descriptors.slice(0..p.len() * DESCR_LEN),
                    p.len(),
                    args.ratio,
                    true,
                )?,
                _ => Vec::new(),
            };
            let t3 = Instant::now();

            // This frame simply becomes the previous one; its descriptors come
            // with it because the result owns them.
            prev = Some(cur);

            t_grab += (t1 - t0).as_secs_f64();
            t_det += (t2 - t1).as_secs_f64();
            t_match += (t3 - t2).as_secs_f64();
            n_kp += n_cur;
            n_pairs += pairs.len();
            processed += 1;

            if processed % 30 == 0 {
                println!(
                    "  frame {processed:>4}: {n_cur:>4} kp, {:>4} matches",
                    pairs.len()
                );
            }
        }

        let total = t_total.elapsed().as_secs_f64();
        let n = processed as f64;
        println!(
            "\nProcessed {processed} frames in {total:.2}s ({:.1} FPS end-to-end)",
            n / total
        );
        println!(
            "  grab+upload {:.2} ms | detect+describe {:.2} ms | match {:.2} ms",
            t_grab / n * 1e3,
            t_det / n * 1e3,
            t_match / n * 1e3,
        );
        println!(
            "  {:.0} keypoints/frame, {:.0} matches/frame",
            n_kp as f64 / n,
            n_pairs as f64 / n,
        );
        Ok(())
    }

    /// Stride the luma plane out of a packed YUYV buffer.
    ///
    /// YUYV stores `Y0 U Y1 V` per two pixels, so luma is every other byte —
    /// no colour conversion is needed for a detector that only wants intensity.
    /// The reference works in 0..255 floats, not 0..1: normalising here would
    /// change what `contrast_threshold` means and collapse the keypoint count.
    fn luma_to_f32(buf: &[u8], out: &mut [f32], w: usize, h: usize) {
        for (dst, src) in out[..w * h].iter_mut().zip(buf.chunks_exact(2)) {
            *dst = src[0] as f32;
        }
    }
}
