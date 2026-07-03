//! Camera → model-input tensor in ONE fused CUDA kernel.
//!
//! Captures raw YUYV frames from a V4L2 camera and turns each one into a
//! normalized `[1, 3, S, S]` CHW `f32` tensor with a single kernel launch:
//! the YUV→RGB decode happens *inside* the resize taps
//! (`Preprocessor::run_raw` with `SourceFormat::Yuyv`) — no intermediate RGB
//! image ever exists in device memory.
//!
//! The upload path is production-shaped too: one page-locked staging buffer
//! and one device buffer, both allocated once and reused (pinned H2D is a
//! straight DMA; `cuMemHostAlloc` is far too expensive per frame).
//!
//! ```text
//! cargo run -p cuda-camera-preprocess --release -- [--device /dev/video0] [--frames 100] [--out-size 640]
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
    use kornia_image::ImageSize;
    use kornia_imgproc::preprocess::{Normalize, Preprocessor, ResizeMode, SourceFormat};
    use kornia_io::v4l::{PixelFormat, V4LCameraConfig, V4lVideoCapture};

    #[derive(FromArgs)]
    /// Fused camera→tensor preprocessing on CUDA.
    struct Args {
        /// V4L2 device path
        #[argh(option, default = "String::from(\"/dev/video0\")")]
        device: String,
        /// number of frames to process
        #[argh(option, default = "100")]
        frames: usize,
        /// square output size (model input side)
        #[argh(option, default = "640")]
        out_size: usize,
    }

    pub fn demo() -> Result<(), Box<dyn std::error::Error>> {
        let args: Args = argh::from_env();

        // ── Camera: raw (uncompressed) YUYV frames ──
        let mut cam = match V4lVideoCapture::new(V4LCameraConfig {
            device_path: args.device.clone(),
            size: ImageSize {
                width: 640,
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
                "Camera negotiated {} instead of YUYV — fused decode needs a raw \
                 YUYV stream (MJPG cameras must decode on CPU first).",
                cam.pixel_format()
            );
            return Ok(());
        }
        let size = cam.size();
        let (w, h) = (size.width, size.height);
        let frame_len = SourceFormat::Yuyv.buffer_len(w, h);
        println!(
            "Camera: {} {}x{} YUYV ({frame_len} B/frame)",
            args.device, w, h
        );

        // ── CUDA: fused decode+resize+normalize preprocessor (one JIT compile) ──
        let stream = CudaContext::new(0)?.default_stream();
        let pre = Preprocessor::builder()
            .mode(ResizeMode::Letterbox)
            .source_format(SourceFormat::Yuyv)
            .normalize(Normalize::MeanStd {
                mean: [0.485, 0.456, 0.406],
                std: [0.229, 0.224, 0.225],
            })
            .build_cuda(stream.clone())?;

        // Persistent buffers: page-locked host staging + device frame + output.
        let mut pinned = kornia_tensor::zeros_pinned::<u8, 1>([frame_len], stream.context())?;
        let mut d_frame = stream.alloc_zeros::<u8>(frame_len)?;
        let s = args.out_size;
        let mut d_out = kornia_tensor::zeros_cuda::<f32, 4>([1, 3, s, s], &stream)?;

        // ── Frame loop ──
        let (mut t_grab, mut t_gpu) = (0.0f64, 0.0f64);
        let mut processed = 0usize;
        let t_total = Instant::now();
        while processed < args.frames {
            let t0 = Instant::now();
            let Some(frame) = cam.grab_frame()? else {
                continue;
            };
            if frame.buffer.len() < frame_len {
                continue; // short/corrupt buffer
            }
            pinned
                .as_slice_mut()
                .copy_from_slice(&frame.buffer.as_slice()[..frame_len]);
            let t1 = Instant::now();

            // Pinned DMA + one fused kernel + sync = camera bytes → model tensor.
            stream.memcpy_htod(pinned.as_slice(), &mut d_frame)?;
            pre.run_raw(&d_frame, w, h, &mut d_out)?;
            stream.synchronize()?;
            let t2 = Instant::now();

            t_grab += (t1 - t0).as_secs_f64();
            t_gpu += (t2 - t1).as_secs_f64();
            processed += 1;
        }
        let total = t_total.elapsed().as_secs_f64();

        // Prove real data flowed: download and summarize the last tensor.
        let host = stream.clone_dtoh(d_out.as_cudaslice().expect("device tensor"))?;
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        let mean = host.iter().fold(0.0f64, |a, &v| {
            lo = lo.min(v);
            hi = hi.max(v);
            a + v as f64
        }) / host.len() as f64;

        let n = processed as f64;
        println!(
            "Processed {processed} frames in {total:.2}s ({:.1} FPS end-to-end)",
            n / total
        );
        println!(
            "  grab+stage: {:.3} ms/frame | H2D+fused kernel+sync: {:.3} ms/frame",
            t_grab / n * 1e3,
            t_gpu / n * 1e3
        );
        println!(
            "  last tensor [1, 3, {s}, {s}]: min {lo:.3} max {hi:.3} mean {mean:.3} (ImageNet-normalized)"
        );
        Ok(())
    }
}
