//! Cost of `StreamCapture::grab_rgb8` for tightly packed versus padded RGB rows.
//!
//! `videotestsrc` runs unsynchronised, so the 5-frame ring buffer stays full, and only the
//! `grab_rgb8` calls that return a frame are timed. Widths divisible by 4 are tightly packed
//! and borrowed from the GStreamer buffer (zero-copy). The other widths are padded by
//! GStreamer's default layout, so each grab copies the rows into a packed image.

use criterion::{BenchmarkId, Criterion};
use kornia_io::stream::StreamCapture;
use std::{
    hint::black_box,
    time::{Duration, Instant},
};

fn bench_grab_rgb8_layout(c: &mut Criterion) {
    let mut group = c.benchmark_group("grab_rgb8");
    for (width, height) in [(856, 480), (854, 480), (1920, 1080), (1918, 1080)] {
        let layout = if (width * 3) % 4 == 0 {
            "packed"
        } else {
            "padded"
        };
        let pipeline_desc = format!(
            "videotestsrc pattern=black ! \
             video/x-raw,format=RGB,width={width},height={height},framerate=30/1 ! \
             appsink name=sink sync=false"
        );
        let mut capture = StreamCapture::new(&pipeline_desc).expect("Failed to create pipeline");
        capture.start().expect("Failed to start pipeline");

        group.bench_function(BenchmarkId::new(layout, format!("{width}x{height}")), |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    // Fail instead of hanging if the pipeline stops producing frames.
                    let deadline = Instant::now() + Duration::from_secs(5);
                    loop {
                        let start = Instant::now();
                        let frame = capture.grab_rgb8().expect("Failed to grab the image");
                        let elapsed = start.elapsed();
                        if let Some(frame) = frame {
                            total += elapsed;
                            black_box(frame);
                            break;
                        }
                        assert!(
                            Instant::now() < deadline,
                            "no frame from the pipeline within 5 s"
                        );
                        // Only the grab is timed, so yielding here does not bias the result.
                        std::thread::yield_now();
                    }
                }
                total
            });
        });

        capture.close().expect("Failed to close pipeline");
    }
    group.finish();
}

criterion::criterion_group! {
    name = benches;
    config = Criterion::default().warm_up_time(Duration::from_secs(1));
    targets = bench_grab_rgb8_layout
}

criterion::criterion_main!(benches);
