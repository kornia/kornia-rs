use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use kornia_image::Image;
use kornia_imgproc::threshold::AdaptiveThresholdMean;
use opencv::{core, imgproc, prelude::*};
use std::hint::black_box;

const OFFSET: i16 = 10;
const MAX_VALUE: u8 = 255;

/// Compare adaptive mean thresholding in kornia and OpenCV.
fn bench_adaptive_mean(c: &mut Criterion) {
    // The kornia path uses one thread. Make OpenCV also use one thread.
    core::set_num_threads(1).unwrap();
    core::set_use_opencl(false).unwrap();

    let mut group = c.benchmark_group("adaptive_mean");

    for (width, height) in [(640, 480), (1920, 1080)] {
        let src = Image::<u8, 1>::new(
            [width, height].into(),
            (0..width * height)
                .map(|i| ((i * 73 + i / width * 11) % 256) as u8)
                .collect(),
        )
        .unwrap();
        let mut dst = Image::from_size_val(src.size(), 0).unwrap();

        // The OpenCV matrix uses the same pixels as the kornia image.
        let cv_src_flat = core::Mat::from_slice(src.as_slice()).unwrap();
        let cv_src = cv_src_flat.reshape(1, height as i32).unwrap();
        let mut cv_dst = core::Mat::default();

        group.throughput(Throughput::Elements((width * height) as u64));

        for block_size in [3, 5, 15, 31, 51, 67, 101] {
            let parameter = format!("{width}x{height}/{block_size}");

            let mut threshold = AdaptiveThresholdMean::new(block_size, OFFSET, MAX_VALUE).unwrap();
            imgproc::adaptive_threshold(
                &cv_src,
                &mut cv_dst,
                MAX_VALUE as f64,
                imgproc::ADAPTIVE_THRESH_MEAN_C,
                imgproc::THRESH_BINARY,
                block_size as i32,
                OFFSET as f64,
            )
            .unwrap();

            // The first call fills the internal scratch. It also lets the
            // bench confirm that both outputs are the same before timing.
            threshold.apply(&src, &mut dst).unwrap();
            assert_eq!(
                dst.as_slice(),
                cv_dst.data_bytes().unwrap(),
                "kornia and OpenCV outputs differ at {parameter}"
            );

            group.bench_function(BenchmarkId::new("kornia", &parameter), |b| {
                b.iter(|| {
                    threshold.apply(black_box(&src), &mut dst).unwrap();
                    black_box(dst.as_slice());
                })
            });

            group.bench_function(BenchmarkId::new("opencv", &parameter), |b| {
                b.iter(|| {
                    imgproc::adaptive_threshold(
                        black_box(&cv_src),
                        &mut cv_dst,
                        MAX_VALUE as f64,
                        imgproc::ADAPTIVE_THRESH_MEAN_C,
                        imgproc::THRESH_BINARY,
                        block_size as i32,
                        OFFSET as f64,
                    )
                    .unwrap();
                })
            });
        }
    }

    group.finish();
}

criterion_group!(benches, bench_adaptive_mean);
criterion_main!(benches);
