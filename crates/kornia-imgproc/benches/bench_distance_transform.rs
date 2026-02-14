use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use kornia_image::{allocator::CpuAllocator, Image, ImageSize};
use kornia_imgproc::distance_transform::{distance_transform_vanilla, DistanceTransformExecutor};
use opencv::{core, imgproc, prelude::*};

fn bench_distance_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("DistanceTransform");

    for (width, height) in [(64, 64)].iter() {
        group.throughput(Throughput::Elements((*width * *height) as u64));
        let parameter_string = format!("{width}x{height}");

        let mut data = vec![0.0f32; width * height];
        for i in 0..(*width).min(*height) {
            if i % 10 == 0 {
                data[i * width + i] = 1.0;
            }
        }

        let image = Image::<f32, 1, _>::new(
            ImageSize {
                width: *width,
                height: *height,
            },
            data.clone(),
            CpuAllocator,
        )
        .unwrap();

        // OpenCV Setup
        let mut cv_src = core::Mat::zeros(*height as i32, *width as i32, core::CV_8UC1)
            .unwrap()
            .to_mat()
            .unwrap();
        for i in 0..(*width).min(*height) {
            if i % 10 == 0 {
                *cv_src.at_2d_mut::<u8>(i as i32, i as i32).unwrap() = 255;
            }
        }
        let mut cv_dst = core::Mat::default();

        // 1. Vanilla (Baseline)
        group.bench_with_input(
            BenchmarkId::new("vanilla", &parameter_string),
            &image,
            |b, i| b.iter(|| std::hint::black_box(distance_transform_vanilla(i))),
        );

        // 2. Felzenszwalb
        group.bench_with_input(
            BenchmarkId::new("kornia_cpu", &parameter_string),
            &image,
            |b, i| {
                let mut executor = DistanceTransformExecutor::new();
                b.iter(|| executor.execute(i))
            },
        );

        // 3. OpenCV
        group.bench_function(BenchmarkId::new("opencv", &parameter_string), |b| {
            b.iter(|| {
                imgproc::distance_transform(
                    &cv_src,
                    &mut cv_dst,
                    imgproc::DIST_L2,
                    imgproc::DIST_MASK_5,
                    core::CV_32F,
                )
                .unwrap()
            })
        });
    }

    for (width, height) in [(512, 512), (1024, 1024), (2048, 2048)].iter() {
        group.throughput(Throughput::Elements((*width * *height) as u64));
        let parameter_string = format!("{width}x{height}");

        let mut data = vec![0.0f32; width * height];
        for i in 0..(*width).min(*height) {
            if i % 10 == 0 {
                data[i * width + i] = 1.0;
            }
        }

        let image = Image::<f32, 1, _>::new(
            ImageSize {
                width: *width,
                height: *height,
            },
            data,
            CpuAllocator,
        )
        .unwrap();

        let mut cv_src = core::Mat::zeros(*height as i32, *width as i32, core::CV_8UC1)
            .unwrap()
            .to_mat()
            .unwrap();
        for i in 0..(*width).min(*height) {
            if i % 10 == 0 {
                *cv_src.at_2d_mut::<u8>(i as i32, i as i32).unwrap() = 255;
            }
        }
        let mut cv_dst = core::Mat::default();

        // 1.Felzenszwalb (STRUCT REUSE)
        group.bench_with_input(
            BenchmarkId::new("kornia_cpu", &parameter_string),
            &image,
            |b, i| {
                let mut executor = DistanceTransformExecutor::new();
                b.iter(|| executor.execute(i))
            },
        );

        // 2. OpenCV
        group.bench_function(BenchmarkId::new("opencv", &parameter_string), |b| {
            b.iter(|| {
                imgproc::distance_transform(
                    &cv_src,
                    &mut cv_dst,
                    imgproc::DIST_L2,
                    imgproc::DIST_MASK_5,
                    core::CV_32F,
                )
                .unwrap()
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_distance_transform);
criterion_main!(benches);
