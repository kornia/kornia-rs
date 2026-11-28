use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use kornia_image::{Image, ImageSize};
use kornia_imgproc::threshold::{compute_otsu_threshold, threshold_binary, threshold_otsu};

const SIZES: [(usize, usize); 3] = [(256, 256), (640, 480), (1920, 1080)];

fn bench_threshold(c: &mut Criterion) {
    let mut group = c.benchmark_group("Threshold");

    for (w, h) in SIZES {
        let parameter_string = format!("{w}x{h}");
        let size = ImageSize {
            width: w,
            height: h,
        };
        // Realistic bimodal grayscale distribution (foreground ~192, background ~64)
        let data: Vec<u8> = (0..(w * h))
            .map(|i| {
                if (i % 2) == 0 {
                    64u8.wrapping_add((i % 16) as u8)
                } else {
                    192u8.wrapping_sub((i % 16) as u8)
                }
            })
            .collect();

        let image = Image::<u8, 1>::new(size, data).unwrap();
        let dst = Image::<u8, 1>::from_size_val(size, 0).unwrap();

        group.throughput(Throughput::Elements((w * h) as u64));

        group.bench_with_input(
            BenchmarkId::new("compute_otsu_threshold", &parameter_string),
            &image,
            |b, img| {
                b.iter(|| std::hint::black_box(compute_otsu_threshold(std::hint::black_box(img))))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("threshold_otsu", &parameter_string),
            &(&image, &dst),
            |b, &(src, ref_dst)| {
                let mut out = ref_dst.clone();
                b.iter(|| {
                    std::hint::black_box(threshold_otsu(
                        std::hint::black_box(src),
                        std::hint::black_box(&mut out),
                        std::hint::black_box(255),
                    ))
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("threshold_binary", &parameter_string),
            &(&image, &dst),
            |b, &(src, ref_dst)| {
                let mut out = ref_dst.clone();
                b.iter(|| {
                    std::hint::black_box(threshold_binary(
                        std::hint::black_box(src),
                        std::hint::black_box(&mut out),
                        std::hint::black_box(128),
                        std::hint::black_box(255),
                    ))
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_threshold);
criterion_main!(benches);
