use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kornia_image::Image;
use kornia_imgproc::simd;
use kornia_imgproc::threshold::threshold_binary as scalar_threshold_binary;
use kornia_tensor::CpuAllocator;

fn bench_threshold(c: &mut Criterion) {
    let mut group = c.benchmark_group("ThresholdU8");
    group.warm_up_time(std::time::Duration::from_millis(300));
    group.measurement_time(std::time::Duration::from_secs(1));
    group.sample_size(25);

    for (w, h) in [(256, 224), (512, 448), (1024, 896)].iter() {
        group.throughput(criterion::Throughput::Elements((*w * *h) as u64));
        let ps = format!("{w}x{h}");
        let size = [*w, *h].into();
        let data: Vec<u8> = (0..w * h).map(|i| (i * 37 % 256) as u8).collect();
        let src = Image::<u8, 1, _>::new(size, data, CpuAllocator).unwrap();
        let dst = Image::<u8, 1, _>::from_size_val(size, 0, CpuAllocator).unwrap();

        group.bench_with_input(BenchmarkId::new("scalar", &ps), &(&src, &dst), |b, i| {
            let (s, mut d) = (i.0, i.1.clone());
            b.iter(|| {
                scalar_threshold_binary(s, &mut d, 100u8, 255u8).unwrap();
                std::hint::black_box(&mut d);
            })
        });

        group.bench_with_input(BenchmarkId::new("simd", &ps), &(&src, &dst), |b, i| {
            let (s, mut d) = (i.0, i.1.clone());
            b.iter(|| {
                simd::threshold_binary(s, &mut d, 100u8, 255u8).unwrap();
                std::hint::black_box(&mut d);
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_threshold);
criterion_main!(benches);
