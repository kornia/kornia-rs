use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kornia_image::{allocator::CpuAllocator, Image, ImageSize};
use kornia_imgproc::parallel::ExecutionStrategy;
use kornia_imgproc::threshold::threshold_binary;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn create_test_image(width: usize, height: usize) -> Image<u8, 1, CpuAllocator> {
    let mut rng = StdRng::seed_from_u64(42);
    let data: Vec<u8> = (0..(width * height)).map(|_| rng.random()).collect();
    let size = ImageSize { width, height };
    Image::new(size, data, CpuAllocator).unwrap()
}

fn bench_threshold(c: &mut Criterion) {
    let mut group = c.benchmark_group("Threshold");

    // We will test on Full HD only for clear results
    let (w, h) = (1920, 1080);
    let src = create_test_image(w, h);

    // 1. Benchmark Serial Execution
    group.bench_with_input(
        BenchmarkId::new("binary_serial", format!("{}x{}", w, h)),
        &src,
        |b, src| {
            // Allocate outside to measure only algorithm performance
            let mut dst = Image::from_size_val(src.size(), 0, CpuAllocator).unwrap();
            b.iter(|| {
                threshold_binary(src, &mut dst, 127, 255, ExecutionStrategy::Serial).unwrap();
            })
        },
    );

    // 2. Benchmark Parallel Elements
    group.bench_with_input(
        BenchmarkId::new("binary_parallel_elements", format!("{}x{}", w, h)),
        &src,
        |b, src| {
            let mut dst = Image::from_size_val(src.size(), 0, CpuAllocator).unwrap();
            b.iter(|| {
                threshold_binary(src, &mut dst, 127, 255, ExecutionStrategy::ParallelElements)
                    .unwrap();
            })
        },
    );

    // 3. Benchmark AutoRows (Parallel Rows)
    group.bench_with_input(
        BenchmarkId::new("binary_auto_rows", format!("{}x{}", w, h)),
        &src,
        |b, src| {
            let mut dst = Image::from_size_val(src.size(), 0, CpuAllocator).unwrap();
            b.iter(|| {
                // stride = width * channels (1)
                threshold_binary(
                    src,
                    &mut dst,
                    127,
                    255,
                    ExecutionStrategy::AutoRows(src.width()),
                )
                .unwrap();
            })
        },
    );

    // 4. Benchmark Fixed (Custom Pool)
    group.bench_with_input(
        BenchmarkId::new("binary_fixed_4", format!("{}x{}", w, h)),
        &src,
        |b, src| {
            let mut dst = Image::from_size_val(src.size(), 0, CpuAllocator).unwrap();
            b.iter(|| {
                threshold_binary(src, &mut dst, 127, 255, ExecutionStrategy::Fixed(4)).unwrap();
            })
        },
    );

    group.finish();
}

criterion_group!(benches, bench_threshold);
criterion_main!(benches);
