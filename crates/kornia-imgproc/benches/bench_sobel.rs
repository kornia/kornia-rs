//! Performance benchmarks for Sobel edge detection
//!
//! Measures:
//! - Single-frame latency at various resolutions (1080p, 4K, etc.)
//! - Different kernel sizes (1, 3, 5, 7)
//! - Throughput (FPS) for real-time processing

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use kornia_image::{Image, ImageSize};
use kornia_imgproc::filter::sobel;
use kornia_tensor::CpuAllocator;

/// Helper: Create synthetic test image at given resolution
fn create_test_image(width: usize, height: usize) -> Image<f32, 1, CpuAllocator> {
    let size = ImageSize { width, height };
    let mut data = vec![0.0f32; width * height];
    
    // Create a pattern: vertical line + horizontal line (cross)
    for y in 0..height {
        data[y * width + width / 2] = 0.5;
    }
    for x in 0..width {
        data[(height / 2) * width + x] = 0.5;
    }
    
    Image::new(size, data, CpuAllocator).unwrap()
}

fn bench_sobel_1080p(c: &mut Criterion) {
    let mut group = c.benchmark_group("Sobel Edge Detection - 1080p");
    group.throughput(Throughput::Elements((1920 * 1080) as u64));
    
    let img = create_test_image(1920, 1080);
    
    for kernel_size in &[1, 3, 5, 7] {
        group.bench_with_input(
            BenchmarkId::new("1080p_kernel_", kernel_size),
            kernel_size,
            |b, &k_size| {
                let mut dst = Image::from_size_val(img.size(), 0.0, CpuAllocator).unwrap();
                b.iter(|| sobel(&img, &mut dst, k_size))
            },
        );
    }
    
    group.finish();
}

fn bench_sobel_720p(c: &mut Criterion) {
    let mut group = c.benchmark_group("Sobel Edge Detection - 720p");
    group.throughput(Throughput::Elements((1280 * 720) as u64));
    
    let img = create_test_image(1280, 720);
    
    for kernel_size in &[1, 3, 5, 7] {
        group.bench_with_input(
            BenchmarkId::new("720p_kernel_", kernel_size),
            kernel_size,
            |b, &k_size| {
                let mut dst = Image::from_size_val(img.size(), 0.0, CpuAllocator).unwrap();
                b.iter(|| sobel(&img, &mut dst, k_size))
            },
        );
    }
    
    group.finish();
}

fn bench_sobel_4k(c: &mut Criterion) {
    let mut group = c.benchmark_group("Sobel Edge Detection - 4K");
    group.throughput(Throughput::Elements((3840 * 2160) as u64));
    
    let img = create_test_image(3840, 2160);
    
    for kernel_size in &[1, 3, 5, 7] {
        group.bench_with_input(
            BenchmarkId::new("4k_kernel_", kernel_size),
            kernel_size,
            |b, &k_size| {
                let mut dst = Image::from_size_val(img.size(), 0.0, CpuAllocator).unwrap();
                b.iter(|| sobel(&img, &mut dst, k_size))
            },
        );
    }
    
    group.finish();
}

fn bench_sobel_small_image(c: &mut Criterion) {
    let mut group = c.benchmark_group("Sobel Edge Detection - Small");
    group.throughput(Throughput::Elements((640 * 480) as u64));
    
    let img = create_test_image(640, 480);
    
    for kernel_size in &[1, 3, 5, 7] {
        group.bench_with_input(
            BenchmarkId::new("small_kernel_", kernel_size),
            kernel_size,
            |b, &k_size| {
                let mut dst = Image::from_size_val(img.size(), 0.0, CpuAllocator).unwrap();
                b.iter(|| sobel(&img, &mut dst, k_size))
            },
        );
    }
    
    group.finish();
}

fn bench_sobel_kernel_size_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Sobel Kernel Size Comparison");
    group.throughput(Throughput::Elements((1920 * 1080) as u64));
    
    let img = create_test_image(1920, 1080);
    
    group.bench_function("kernel_1", |b| {
        let mut dst = Image::from_size_val(img.size(), 0.0, CpuAllocator).unwrap();
        b.iter(|| sobel(&img, &mut dst, 1))
    });
    
    group.bench_function("kernel_3", |b| {
        let mut dst = Image::from_size_val(img.size(), 0.0, CpuAllocator).unwrap();
        b.iter(|| sobel(&img, &mut dst, 3))
    });
    
    group.bench_function("kernel_5", |b| {
        let mut dst = Image::from_size_val(img.size(), 0.0, CpuAllocator).unwrap();
        b.iter(|| sobel(&img, &mut dst, 5))
    });
    
    group.bench_function("kernel_7", |b| {
        let mut dst = Image::from_size_val(img.size(), 0.0, CpuAllocator).unwrap();
        b.iter(|| sobel(&img, &mut dst, 7))
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_sobel_small_image,
    bench_sobel_720p,
    bench_sobel_1080p,
    bench_sobel_4k,
    bench_sobel_kernel_size_comparison,
);
criterion_main!(benches);
