use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kornia_image::{Image, ImageSize};
use kornia_imgproc::parallel;
use kornia_tensor::CpuAllocator;

//old version using iter
#[inline]
fn in_range_old(
    src: &Image<u8, 3, CpuAllocator>,
    dst: &mut Image<u8, 1, CpuAllocator>,
    lower: &[u8; 3],
    upper: &[u8; 3],
) {
    parallel::par_iter_rows(src, dst, |src_pixel, dst_pixel| {
        let mut is_in_range = true;

        src_pixel
            .iter()
            .zip(lower.iter().zip(upper.iter()))
            .for_each(|(src_val, (l, u))| {
                is_in_range &= src_val >= l && src_val <= u;
            });

        dst_pixel[0] = if is_in_range { 255 } else { 0 };
    });
}

//new version- no iterator
#[inline]
fn in_range_new(
    src: &Image<u8, 3, CpuAllocator>,
    dst: &mut Image<u8, 1, CpuAllocator>,
    lower: &[u8; 3],
    upper: &[u8; 3],
) {
    parallel::par_iter_rows(src, dst, |src_pixel, dst_pixel| {
        for c in 0..3 {
            let v = src_pixel[c];
            if v < lower[c] || v > upper[c] {
                dst_pixel[0] = 0;
                return;
            }
        }
        dst_pixel[0] = 255;
    });
}

//Benchmark

fn bench_in_range(c: &mut Criterion) {
    let mut group = c.benchmark_group("InRange");

    for (width, height) in [(256, 224), (512, 448), (1024, 896)].iter() {
        group.throughput(criterion::Throughput::Elements((*width * *height) as u64));
        let label = format!("{width}x{height}");

        let img_data = vec![128u8; width * height * 3];
        let size = ImageSize {
            width: *width,
            height: *height,
        };

        let img = Image::<u8, 3, _>::new(size, img_data, CpuAllocator).unwrap();
        let dst = Image::<u8, 1, _>::from_size_val(size, 0, CpuAllocator).unwrap();

        let lower = [40, 40, 40];
        let upper = [200, 200, 200];

        // old version
        group.bench_with_input(
            BenchmarkId::new("old_iter_zip_parallel", &label),
            &(&img, &dst),
            |b, i| {
                let (src, mut out) = (i.0, i.1.clone());
                b.iter(|| std::hint::black_box(in_range_old(src, &mut out, &lower, &upper)))
            },
        );

        // new version
        group.bench_with_input(
            BenchmarkId::new("new_manual_loop_parallel", &label),
            &(&img, &dst),
            |b, i| {
                let (src, mut out) = (i.0, i.1.clone());
                b.iter(|| std::hint::black_box(in_range_new(src, &mut out, &lower, &upper)))
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_in_range);
criterion_main!(benches);
