use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use kornia_image::Image;
use kornia_imgproc::pyramid::{pyrup, pyrdown};
use kornia_tensor::CpuAllocator;

fn bench_pyramid(c: &mut Criterion) {
    let mut group = c.benchmark_group("Pyramid Operations");

    for (width, height) in [(256, 224), (512, 448), (1024, 896)].iter() {
        group.throughput(criterion::Throughput::Elements((*width * *height) as u64));

        let parameter_string = format!("{width}x{height}");

        let small_image_size = [*width / 2, *height / 2].into();
        let small_image_data = (0..((*width / 2) * (*height / 2)))
            .map(|x| x as f32)
            .collect();
        let small_image =
            Image::<f32, 1, _>::new(small_image_size, small_image_data, CpuAllocator).unwrap();

        let image_size = [*width, *height].into();

        let up_image = Image::<f32, 1, _>::from_size_val(image_size, 0.0, CpuAllocator).unwrap();

        group.bench_with_input(
            BenchmarkId::new("pyrup", &parameter_string),
            &(&small_image, &up_image),
            |b, i| {
                let (src, mut dst) = (i.0, i.1.clone());
                b.iter(|| {
                    std::hint::black_box(pyrup(src, &mut dst)).unwrap();
                })
            },
        );

        // For multi-channel images
        let small_image_data_3c = (0..((*width / 2) * (*height / 2) * 3))
            .map(|x| x as f32)
            .collect();
        let small_image_3c =
            Image::<f32, 3, _>::new(small_image_size, small_image_data_3c, CpuAllocator).unwrap();
        let up_image_3c = Image::<f32, 3, _>::from_size_val(image_size, 0.0, CpuAllocator).unwrap();

        group.bench_with_input(
            BenchmarkId::new("pyrup_3c", &parameter_string),
            &(&small_image_3c, &up_image_3c),
            |b, i| {
                let (src, mut dst) = (i.0, i.1.clone());
                b.iter(|| {
                    std::hint::black_box(pyrup(src, &mut dst)).unwrap();
                })
            },
        );

        // Benchmark pyrdown (downsampling)
        let large_image_size = [*width, *height].into();
        let large_image_data = (0..((*width) * (*height)))
            .map(|x| x as f32)
            .collect();
        let large_image =
            Image::<f32, 1, _>::new(large_image_size, large_image_data, CpuAllocator).unwrap();

        let down_image_size = [(*width + 1) / 2, (*height + 1) / 2].into();
        let down_image =
            Image::<f32, 1, _>::from_size_val(down_image_size, 0.0, CpuAllocator).unwrap();

        group.bench_with_input(
            BenchmarkId::new("pyrdown", &parameter_string),
            &(&large_image, &down_image),
            |b, i| {
                let (src, mut dst) = (i.0, i.1.clone());
                b.iter(|| {
                    black_box(pyrdown(src, &mut dst)).unwrap();
                })
            },
        );

        // For multi-channel images
        let large_image_data_3c = (0..((*width) * (*height) * 3))
            .map(|x| x as f32)
            .collect();
        let large_image_3c =
            Image::<f32, 3, _>::new(large_image_size, large_image_data_3c, CpuAllocator).unwrap();
        let down_image_3c =
            Image::<f32, 3, _>::from_size_val(down_image_size, 0.0, CpuAllocator).unwrap();

        group.bench_with_input(
            BenchmarkId::new("pyrdown_3c", &parameter_string),
            &(&large_image_3c, &down_image_3c),
            |b, i| {
                let (src, mut dst) = (i.0, i.1.clone());
                b.iter(|| {
                    black_box(pyrdown(src, &mut dst)).unwrap();
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_pyramid);
criterion_main!(benches);
