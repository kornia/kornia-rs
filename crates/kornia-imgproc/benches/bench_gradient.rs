use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use kornia_image::Image;
use kornia_imgproc::filter::{
    scharr_spatial_gradient_float, scharr_spatial_gradient_separable_f32, spatial_gradient_float,
    spatial_gradient_float_parallel, spatial_gradient_float_parallel_row,
};

fn bench_gradient(c: &mut Criterion) {
    let mut group = c.benchmark_group("Spatial Gradient Float");

    for (width, height) in [(752, 480), (512, 512), (1024, 1024), (2048, 2048)].iter() {
        group.throughput(criterion::Throughput::Elements((*width * *height) as u64));

        let parameter_string = format!("{width}x{height}");

        // input image
        let image_data = vec![0f32; width * height * 3];
        let image_size = [*width, *height].into();

        let image = Image::<_, 3>::new(image_size, image_data).unwrap();

        // output image
        let output_dx = Image::from_size_val(image.size(), 0.0).unwrap();
        let output_dy = Image::from_size_val(image.size(), 0.0).unwrap();

        group.bench_with_input(
            BenchmarkId::new("spatial_gradient_float", &parameter_string),
            &(&image, &output_dx, &output_dy),
            |b, i| {
                let (src, mut dx, mut dy) = (i.0, i.1.clone(), i.2.clone());
                b.iter(|| std::hint::black_box(spatial_gradient_float(src, &mut dx, &mut dy)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("spatial_gradient_float_parallel_row", &parameter_string),
            &(&image, &output_dx, &output_dy),
            |b, i| {
                let (src, mut dx, mut dy) = (i.0, i.1.clone(), i.2.clone());
                b.iter(|| {
                    std::hint::black_box(spatial_gradient_float_parallel_row(src, &mut dx, &mut dy))
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("spatial_gradient_float_parallel", &parameter_string),
            &(&image, &output_dx, &output_dy),
            |b, i| {
                let (src, mut dx, mut dy) = (i.0, i.1.clone(), i.2.clone());
                b.iter(|| {
                    std::hint::black_box(spatial_gradient_float_parallel(src, &mut dx, &mut dy))
                })
            },
        );

        // single-channel Scharr, as optical flow uses it
        let gray = Image::<f32, 1>::new(
            image_size,
            (0..width * height).map(|i| (i % 251) as f32).collect(),
        )
        .unwrap();
        let gray_dx = Image::<f32, 1>::from_size_val(image_size, 0.0).unwrap();
        let gray_dy = gray_dx.clone();
        group.bench_with_input(
            BenchmarkId::new("scharr_spatial_gradient_float_1c", &parameter_string),
            &(&gray, &gray_dx, &gray_dy),
            |b, i| {
                let (src, mut dx, mut dy) = (i.0, i.1.clone(), i.2.clone());
                b.iter(|| {
                    std::hint::black_box(scharr_spatial_gradient_float(src, &mut dx, &mut dy))
                })
            },
        );
    }

    group.finish();
}

fn bench_scharr(c: &mut Criterion) {
    let mut group = c.benchmark_group("Scharr Gradient f32 1c");
    for (width, height) in [(752, 480), (1024, 1024)] {
        group.throughput(criterion::Throughput::Elements((width * height) as u64));
        let parameter_string = format!("{width}x{height}");
        let data = (0..width * height)
            .map(|i| ((i * 7919) % 251) as f32)
            .collect();
        let image = Image::<f32, 1>::new([width, height].into(), data).unwrap();
        let output = Image::<f32, 1>::from_size_val(image.size(), 0.0).unwrap();
        group.bench_with_input(
            BenchmarkId::new("scharr_spatial_gradient_float", &parameter_string),
            &(&image, &output),
            |b, i| {
                let (src, mut dx, mut dy) = (i.0, i.1.clone(), i.1.clone());
                b.iter(|| {
                    std::hint::black_box(scharr_spatial_gradient_float(src, &mut dx, &mut dy))
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("scharr_spatial_gradient_separable_f32", &parameter_string),
            &(&image, &output),
            |b, i| {
                let (src, mut dx, mut dy) = (i.0, i.1.clone(), i.1.clone());
                b.iter(|| {
                    std::hint::black_box(scharr_spatial_gradient_separable_f32(
                        src, &mut dx, &mut dy,
                    ))
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_gradient, bench_scharr);
criterion_main!(benches);
