use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use kornia_image::Image;
use kornia_imgproc::filter::{
    spatial_gradient_float, spatial_gradient_float_by_separable_filter,
    spatial_gradient_float_parallel, spatial_gradient_float_rayon_parallel,
    spatial_gradient_float_rayon_row_parallel, spatial_gradient_float_row_parallel,
};

fn bench_gradient(c: &mut Criterion) {
    let mut group = c.benchmark_group("Spatial Gradient");

    for (width, height) in [(512, 512), (1024, 1024), (2048, 2048)].iter() {
        group.throughput(criterion::Throughput::Elements((*width * *height) as u64));

        let parameter_string = format!("{}x{}", width, height);

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
                b.iter(|| black_box(spatial_gradient_float(src, &mut dx, &mut dy)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new(
                "spatial_gradient_float_by_separable_filter",
                &parameter_string,
            ),
            &(&image, &output_dx, &output_dy),
            |b, i| {
                let (src, mut dx, mut dy) = (i.0, i.1.clone(), i.2.clone());
                b.iter(|| {
                    black_box(spatial_gradient_float_by_separable_filter(
                        src, &mut dx, &mut dy,
                    ))
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("spatial_gradient_float_parallel", &parameter_string),
            &(&image, &output_dx, &output_dy),
            |b, i| {
                let (src, mut dx, mut dy) = (i.0, i.1.clone(), i.2.clone());
                b.iter(|| black_box(spatial_gradient_float_parallel(src, &mut dx, &mut dy)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("spatial_gradient_float_rayon_parallel", &parameter_string),
            &(&image, &output_dx, &output_dy),
            |b, i| {
                let (src, mut dx, mut dy) = (i.0, i.1.clone(), i.2.clone());
                b.iter(|| black_box(spatial_gradient_float_rayon_parallel(src, &mut dx, &mut dy)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new(
                "spatial_gradient_float_rayon_row_parallel",
                &parameter_string,
            ),
            &(&image, &output_dx, &output_dy),
            |b, i| {
                let (src, mut dx, mut dy) = (i.0, i.1.clone(), i.2.clone());
                b.iter(|| {
                    black_box(spatial_gradient_float_rayon_row_parallel(
                        src, &mut dx, &mut dy,
                    ))
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("spatial_gradient_float_row_parallel", &parameter_string),
            &(&image, &output_dx, &output_dy),
            |b, i| {
                let (src, mut dx, mut dy) = (i.0, i.1.clone(), i.2.clone());
                b.iter(|| black_box(spatial_gradient_float_row_parallel(src, &mut dx, &mut dy)))
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_gradient);
criterion_main!(benches);
