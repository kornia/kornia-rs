use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use kornia_image::Image;
use kornia_imgproc::{
    interpolation::InterpolationMode,
    warp::{get_rotation_matrix2d, warp_affine, warp_perspective},
};

fn bench_warp_affine(c: &mut Criterion) {
    let mut group = c.benchmark_group("WarpAffine");

    for (width, height) in [(256, 224), (512, 448), (1024, 896)].iter() {
        group.throughput(criterion::Throughput::Elements((*width * *height) as u64));

        let parameter_string = format!("{}x{}", width, height);

        // input image
        let image_size = [*width, *height].into();
        let image = Image::<u8, 3>::new(image_size, vec![0u8; width * height * 3]).unwrap();
        let image_f32 = image.clone().cast::<f32>().unwrap();

        // output image
        let output = Image::<f32, 3>::from_size_val(image_size, 0.0).unwrap();
        let m = get_rotation_matrix2d((*width as f32 / 2.0, *height as f32 / 2.0), 45.0, 1.0);

        group.bench_with_input(
            BenchmarkId::new("ndarray_zip_par", &parameter_string),
            &(&image_f32, &output, m),
            |b, i| {
                let (src, mut dst, m) = (i.0.clone(), i.1.clone(), i.2);
                b.iter(|| {
                    warp_affine(
                        black_box(&src),
                        black_box(&mut dst),
                        black_box(&m),
                        black_box(InterpolationMode::Bilinear),
                    )
                })
            },
        );
    }
    group.finish();
}

fn bench_warp_perspective(c: &mut Criterion) {
    let mut group = c.benchmark_group("WarpPerspective");

    for (width, height) in [(256, 224), (512, 448), (1024, 896)].iter() {
        group.throughput(criterion::Throughput::Elements((*width * *height) as u64));

        let parameter_string = format!("{}x{}", width, height);

        // input image
        let image_size = [*width, *height].into();
        let image = Image::<u8, 3>::new(image_size, vec![0u8; width * height * 3]).unwrap();
        let image_f32 = image.clone().cast::<f32>().unwrap();

        // output image
        let output = Image::<f32, 3>::from_size_val(image_size, 0.0).unwrap();
        let m = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

        group.bench_with_input(
            BenchmarkId::new("ndarray_zip_par", &parameter_string),
            &(&image_f32, &output, m),
            |b, i| {
                let (src, mut dst, m) = (i.0.clone(), i.1.clone(), i.2);
                b.iter(|| {
                    warp_perspective(
                        black_box(&src),
                        black_box(&mut dst),
                        black_box(&m),
                        black_box(InterpolationMode::Bilinear),
                    )
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_warp_affine, bench_warp_perspective);
criterion_main!(benches);
