use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use kornia_image::Image;
use kornia_imgproc::filter::gaussian_blur;

use image::RgbImage;
use imageproc::filter::gaussian_blur_f32;

fn bench_filters(c: &mut Criterion) {
    let mut group = c.benchmark_group("Gaussian Blur");

    for (width, height) in [(256, 224), (512, 448), (1024, 896)].iter() {
        for kernel_size in [3, 5, 7, 9, 11, 17].iter() {
            group.throughput(criterion::Throughput::Elements(
                (*width * *height * *kernel_size) as u64,
            ));

            let parameter_string = format!("{}x{}x{}", width, height, kernel_size);

            // input image
            let image_data = vec![0f32; width * height * 3];
            let image_size = [*width, *height].into();

            let image = Image::<_, 3>::new(image_size, image_data).unwrap();

            // output image
            let output = Image::from_size_val(image.size(), 0.0).unwrap();

            group.bench_with_input(
                BenchmarkId::new("gaussian_blur_native", &parameter_string),
                &(&image, &output),
                |b, i| {
                    let (src, mut dst) = (i.0, i.1.clone());
                    b.iter(|| {
                        black_box(gaussian_blur(
                            src,
                            &mut dst,
                            (*kernel_size, *kernel_size),
                            (1.5, 1.5),
                        ))
                    })
                },
            );

            group.bench_with_input(
                BenchmarkId::new("gaussian_blur_imageproc", &parameter_string),
                &image,
                |b, i| {
                    let rgb_image = RgbImage::new(i.cols() as u32, i.rows() as u32);
                    let sigma = (*kernel_size as f32) / 2.0;
                    b.iter(|| black_box(gaussian_blur_f32(&rgb_image, sigma)))
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, bench_filters);
criterion_main!(benches);
