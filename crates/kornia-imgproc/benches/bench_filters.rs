use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use kornia_image::{Image, ImageError};
use kornia_imgproc::filter::{box_blur_fast, gaussian_blur, kernels, separable_filter};

use image::RgbImage;
use imageproc::filter::gaussian_blur_f32;

fn gaussian_blur_u8<const C: usize>(
    src: &Image<u8, C>,
    dst: &mut Image<u8, C>,
    kernel_size: usize,
    sigma: f32,
) -> Result<(), ImageError> {
    let kernel_x = kernels::gaussian_kernel_1d(kernel_size, sigma);
    let kernel_y = kernels::gaussian_kernel_1d(kernel_size, sigma);
    separable_filter(src, dst, &kernel_x, &kernel_y)
}

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

            let image_f32 = Image::<_, 3>::new(image_size, image_data).unwrap();
            let image_u8 = image_f32.cast::<u8>().unwrap();

            // output image
            let output_f32 = Image::<_, 3>::from_size_val(image_size, 0.0).unwrap();
            let output_u8 = output_f32.cast::<u8>().unwrap();

            group.bench_with_input(
                BenchmarkId::new("gaussian_blur_native_f32", &parameter_string),
                &(&image_f32, &output_f32),
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
                BenchmarkId::new("gaussian_blur_native_u8", &parameter_string),
                &(&image_u8, &output_u8),
                |b, i| {
                    let (src, mut dst) = (i.0, i.1.clone());
                    b.iter(|| black_box(gaussian_blur_u8(src, &mut dst, *kernel_size, 1.5)))
                },
            );

            group.bench_with_input(
                BenchmarkId::new("gaussian_blur_imageproc", &parameter_string),
                &image_f32,
                |b, i| {
                    let rgb_image = RgbImage::new(i.cols() as u32, i.rows() as u32);
                    let sigma = (*kernel_size as f32) / 2.0;
                    b.iter(|| black_box(gaussian_blur_f32(&rgb_image, sigma)))
                },
            );

            group.bench_with_input(
                BenchmarkId::new("box_blur_fast", &parameter_string),
                &(&image_f32, &output_f32),
                |b, i| {
                    let (src, mut dst) = (i.0, i.1.clone());
                    b.iter(|| black_box(box_blur_fast(src, &mut dst, (1.5, 1.5))))
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, bench_filters);
criterion_main!(benches);
