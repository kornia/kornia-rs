use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use image::imageops::blur;
use image::{GenericImageView, ImageBuffer, Pixel};
use kornia_image::Image;
use kornia_imgproc::features::{
    dog_response_rayon, dog_response_row_parallel, dog_response_serial,
};

fn extern_dog_response_serial<I: GenericImageView>(
    src: &I,
    dst: &mut ImageBuffer<I::Pixel, Vec<<I::Pixel as Pixel>::Subpixel>>,
    sigma1: f32,
    sigma2: f32,
) where
    I::Pixel: 'static,
{
    if src.dimensions() != dst.dimensions() {
        panic!("src and dst must have the same dimensions");
    }

    let gauss1 = blur(src, sigma1);
    let gauss2 = blur(src, sigma2);

    // Get raw data from both Gaussian blurred images
    let gauss1_data = gauss1.into_raw();
    let gauss2_data = gauss2.into_raw();

    // Get mutable access to destination buffer
    let dst_data = dst.as_mut();
    // Directly write the difference to the destination buffer
    dst_data
        .iter_mut()
        .zip(gauss1_data.iter().zip(gauss2_data.iter()))
        .for_each(|(dst, (&a, &b))| {
            *dst = a - b;
        });
}

fn bench_dog_response(c: &mut Criterion) {
    let mut group = c.benchmark_group("Dog Response");
    group.sample_size(30);
    let test_sizes = [(32, 32), (512, 512), (8192, 8192)];

    for (width, height) in test_sizes.iter() {
        group.throughput(criterion::Throughput::Elements((*width * *height) as u64));

        let src = Image::<f32, 1>::from_size_val([*width, *height].into(), 1.0).unwrap();
        let mut dst = Image::<f32, 1>::from_size_val([*width, *height].into(), 0.0).unwrap();

        // Benchmark DoG response (parallel version)
        group.bench_with_input(
            BenchmarkId::new("dog_response_row_parallel", format!("{}x{}", width, height)),
            &(width, height),
            |b, _| {
                b.iter(|| {
                    dog_response_row_parallel(
                        black_box(&src),
                        black_box(&mut dst),
                        black_box(0.5),
                        black_box(1.0),
                    )
                    .unwrap()
                })
            },
        );

        // Benchmark DoG response (serial version)
        group.bench_with_input(
            BenchmarkId::new("dog_response_serial", format!("{}x{}", width, height)),
            &(width, height),
            |b, _| {
                b.iter(|| {
                    dog_response_serial(
                        black_box(&src),
                        black_box(&mut dst),
                        black_box(0.5),
                        black_box(1.0),
                    )
                    .unwrap()
                })
            },
        );

        // Benchmark DoG response (rayon version)
        group.bench_with_input(
            BenchmarkId::new("dog_response_rayon", format!("{}x{}", width, height)),
            &(width, height),
            |b, _| {
                b.iter(|| {
                    dog_response_rayon(
                        black_box(&src),
                        black_box(&mut dst),
                        black_box(0.5),
                        black_box(1.0),
                    )
                    .unwrap()
                })
            },
        );

        // Benchmark external DoG response implementation
        let img_width = *width as u32;
        let img_height = *height as u32;
        let img_src = image::ImageBuffer::from_pixel(img_width, img_height, image::Luma([1u8]));
        let mut img_dst = image::ImageBuffer::from_pixel(img_width, img_height, image::Luma([0u8]));

        group.bench_with_input(
            BenchmarkId::new("extern_dog_response", format!("{}x{}", width, height)),
            &(width, height),
            |b, _| {
                b.iter(|| {
                    extern_dog_response_serial(
                        black_box(&img_src),
                        black_box(&mut img_dst),
                        black_box(0.5),
                        black_box(1.0),
                    )
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_dog_response);
criterion_main!(benches);

#[cfg(test)]
mod tests {

    #[test]
    fn test_extern_dog_response() {
        let img_width = 1024;
        let img_height = 1024;
        let img_src = image::ImageBuffer::from_pixel(img_width, img_height, image::Luma([1u8]));
        let mut img_dst = image::ImageBuffer::from_pixel(img_width, img_height, image::Luma([0u8]));

        let result = extern_dog_response_serial(&img_src, &mut img_dst, 0.5, 1.0);
        assert_eq!(result.dimensions(), (img_width, img_height));
    }
}
