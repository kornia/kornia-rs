use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use kornia::image::{Image, ImageSize};
use kornia::imgproc::{interpolation::InterpolationMode, resize};

fn resize_image_crate(image: Image<u8, 3>, new_size: ImageSize) -> Image<u8, 3> {
    let image_data = image.as_slice();
    let rgb = image::RgbImage::from_raw(
        image.size().width as u32,
        image.size().height as u32,
        image_data.to_vec(),
    )
    .unwrap();
    let image_crate = image::DynamicImage::ImageRgb8(rgb);

    let image_resized = image_crate.resize_exact(
        new_size.width as u32,
        new_size.height as u32,
        image::imageops::FilterType::Nearest,
    );
    let data = image_resized.into_rgb8().into_raw();
    Image::new(new_size, data).unwrap()
}

fn bench_resize(c: &mut Criterion) {
    let mut group = c.benchmark_group("Resize");

    for (width, height) in [(256, 224), (512, 448), (1024, 896)].iter() {
        group.throughput(criterion::Throughput::Elements((*width * *height) as u64));

        let parameter_string = format!("{}x{}", width, height);

        // input image
        let image_size = [*width, *height].into();
        let image = Image::<u8, 3>::new(image_size, vec![0u8; width * height * 3]).unwrap();
        let image_f32 = image.clone().cast::<f32>().unwrap();

        // output image
        let new_size = ImageSize {
            width: width / 2,
            height: height / 2,
        };

        let out_f32 = Image::<f32, 3>::from_size_val(new_size, 0.0).unwrap();
        let out_u8 = Image::<u8, 3>::from_size_val(new_size, 0).unwrap();

        group.bench_with_input(
            BenchmarkId::new("image_rs", &parameter_string),
            &image,
            |b, i| b.iter(|| resize_image_crate(black_box(i.clone()), new_size)),
        );

        group.bench_with_input(
            BenchmarkId::new("kornia_par", &parameter_string),
            &(&image_f32, &out_f32),
            |b, i| {
                let (src, mut dst) = (i.0, i.1.clone());
                b.iter(|| {
                    black_box(resize::resize_native(
                        src,
                        &mut dst,
                        InterpolationMode::Nearest,
                    ))
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("fast_resize_lib", &parameter_string),
            &(image, out_u8),
            |b, i| {
                let (src, mut dst) = (i.0.clone(), i.1.clone());
                b.iter(|| {
                    black_box(resize::resize_fast(
                        &src,
                        &mut dst,
                        InterpolationMode::Nearest,
                    ))
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_resize);
criterion_main!(benches);
