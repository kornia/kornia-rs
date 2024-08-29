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
    let mut group = c.benchmark_group("resize");
    let image_sizes = vec![(256, 224), (512, 448), (1024, 896)];

    for (width, height) in image_sizes {
        let image_size = ImageSize { width, height };
        let id = format!("{}x{}", width, height);
        // input image
        let image = Image::<u8, 3>::new(image_size, vec![0u8; width * height * 3]).unwrap();
        let image_f32 = image.clone().cast::<f32>();
        // output image
        let new_size = ImageSize {
            width: width / 2,
            height: height / 2,
        };
        let mut out_f32 = Image::<f32, 3>::from_size_val(new_size, 0.0).unwrap();
        let mut out_u8 = Image::<u8, 3>::from_size_val(new_size, 0).unwrap();

        group.bench_with_input(BenchmarkId::new("native", &id), &image_f32, |b, i| {
            b.iter(|| {
                resize::resize_native(
                    black_box(i),
                    black_box(&mut out_f32),
                    InterpolationMode::Nearest,
                )
            })
        });
        group.bench_with_input(BenchmarkId::new("image_rs", &id), &image, |b, i| {
            b.iter(|| resize_image_crate(black_box(i.clone()), new_size))
        });
        group.bench_with_input(BenchmarkId::new("fast", &id), &image, |b, i| {
            b.iter(|| {
                resize::resize_fast(
                    black_box(i),
                    black_box(&mut out_u8),
                    InterpolationMode::Nearest,
                )
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_resize);
criterion_main!(benches);
