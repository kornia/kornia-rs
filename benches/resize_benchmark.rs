use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use kornia_rs::image::{Image, ImageSize};
use kornia_rs::resize as F;
use kornia_rs::resize::{InterpolationMode, ResizeOptions};

fn resize_image_crate(image: Image, new_size: ImageSize) -> Image {
    let image_data = image.data.as_slice().unwrap();
    let rgb = image::RgbImage::from_raw(
        image.image_size().width as u32,
        image.image_size().height as u32,
        image_data.to_vec(),
    )
    .unwrap();
    let image_crate = image::DynamicImage::ImageRgb8(rgb);

    let image_resized = image_crate.resize_exact(
        new_size.width as u32,
        new_size.height as u32,
        image::imageops::FilterType::Gaussian,
    );
    let data = image_resized.into_rgb8().into_raw();
    Image::from_shape_vec([new_size.height as usize, new_size.width as usize, 3], data)
}

fn bench_resize(c: &mut Criterion) {
    let mut group = c.benchmark_group("Resize");
    let image_sizes = vec![(256, 224), (512, 448), (1024, 896)];

    for (width, height) in image_sizes {
        let image_size = ImageSize { width, height };
        let id = format!("{}x{}", width, height);
        let image = Image::new(image_size.clone(), vec![0; width * height * 3]);
        let new_size = ImageSize {
            width: width / 2,
            height: height / 2,
        };
        group.bench_with_input(BenchmarkId::new("zip", &id), &image, |b, i| {
            b.iter(|| {
                F::resize(
                    black_box(i),
                    new_size.clone(),
                    ResizeOptions {
                        interpolation: InterpolationMode::Bilinear,
                    },
                )
            })
        });
        group.bench_with_input(BenchmarkId::new("image_crate", &id), &image, |b, i| {
            b.iter(|| resize_image_crate(black_box(i.clone()), new_size.clone()))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_resize);
criterion_main!(benches);
