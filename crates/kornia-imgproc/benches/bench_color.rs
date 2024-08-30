use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use kornia::imgproc::color::gray_from_rgb;
use kornia_image::{Image, ImageSize};

// vanilla version
fn gray_iter(image: &Image<f32, 3>) -> Image<u8, 1> {
    let data = vec![0u8; image.size().width * image.size().height];
    let gray_image = Image::new(image.size(), data).unwrap();
    for y in 0..image.height() {
        for x in 0..image.width() {
            let r = image.get_unchecked([y, x, 0]);
            let g = image.get_unchecked([y, x, 1]);
            let b = image.get_unchecked([y, x, 2]);
            let _gray_pixel = (76. * r + 150. * g + 29. * b) / 255.;
            // TODO: implement set_unchecked
        }
    }
    gray_image
}

fn gray_image_crate(image: &Image<u8, 3>) -> Image<u8, 1> {
    let image_data = image.as_slice();
    let rgb = image::RgbImage::from_raw(
        image.size().width as u32,
        image.size().height as u32,
        image_data.to_vec(),
    )
    .unwrap();
    let image_crate = image::DynamicImage::ImageRgb8(rgb);

    let image_gray = image_crate.grayscale();

    Image::new(image.size(), image_gray.into_bytes()).unwrap()
}

fn bench_grayscale(c: &mut Criterion) {
    let mut group = c.benchmark_group("Grayscale");
    let image_sizes = vec![(256, 224), (512, 448), (1024, 896)];

    for (width, height) in image_sizes {
        let id = format!("{}x{}", width, height);
        // input image
        let image_data = vec![0u8; width * height * 3];
        let image = Image::new(ImageSize { width, height }, image_data).unwrap();
        let image_f32 = image.clone().cast::<f32>().unwrap();
        // output image
        let mut gray = Image::from_size_val(image.size(), 0.0).unwrap();
        group.bench_with_input(BenchmarkId::new("zip", &id), &image_f32, |b, _i| {
            b.iter(|| gray_from_rgb(black_box(&image_f32), black_box(&mut gray)))
        });
        group.bench_with_input(BenchmarkId::new("iter", &id), &image_f32, |b, i| {
            b.iter(|| gray_iter(black_box(&i.clone())))
        });
        group.bench_with_input(BenchmarkId::new("image_crate", &id), &image, |b, i| {
            b.iter(|| gray_image_crate(black_box(&i.clone())))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_grayscale);
criterion_main!(benches);
