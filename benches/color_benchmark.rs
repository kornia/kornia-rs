use criterion::{black_box, criterion_group, criterion_main, Criterion};

use kornia_rs::color::grayscale_from_rgb;
use kornia_rs::image::{Image, ImageSize};

fn grayscale_benchmark(c: &mut Criterion) {
    let image_sizes = vec![(100, 100), (500, 500), (1000, 1000)]; // Define different image sizes

    for (width, height) in image_sizes {
        let image = Image::new(ImageSize { width, height }, vec![0; width * height * 3]);
        c.bench_function(&format!("gray_{}_{}", width, height), |b| {
            b.iter(|| grayscale_from_rgb(black_box(image.clone())))
        });
    }
}

criterion_group!(benches, grayscale_benchmark);
criterion_main!(benches);
