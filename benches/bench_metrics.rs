use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use kornia_rs::image::{Image, ImageSize};

// TODO: figure how to auto select a function backend based on the input size
fn mse_zip(image1: &Image<f32, 3>, image2: &Image<f32, 3>) -> f32 {
    ndarray::Zip::from(&image1.data)
        .and(&image2.data)
        .fold(0f32, |acc, &a, &b| acc + (a - b).powi(2))
        / (image1.data.len() as f32)
}

fn bench_mse(c: &mut Criterion) {
    let mut group = c.benchmark_group("mse");
    let image_sizes = vec![(256, 224), (512, 448), (1024, 896)];

    for (width, height) in image_sizes {
        let image_size = ImageSize { width, height };
        let id = format!("{}x{}", width, height);
        let image = Image::<u8, 3>::new(image_size, vec![0u8; width * height * 3]).unwrap();
        let image_f32 = image.cast::<f32>().unwrap();
        group.bench_with_input(BenchmarkId::new("mapv", &id), &image_f32, |b, i| {
            b.iter(|| kornia_rs::metrics::mse(black_box(i), black_box(i)))
        });
        group.bench_with_input(BenchmarkId::new("zip", &id), &image_f32, |b, i| {
            b.iter(|| mse_zip(black_box(i), black_box(i)))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_mse);
criterion_main!(benches);
