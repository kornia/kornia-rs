use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use kornia::{
    image::{Image, ImageSize},
    imgproc::metrics,
};

fn bench_mse(c: &mut Criterion) {
    let mut group = c.benchmark_group("mse");
    let image_sizes = vec![(256, 224), (512, 448), (1024, 896)];

    for (width, height) in image_sizes {
        let image_size = ImageSize { width, height };
        let id = format!("{}x{}", width, height);
        let image = Image::<u8, 3>::new(image_size, vec![0u8; width * height * 3]).unwrap();
        let image_f32 = image.cast::<f32>();
        group.bench_with_input(BenchmarkId::new("mapv", &id), &image_f32, |b, i| {
            b.iter(|| metrics::mse(black_box(i), black_box(i)))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_mse);
criterion_main!(benches);
