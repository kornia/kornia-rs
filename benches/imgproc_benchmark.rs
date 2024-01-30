use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use kornia_rs::image::{Image, ImageSize};
use kornia_rs::resize as F;
use kornia_rs::resize::ResizeOptions;

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
        group.bench_with_input(BenchmarkId::new("iter", &id), &image, |b, i| {
            b.iter(|| {
                F::resize(
                    black_box(i.clone()),
                    new_size.clone(),
                    ResizeOptions::default(),
                )
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_resize);
criterion_main!(benches);
