use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use kornia_rs::image::{Image, ImageSize};
use kornia_rs::interpolation::InterpolationMode;
use kornia_rs::warp::{get_rotation_matrix2d, warp_affine};

fn bench_warp_affine(c: &mut Criterion) {
    let mut group = c.benchmark_group("warp_affine");
    let image_sizes = vec![(256, 224), (512, 448), (1024, 896)];

    for (width, height) in image_sizes {
        let image_size = ImageSize { width, height };
        let id = format!("{}x{}", width, height);
        let image = Image::<u8, 3>::new(image_size, vec![0u8; width * height * 3]).unwrap();
        let image_f32 = image.clone().cast::<f32>().unwrap();
        let m = get_rotation_matrix2d((width as f32 / 2.0, height as f32 / 2.0), 45.0, 1.0);
        group.bench_with_input(BenchmarkId::new("native", &id), &image_f32, |b, i| {
            b.iter(|| warp_affine(black_box(i), m, image_size, InterpolationMode::Bilinear))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_warp_affine);
criterion_main!(benches);
