use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use kornia_image::Image;
use kornia_imgproc::metrics;

fn bench_mse(c: &mut Criterion) {
    let mut group = c.benchmark_group("mse");

    for (width, height) in [(256, 224), (512, 448), (1024, 896)].iter() {
        group.throughput(criterion::Throughput::Elements((*width * *height) as u64));

        let parameter_string = format!("{}x{}", width, height);

        // input image
        let image_size = [*width, *height].into();
        let image = Image::<u8, 3>::new(image_size, vec![0u8; width * height * 3]).unwrap();
        let image_f32 = image.cast::<f32>().unwrap();

        group.bench_with_input(
            BenchmarkId::new("mse_map", &parameter_string),
            &image_f32,
            |b, i| b.iter(|| metrics::mse(black_box(i), black_box(i))),
        );
    }
    group.finish();
}

criterion_group!(benches, bench_mse);
criterion_main!(benches);
