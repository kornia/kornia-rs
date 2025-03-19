use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use kornia_image::Image;
use kornia_imgproc::pyramid::pyrup;

fn bench_pyramid(c: &mut Criterion) {
    let mut group = c.benchmark_group("Pyramid Operations");

    for (width, height) in [(256, 224), (512, 448), (1024, 896)].iter() {
        group.throughput(criterion::Throughput::Elements((*width * *height) as u64));

        let parameter_string = format!("{}x{}", width, height);

        let small_image_size = [*width / 2, *height / 2].into();
        let small_image_data = (0..((*width / 2) * (*height / 2)))
            .map(|x| x as f32)
            .collect();
        let small_image = Image::<f32, 1>::new(small_image_size, small_image_data).unwrap();

        let image_size = [*width, *height].into();

        let up_image = Image::<f32, 1>::from_size_val(image_size, 0.0).unwrap();

        group.bench_with_input(
            BenchmarkId::new("pyrup", &parameter_string),
            &(&small_image, &up_image),
            |b, i| {
                let (src, mut dst) = (i.0, i.1.clone());
                b.iter(|| {
                    black_box(pyrup(src, &mut dst)).unwrap();
                })
            },
        );

        // For multi-channel images
        let small_image_data_3c = (0..((*width / 2) * (*height / 2) * 3))
            .map(|x| x as f32)
            .collect();
        let small_image_3c = Image::<f32, 3>::new(small_image_size, small_image_data_3c).unwrap();
        let up_image_3c = Image::<f32, 3>::from_size_val(image_size, 0.0).unwrap();

        group.bench_with_input(
            BenchmarkId::new("pyrup_3c", &parameter_string),
            &(&small_image_3c, &up_image_3c),
            |b, i| {
                let (src, mut dst) = (i.0, i.1.clone());
                b.iter(|| {
                    black_box(pyrup(src, &mut dst)).unwrap();
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_pyramid);
criterion_main!(benches);
