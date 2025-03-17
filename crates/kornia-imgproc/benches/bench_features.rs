use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::Rng;
use kornia_image::Image;
use kornia_imgproc::features::harris_response;


fn bench_features(c: &mut Criterion) {
    let mut group = c.benchmark_group("Features");
    let mut rng = rand::thread_rng();

    for (width, height) in [(1920, 1080)].iter() {
        group.throughput(criterion::Throughput::Elements((*width * *height) as u64));

        let parameter_string = format!("{}x{}", width, height);

        // input image
        let image_data: Vec<f32> = (0..(*width * *height))
            .map(|_| rng.gen_range(0.0..1.0))
            .collect();
        let image_size = [*width, *height].into();

        let image_f32: Image<f32, 1> = Image::new(image_size, image_data).unwrap();

        // output image
        let response_f32: Image<f32, 1> = Image::from_size_val(image_size, 0.0).unwrap();

        group.bench_with_input(
            BenchmarkId::new("harris", &parameter_string),
            &(&image_f32, &response_f32),
            |b, i| {
                let (src, mut dst) = (i.0, i.1.clone());
                b.iter(|| (harris_response(src, &mut dst, None, Default::default(), None)))
            },
        );
    }
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().warm_up_time(std::time::Duration::new(10, 0));
    targets = bench_features
);
criterion_main!(benches);
