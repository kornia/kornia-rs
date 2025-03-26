use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use kornia_image::Image;
use kornia_imgproc::features::{dog_response, HarrisResponse};
use rand::Rng;

fn bench_harris_response(c: &mut Criterion) {
    let mut group = c.benchmark_group("Features");
    let mut rng = rand::thread_rng();

    for (width, height) in [(224, 224), (1920, 1080)].iter() {
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
        let mut harris_response = HarrisResponse::new(image_size);

        group.bench_with_input(
            BenchmarkId::new("harris", &parameter_string),
            &(&image_f32, &response_f32),
            |b, i| {
                let (src, mut dst) = (i.0, i.1.clone());
                b.iter(|| black_box(harris_response.compute(src, &mut dst)))
            },
        );
    }
    group.finish();
}

fn bench_dog_response(c: &mut Criterion) {
    let mut group = c.benchmark_group("Features");
    group.sample_size(30);
    let test_sizes = [(32, 32), (512, 512), (8192, 8192)];

    for (width, height) in test_sizes.iter() {
        group.throughput(criterion::Throughput::Elements((*width * *height) as u64));

        let src = Image::<f32, 1>::from_size_val([*width, *height].into(), 1.0).unwrap();
        let mut dst = Image::<f32, 1>::from_size_val([*width, *height].into(), 0.0).unwrap();

        // Benchmark DoG response (serial version)
        group.bench_with_input(
            BenchmarkId::new("dog_response", format!("{}x{}", width, height)),
            &(width, height),
            |b, _| {
                b.iter(|| {
                    dog_response(
                        black_box(&src),
                        black_box(&mut dst),
                        black_box(0.5),
                        black_box(1.0),
                    )
                    .unwrap()
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().warm_up_time(std::time::Duration::new(10, 0));
    targets = bench_harris_response, bench_dog_response
);
criterion_main!(benches);

#[cfg(test)]
mod tests {

    #[test]
    fn test_extern_dog_response() {
        let img_width = 1024;
        let img_height = 1024;
        let img_src = image::ImageBuffer::from_pixel(img_width, img_height, image::Luma([1u8]));
        let mut img_dst = image::ImageBuffer::from_pixel(img_width, img_height, image::Luma([0u8]));

        let result = extern_dog_response_serial(&img_src, &mut img_dst, 0.5, 1.0);
        assert_eq!(result.dimensions(), (img_width, img_height));
    }
}
