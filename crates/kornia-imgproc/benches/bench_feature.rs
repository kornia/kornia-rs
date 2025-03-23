use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use kornia_image::Image;
use kornia_imgproc::features::dog_response;

fn bench_dog_response(c: &mut Criterion) {
    let mut group = c.benchmark_group("Dog Response");
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

criterion_group!(benches, bench_dog_response);
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
