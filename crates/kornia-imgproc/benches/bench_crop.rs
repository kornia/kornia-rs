use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use kornia_image::{Image, ImageSize};

fn bench_resize(c: &mut Criterion) {
    let mut group = c.benchmark_group("Crop");

    for (width, height) in [(256, 224), (512, 448), (1024, 896)].iter() {
        group.throughput(criterion::Throughput::Elements((*width * *height) as u64));

        let parameter_string = format!("{}x{}", width, height);

        // input image
        let image_size = [*width, *height].into();
        let data = vec![0u8; width * height * 3];
        let image = Image::<u8, 3>::new(image_size, data).unwrap();
        let (x, y) = (13, 21);

        // output image
        let new_size = ImageSize {
            width: width / 2,
            height: height / 2,
        };

        let out_u8 = Image::<u8, 3>::from_size_val(new_size, 0).unwrap();

        group.bench_with_input(
            BenchmarkId::new("image_rs", &parameter_string),
            &image,
            |b, i| {
                let mut image = image::DynamicImage::ImageRgb8(
                    image::RgbImage::from_raw(
                        i.size().width as u32,
                        i.size().height as u32,
                        i.as_slice().to_vec(),
                    )
                    .unwrap(),
                );
                b.iter(|| {
                    let _image_cropped = image.crop(
                        x as u32,
                        y as u32,
                        new_size.width as u32,
                        new_size.height as u32,
                    );
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("kornia_par", &parameter_string),
            &(&image, &out_u8),
            |b, i| {
                let (src, mut dst) = (i.0.clone(), i.1.clone());
                b.iter(|| {
                    kornia_imgproc::crop::crop_image(
                        black_box(&src),
                        black_box(&mut dst),
                        black_box(0),
                        black_box(0),
                    )
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_resize);
criterion_main!(benches);
