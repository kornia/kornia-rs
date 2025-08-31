use std::path::PathBuf;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kornia_image::Image;
use kornia_imgproc::{
    color::gray_from_rgb_u8, features::*, interpolation::InterpolationMode, resize::resize_fast_rgb,
};
use kornia_io::functional as io;
use kornia_tensor::CpuAllocator;
use rand::Rng;

fn bench_fast_corner_detect(c: &mut Criterion) {
    let mut group = c.benchmark_group("FastCornerDetect");

    let img_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/data/apriltags_tag36h11.jpg");
    let img_rgb8 = io::read_image_any_rgb8(img_path).unwrap();

    let new_size = [1920, 1080].into();
    let mut img_resized = Image::from_size_val(new_size, 0, CpuAllocator).unwrap();
    resize_fast_rgb(&img_rgb8, &mut img_resized, InterpolationMode::Bilinear).unwrap();

    let mut img_gray8 = Image::from_size_val(new_size, 0, CpuAllocator).unwrap();
    gray_from_rgb_u8(&img_resized, &mut img_gray8).unwrap();

    let mut img_grayf32 = Image::from_size_val(new_size, 0.0, CpuAllocator).unwrap();
    img_gray8
        .as_slice()
        .iter()
        .zip(img_grayf32.as_slice_mut())
        .for_each(|(&p, m)| {
            *m = p as f32 / 255.0;
        });

    let mut fast_detector = FastDetector::new(new_size, 0.23, 9, 1).unwrap();

    let parameter_string = format!("{}x{}", new_size.width, new_size.height);

    group.bench_with_input(
        BenchmarkId::new("fast_native_cpu", &parameter_string),
        &(img_grayf32),
        |b, i| {
            let src = i.clone();
            b.iter(|| {
                fast_detector.compute_corner_response(&src).unwrap();
                let _res = std::hint::black_box(fast_detector.extract_keypoints());

                fast_detector.clear();
            })
        },
    );
}

fn bench_harris_response(c: &mut Criterion) {
    let mut group = c.benchmark_group("Features");
    let mut rng = rand::rng();

    for (width, height) in [(224, 224), (1920, 1080)].iter() {
        group.throughput(criterion::Throughput::Elements((*width * *height) as u64));

        let parameter_string = format!("{width}x{height}");

        // input image
        let image_data: Vec<f32> = (0..(*width * *height))
            .map(|_| rng.random_range(0.0..1.0))
            .collect();
        let image_size = [*width, *height].into();

        let image_f32: Image<f32, 1, _> = Image::new(image_size, image_data, CpuAllocator).unwrap();

        // output image
        let response_f32: Image<f32, 1, _> =
            Image::from_size_val(image_size, 0.0, CpuAllocator).unwrap();
        let mut harris_response = HarrisResponse::new(image_size);

        group.bench_with_input(
            BenchmarkId::new("harris", &parameter_string),
            &(&image_f32, &response_f32),
            |b, i| {
                let (src, mut dst) = (i.0, i.1.clone());
                b.iter(|| std::hint::black_box(harris_response.compute(src, &mut dst)))
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

        let src =
            Image::<f32, 1, _>::from_size_val([*width, *height].into(), 1.0, CpuAllocator).unwrap();
        let mut dst =
            Image::<f32, 1, _>::from_size_val([*width, *height].into(), 0.0, CpuAllocator).unwrap();

        // Benchmark DoG response (serial version)
        group.bench_with_input(
            BenchmarkId::new("dog_response", format!("{width}x{height}")),
            &(width, height),
            |b, _| {
                b.iter(|| {
                    dog_response(
                        std::hint::black_box(&src),
                        std::hint::black_box(&mut dst),
                        std::hint::black_box(0.5),
                        std::hint::black_box(1.0),
                    )
                    .unwrap()
                })
            },
        );
    }

    group.finish();
}

fn bench_orb(c: &mut Criterion) {
    let mut group = c.benchmark_group("OrbDetector");

    let img_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/data/apriltags_tag36h11.jpg");
    let img_rgb8 = io::read_image_any_rgb8(img_path).unwrap();

    let new_size = [1920, 1080].into();
    let mut img_resized = Image::from_size_val(new_size, 0, CpuAllocator).unwrap();
    resize_fast_rgb(&img_rgb8, &mut img_resized, InterpolationMode::Bilinear).unwrap();

    let mut img_gray8 = Image::from_size_val(new_size, 0, CpuAllocator).unwrap();
    gray_from_rgb_u8(&img_resized, &mut img_gray8).unwrap();

    let mut img_grayf32 = Image::from_size_val(new_size, 0.0, CpuAllocator).unwrap();
    img_gray8
        .as_slice()
        .iter()
        .zip(img_grayf32.as_slice_mut())
        .for_each(|(&p, m)| {
            *m = p as f32 / 255.0;
        });

    let mut orb = OrbDectector::new(OrbDectectorConfig::default(), new_size).unwrap();
    let parameter_string = format!("{}x{}", new_size.width, new_size.height);

    group.bench_with_input(
        BenchmarkId::new("orb_detect", &parameter_string),
        &img_grayf32,
        |b, i| {
            b.iter(|| {
                let detection = orb.detect(i).unwrap();
                let _ = std::hint::black_box(orb.extract(i, &detection)).unwrap();
            })
        },
    );
    group.finish();
}

fn bench_orb_matching(c: &mut Criterion) {
    let mut group = c.benchmark_group("OrbDescriptorMatching");

    let img_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/data/apriltags_tag36h11.jpg");
    let img_rgb8 = io::read_image_any_rgb8(img_path).unwrap();

    let new_size = [1920, 1080].into();
    let mut img_resized = Image::from_size_val(new_size, 0, CpuAllocator).unwrap();
    resize_fast_rgb(&img_rgb8, &mut img_resized, InterpolationMode::Bilinear).unwrap();

    let mut img_gray8 = Image::from_size_val(new_size, 0, CpuAllocator).unwrap();
    gray_from_rgb_u8(&img_resized, &mut img_gray8).unwrap();

    let mut img_grayf32 = Image::from_size_val(new_size, 0.0, CpuAllocator).unwrap();
    img_gray8
        .as_slice()
        .iter()
        .zip(img_grayf32.as_slice_mut())
        .for_each(|(&p, m)| {
            *m = p as f32 / 255.0;
        });

    // Create a vertically flipped version
    let mut img_grayf32_flipped = Image::from_size_val(new_size, 0.0, CpuAllocator).unwrap();
    kornia_imgproc::flip::vertical_flip(&img_grayf32, &mut img_grayf32_flipped).unwrap();

    let mut orb = OrbDectector::new(OrbDectectorConfig::default(), img_grayf32.size()).unwrap();

    // Detect and extract descriptors for both images
    let detection1 = orb.detect(&img_grayf32).unwrap();
    let (desc1, _) = orb.extract(&img_grayf32, &detection1).unwrap();

    let detection2 = orb.detect(&img_grayf32_flipped).unwrap();
    let (desc2, _) = orb.extract(&img_grayf32_flipped, &detection2).unwrap();

    let parameter_string = format!("{}x{}", new_size.width, new_size.height);

    group.bench_with_input(
        BenchmarkId::new("orb_match", &parameter_string),
        &(desc1, desc2),
        |b, (d1, d2)| {
            b.iter(|| {
                let _ = std::hint::black_box(kornia_imgproc::features::match_descriptors(
                    d1, d2, None, true, None,
                ));
            })
        },
    );
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().warm_up_time(std::time::Duration::new(10, 0));
    targets = bench_harris_response, bench_dog_response, bench_fast_corner_detect, bench_orb, bench_orb_matching
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
