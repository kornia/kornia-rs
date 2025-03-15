use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use kornia_image::Image;
use kornia_imgproc::color::gray_from_rgb_u8;
use kornia_imgproc::features::{fast_feature_detector, fast_feature_detector_nms};
use kornia_imgproc::interpolation::InterpolationMode;
use kornia_imgproc::resize::resize_fast;
use kornia_io::functional as io;

fn bench_fast_corner_detect(c: &mut Criterion) {
    let mut group = c.benchmark_group("FastCornerDetect");

    let img_rgb8 = io::read_image_any_rgb8("/home/edgar/Downloads/kodim08_grayscale.png").unwrap();

    let new_size = [1920, 1080].into();
    let mut img_resized = Image::from_size_val(new_size, 0).unwrap();
    resize_fast(&img_rgb8, &mut img_resized, InterpolationMode::Bilinear).unwrap();

    let mut img_gray8 = Image::from_size_val(new_size, 0).unwrap();
    gray_from_rgb_u8(&img_resized, &mut img_gray8).unwrap();

    let parameter_string = format!("{}x{}", new_size.width, new_size.height);

    group.bench_with_input(
        BenchmarkId::new("fast_native_cpu", &parameter_string),
        &(img_gray8),
        |b, i| {
            let src = i.clone();
            b.iter(|| {
                let _res = black_box(fast_feature_detector(&src, 60, 9)).unwrap();
            })
        },
    );

    //group.bench_with_input(
    //    BenchmarkId::new("fast_nms_cpu", &parameter_string),
    //    &(img_gray8),
    //    |b, i| {
    //        let src = i.clone();
    //        b.iter(|| {
    //            let _res = black_box(fast_feature_detector_nms(&src, 10, 3)).unwrap();
    //        })
    //    },
    //);

    group.finish();
}

criterion_group!(benches, bench_fast_corner_detect);
criterion_main!(benches);
