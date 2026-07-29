use std::path::PathBuf;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kornia_image::Image;
use kornia_imgproc::{
    color::gray_from_rgb_u8, features::*, interpolation::InterpolationMode, resize::resize_fast_rgb,
};
use kornia_io::functional as io;
use rand::RngExt;

fn bench_fast_corner_detect(c: &mut Criterion) {
    let mut group = c.benchmark_group("FastCornerDetect");

    let img_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/data/apriltags_tag36h11.jpg");
    let img_rgb8 = io::read_image_any_rgb8(img_path).unwrap();

    let new_size = [1920, 1080].into();
    let mut img_resized = Image::from_size_val(new_size, 0).unwrap();
    resize_fast_rgb(&img_rgb8, &mut img_resized, InterpolationMode::Bilinear).unwrap();

    let mut img_gray8 = Image::from_size_val(new_size, 0).unwrap();
    gray_from_rgb_u8(&img_resized, &mut img_gray8).unwrap();

    let mut img_grayf32 = Image::from_size_val(new_size, 0.0).unwrap();
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
                fast_detector.compute_corner_response(&src);
                let _res = std::hint::black_box(fast_detector.extract_keypoints()).unwrap();

                fast_detector.clear();
            })
        },
    );

    // Fused single-pass u8 path — same entry the Python `K.features.fast_detect`
    // binding uses. No dense f32 response buffer, no separate NMS pass.
    let threshold_norm = 20.0f32 / 255.0;
    let border = 3usize;
    let rows = border..(new_size.height - border);
    group.bench_with_input(
        BenchmarkId::new("fast_detect_rows_u8", &parameter_string),
        &img_gray8,
        |b, i| {
            b.iter(|| {
                let kps = fast_detect_rows_u8(i, threshold_norm, 9, border, rows.clone());
                std::hint::black_box(kps);
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

        let image_f32: Image<f32, 1> = Image::new(image_size, image_data).unwrap();

        // output image
        let response_f32: Image<f32, 1> = Image::from_size_val(image_size, 0.0).unwrap();
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

        let src = Image::<f32, 1>::from_size_val([*width, *height].into(), 1.0).unwrap();
        let mut dst = Image::<f32, 1>::from_size_val([*width, *height].into(), 0.0).unwrap();

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

fn load_mh01_gray_f32(name: &str) -> Image<f32, 1> {
    let img_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("../../tests/data/{name}"));
    let img = kornia_io::png::read_image_png_mono8(&img_path).expect("failed to read test PNG");
    let mut dst = Image::from_size_val(img.0.size(), 0.0f32).expect("alloc failed");
    img.0
        .as_slice()
        .iter()
        .zip(dst.as_slice_mut())
        .for_each(|(&s, d)| *d = s as f32 / 255.0);
    dst
}

fn bench_orb_detect_extract(c: &mut Criterion) {
    let frame = load_mh01_gray_f32("mh01_frame1.png");
    let mut group = c.benchmark_group("orb_detect_extract");
    group.sample_size(20);

    for &n_kp in &[500, 1000, 2000] {
        group.bench_with_input(BenchmarkId::new("n_keypoints", n_kp), &n_kp, |b, &n_kp| {
            let orb = OrbDetector {
                n_keypoints: n_kp,
                ..Default::default()
            };
            b.iter(|| std::hint::black_box(orb.detect_and_extract(&frame).unwrap()));
        });
    }
    group.finish();
}

fn bench_descriptor_matching(c: &mut Criterion) {
    let frame0 = load_mh01_gray_f32("mh01_frame1.png");
    let frame1 = load_mh01_gray_f32("mh01_frame2.png");

    let mut group = c.benchmark_group("descriptor_matching");
    group.sample_size(20);

    for &n_kp in &[500, 1000, 2000] {
        group.bench_with_input(BenchmarkId::new("n_keypoints", n_kp), &n_kp, |b, &n_kp| {
            let orb = OrbDetector {
                n_keypoints: n_kp,
                ..Default::default()
            };
            let feat0 = orb.detect_and_extract(&frame0).unwrap();
            let feat1 = orb.detect_and_extract(&frame1).unwrap();
            let config = OrbMatchConfig {
                nn_ratio: 0.6,
                th_low: 50,
                check_orientation: true,
                histo_length: 30,
            };
            b.iter(|| {
                std::hint::black_box(match_orb_descriptors(
                    &feat0.orientations,
                    &feat0.descriptors,
                    &feat1.orientations,
                    &feat1.descriptors,
                    config,
                ))
            });
        });
    }
    group.finish();
}

/// SIFT: the NEON detector end to end, its stages, and the descriptor matcher.
///
/// Grouped here with the other feature benchmarks rather than living as
/// env-gated `#[test]`s, so `cargo bench --bench bench_features -- Sift` reports
/// them next to FAST, Harris and ORB and criterion tracks the regression.
///
/// The CUDA path is not benchmarked from here: it needs a device, a stream and a
/// warm kernel cache, and mixing that into a CPU criterion group makes both
/// numbers harder to read. Use `KORNIA_SIFT_STAGES=1` through the Python
/// binding for it — see `docs/benchmark-sift.md`.
fn bench_sift(c: &mut Criterion) {
    let mut group = c.benchmark_group("Sift");

    // Same source image and working size as the other detector benches here, so
    // the numbers sit on a comparable scale.
    let img_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/data/apriltags_tag36h11.jpg");
    let img_rgb8 = io::read_image_any_rgb8(img_path).unwrap();
    let size = [752, 480].into();
    let mut resized = Image::from_size_val(size, 0).unwrap();
    resize_fast_rgb(&img_rgb8, &mut resized, InterpolationMode::Bilinear).unwrap();
    let mut gray8 = Image::from_size_val(size, 0).unwrap();
    gray_from_rgb_u8(&resized, &mut gray8).unwrap();
    // The reference works in 0..255 floats, not 0..1.
    let img: Vec<f32> = gray8.as_slice().iter().map(|&p| p as f32).collect();
    let (w, h) = (size.width, size.height);

    let cfg = SiftConfig::default();
    // One workspace per configuration, reused across iterations: allocating the
    // ~20 full-resolution planes per call is ~120 MB of overhead and would swamp
    // the measurement.
    for (name, fo, oct, fast) in [
        ("detect/fo=-1", FirstOctave::Double, usize::MAX, false),
        ("detect/fo=-1,fast", FirstOctave::Double, usize::MAX, true),
        ("detect/fo=0", FirstOctave::Native, usize::MAX, false),
        ("detect/fo=0,4oct", FirstOctave::Native, 4, false),
    ] {
        let mut ws = SiftWorkspace::new();
        group.bench_function(BenchmarkId::from_parameter(name), |b| {
            b.iter(|| {
                std::hint::black_box(
                    sift_detect_and_compute(&mut ws, &img, w, h, &cfg, fo, oct, fast).unwrap(),
                )
            })
        });
    }

    // Matching, on descriptors the detector actually produced.
    let mut ws = SiftWorkspace::new();
    let f = sift_detect_and_compute(
        &mut ws,
        &img,
        w,
        h,
        &cfg,
        FirstOctave::Double,
        usize::MAX,
        false,
    )
    .unwrap();
    let n = f.keypoints.len();
    if n > 0 {
        for (name, scalar) in [("match/neon", false), ("match/scalar", true)] {
            group.bench_function(BenchmarkId::from_parameter(name), |b| {
                b.iter(|| {
                    let m = if scalar {
                        sift_match_descriptors_scalar(
                            &f.descriptors,
                            n,
                            &f.descriptors,
                            n,
                            0.8,
                            true,
                        )
                    } else {
                        sift_match_descriptors(&f.descriptors, n, &f.descriptors, n, 0.8, true)
                    };
                    std::hint::black_box(m)
                })
            });
        }
    }
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().warm_up_time(std::time::Duration::new(10, 0));
    targets = bench_harris_response, bench_dog_response, bench_fast_corner_detect,
              bench_orb_detect_extract, bench_descriptor_matching, bench_sift
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
