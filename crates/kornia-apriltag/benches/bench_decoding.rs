use apriltag::DetectorBuilder;
use criterion::{criterion_group, criterion_main, Criterion};
use kornia_apriltag::{family::TagFamilyKind, AprilTagDecoder, DecodeTagsConfig};
use kornia_image::Image;
use kornia_imgproc::color::gray_from_rgb_u8;
use kornia_io::jpeg::read_image_jpeg_rgb8;
use std::path::PathBuf;

fn bench_decoding(c: &mut Criterion) {
    let img_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/data/apriltags_tag36h11.jpg");

    // Kornia
    let img = read_image_jpeg_rgb8(img_path).unwrap();
    let mut gray_img = Image::from_size_val(img.size(), 0).unwrap();
    gray_from_rgb_u8(&img, &mut gray_img).unwrap();

    let kornia_detector_config = DecodeTagsConfig::new(vec![TagFamilyKind::Tag36H11]).unwrap();
    let mut kornia_detector =
        AprilTagDecoder::new(kornia_detector_config, gray_img.size()).unwrap();

    // AprilTag C
    let mut apriltag_c_img =
        apriltag::Image::zeros_with_stride(gray_img.width(), gray_img.height(), gray_img.width())
            .unwrap();
    gray_img
        .as_slice()
        .iter()
        .zip(apriltag_c_img.as_slice_mut())
        .for_each(|(src, dst)| {
            *dst = *src;
        });

    let mut apriltag_c_detector = DetectorBuilder::new()
        .add_family_bits(apriltag::Family::tag_36h11(), 2)
        .build()
        .unwrap();

    apriltag_c_detector.set_decimation(2.0);

    // AprilGrid-rs
    let aprilgrid_img: image::DynamicImage =
        image::GrayImage::from_vec(img.width() as u32, img.height() as u32, img.to_vec())
            .unwrap()
            .into();

    let aprilgrid_detector =
        aprilgrid::detector::TagDetector::new(&aprilgrid::TagFamily::T36H11, None);

    // One-shot stage breakdown before the criterion loops.
    {
        let mut total_us = [0u64; 6];
        const WARMUP: usize = 20;
        for _ in 0..WARMUP {
            let _ = kornia_detector.decode_timed(&gray_img).unwrap();
            kornia_detector.clear();
        }
        const SAMPLES: usize = 50;
        for _ in 0..SAMPLES {
            let (_, us) = kornia_detector.decode_timed(&gray_img).unwrap();
            kornia_detector.clear();
            for i in 0..6 { total_us[i] += us[i]; }
        }
        eprintln!(
            "stages (avg µs over {} samples): decimate={} threshold={} conn_comp={} clusters={} fit_quads={} decode={}  total={}",
            SAMPLES,
            total_us[0] / SAMPLES as u64,
            total_us[1] / SAMPLES as u64,
            total_us[2] / SAMPLES as u64,
            total_us[3] / SAMPLES as u64,
            total_us[4] / SAMPLES as u64,
            total_us[5] / SAMPLES as u64,
            total_us.iter().sum::<u64>() / SAMPLES as u64,
        );
    }

    c.bench_function("kornia-apriltag", |b| {
        b.iter(|| {
            std::hint::black_box(kornia_detector.decode(&gray_img).unwrap());
            kornia_detector.clear();
        });
    });

    c.bench_function("apriltag-c", |b| {
        b.iter(|| std::hint::black_box(apriltag_c_detector.detect(&apriltag_c_img)));
    });

    c.bench_function("aprilgrid-rs", |b| {
        b.iter(|| std::hint::black_box(aprilgrid_detector.detect(&aprilgrid_img)));
    });
}

criterion_group!(benches, bench_decoding);
criterion_main!(benches);
