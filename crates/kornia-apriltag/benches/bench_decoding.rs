#[cfg(feature = "compare-c")]
use apriltag::DetectorBuilder;
use criterion::{criterion_group, criterion_main, Criterion};
use kornia_apriltag::{family::TagFamilyKind, AprilTagDecoder, DecodeTagsConfig};
use kornia_image::{allocator::CpuAllocator, Image};
use kornia_imgproc::color::gray_from_rgb_u8;
use kornia_io::jpeg::read_image_jpeg_rgb8;
use std::path::PathBuf;

fn bench_decoding(c: &mut Criterion) {
    let img_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/data/apriltags_tag36h11.jpg");

    // Kornia
    let img = read_image_jpeg_rgb8(img_path).unwrap();
    let mut gray_img = Image::from_size_val(img.size(), 0, CpuAllocator).unwrap();
    gray_from_rgb_u8(&img, &mut gray_img).unwrap();

    let kornia_detector_config = DecodeTagsConfig::new(vec![TagFamilyKind::Tag36H11]).unwrap();
    let mut kornia_detector =
        AprilTagDecoder::new(kornia_detector_config, gray_img.size()).unwrap();

    // AprilTag C
    #[cfg(feature = "compare-c")]
    let mut apriltag_c_img =
        apriltag::Image::zeros_with_stride(gray_img.width(), gray_img.height(), gray_img.width())
            .unwrap();
    #[cfg(feature = "compare-c")]
    gray_img
        .as_slice()
        .iter()
        .zip(apriltag_c_img.as_slice_mut())
        .for_each(|(src, dst)| {
            *dst = *src;
        });

    #[cfg(feature = "compare-c")]
    let mut apriltag_c_detector = DetectorBuilder::new()
        .add_family_bits(apriltag::Family::tag_36h11(), 2)
        .build()
        .unwrap();

    #[cfg(feature = "compare-c")]
    apriltag_c_detector.set_decimation(2.0);

    // AprilGrid-rs
    #[cfg(feature = "compare-c")]
    let aprilgrid_img: DynamicImage =
        image::GrayImage::from_vec(img.width() as u32, img.height() as u32, img.to_vec())
            .unwrap()
            .into();

    #[cfg(feature = "compare-c")]
    let aprilgrid_detector =
        aprilgrid::detector::TagDetector::new(&aprilgrid::TagFamily::T36H11, None);

    c.bench_function("kornia-apriltag", |b| {
        b.iter(|| {
            std::hint::black_box(kornia_detector.decode(&gray_img).unwrap());
            kornia_detector.clear();
        });
    });

    #[cfg(feature = "compare-c")]
    c.bench_function("apriltag-c", |b| {
        b.iter(|| std::hint::black_box(apriltag_c_detector.detect(&apriltag_c_img)));
    });

    #[cfg(feature = "compare-c")]
    c.bench_function("aprilgrid-rs", |b| {
        b.iter(|| std::hint::black_box(aprilgrid_detector.detect(&aprilgrid_img)));
    });
}

criterion_group!(benches, bench_decoding);
criterion_main!(benches);
