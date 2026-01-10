use apriltag::DetectorBuilder;
use criterion::{criterion_group, criterion_main, Criterion};
use kornia_apriltag::{family::TagFamilyKind, AprilTagDecoder, DecodeTagsConfig};
use kornia_image::ImageSize;

const IMG_SIZE: ImageSize = ImageSize {
    width: 799,
    height: 533,
};

const BITS_CORRECTED: usize = 2;

fn bench_tagfamily(c: &mut Criterion) {
    {
        let mut group = c.benchmark_group("Tag16H5");
        group.bench_function("kornia-apriltag", |b| {
            b.iter(|| {
                let config = DecodeTagsConfig::new(vec![TagFamilyKind::Tag16H5]).unwrap();
                std::hint::black_box(AprilTagDecoder::new(config, IMG_SIZE).unwrap());
            });
        });
        group.bench_function("apriltag-c", |b| {
            b.iter(|| {
                std::hint::black_box(
                    DetectorBuilder::new()
                        .add_family_bits(apriltag::Family::tag_16h5(), BITS_CORRECTED)
                        .build()
                        .unwrap(),
                );
            });
        });
    }

    {
        let mut group = c.benchmark_group("Tag36H11");
        group.bench_function("kornia-apriltag", |b| {
            b.iter(|| {
                let config = DecodeTagsConfig::new(vec![TagFamilyKind::Tag36H11]).unwrap();
                std::hint::black_box(AprilTagDecoder::new(config, IMG_SIZE).unwrap());
            });
        });
        group.bench_function("apriltag-c", |b| {
            b.iter(|| {
                std::hint::black_box(
                    DetectorBuilder::new()
                        .add_family_bits(apriltag::Family::tag_36h11(), BITS_CORRECTED)
                        .build()
                        .unwrap(),
                );
            });
        });
    }

    // Note: Tag36H10 benchmarks are omitted because this family is not available in the apriltag-c bindings crate

    {
        let mut group = c.benchmark_group("Tag25H9");
        group.bench_function("kornia-apriltag", |b| {
            b.iter(|| {
                let config = DecodeTagsConfig::new(vec![TagFamilyKind::Tag25H9]).unwrap();
                std::hint::black_box(AprilTagDecoder::new(config, IMG_SIZE).unwrap());
            });
        });
        group.bench_function("apriltag-c", |b| {
            b.iter(|| {
                std::hint::black_box(
                    DetectorBuilder::new()
                        .add_family_bits(apriltag::Family::tag_25h9(), BITS_CORRECTED)
                        .build()
                        .unwrap(),
                );
            });
        });
    }

    {
        let mut group = c.benchmark_group("TagCircle21H7");
        group.bench_function("kornia-apriltag", |b| {
            b.iter(|| {
                let config = DecodeTagsConfig::new(vec![TagFamilyKind::TagCircle21H7]).unwrap();
                std::hint::black_box(AprilTagDecoder::new(config, IMG_SIZE).unwrap());
            });
        });
        group.bench_function("apriltag-c", |b| {
            b.iter(|| {
                std::hint::black_box(
                    DetectorBuilder::new()
                        .add_family_bits(apriltag::Family::tag_circle_21h7(), BITS_CORRECTED)
                        .build()
                        .unwrap(),
                );
            });
        });
    }

    {
        let mut group = c.benchmark_group("TagCircle49H12");
        group.bench_function("kornia-apriltag", |b| {
            b.iter(|| {
                let config = DecodeTagsConfig::new(vec![TagFamilyKind::TagCircle49H12]).unwrap();
                std::hint::black_box(AprilTagDecoder::new(config, IMG_SIZE).unwrap());
            });
        });
        group.bench_function("apriltag-c", |b| {
            b.iter(|| {
                std::hint::black_box(
                    DetectorBuilder::new()
                        .add_family_bits(apriltag::Family::tag_circle_49h12(), BITS_CORRECTED)
                        .build()
                        .unwrap(),
                );
            });
        });
    }

    {
        let mut group = c.benchmark_group("TagCustom48H12");
        group.bench_function("kornia-apriltag", |b| {
            b.iter(|| {
                let config = DecodeTagsConfig::new(vec![TagFamilyKind::TagCustom48H12]).unwrap();
                std::hint::black_box(AprilTagDecoder::new(config, IMG_SIZE).unwrap());
            });
        });
        group.bench_function("apriltag-c", |b| {
            b.iter(|| {
                std::hint::black_box(
                    DetectorBuilder::new()
                        .add_family_bits(apriltag::Family::tag_custom_48h12(), BITS_CORRECTED)
                        .build()
                        .unwrap(),
                );
            });
        });
    }

    {
        let mut group = c.benchmark_group("TagStandard41H12");
        group.bench_function("kornia-apriltag", |b| {
            b.iter(|| {
                let config = DecodeTagsConfig::new(vec![TagFamilyKind::TagStandard41H12]).unwrap();
                std::hint::black_box(AprilTagDecoder::new(config, IMG_SIZE).unwrap());
            });
        });
        group.bench_function("apriltag-c", |b| {
            b.iter(|| {
                std::hint::black_box(
                    DetectorBuilder::new()
                        .add_family_bits(apriltag::Family::tag_standard_41h12(), BITS_CORRECTED)
                        .build()
                        .unwrap(),
                );
            });
        });
    }

    {
        let mut group = c.benchmark_group("TagStandard52H13");
        group.bench_function("kornia-apriltag", |b| {
            b.iter(|| {
                let config = DecodeTagsConfig::new(vec![TagFamilyKind::TagStandard52H13]).unwrap();
                std::hint::black_box(AprilTagDecoder::new(config, IMG_SIZE).unwrap());
            });
        });
        group.bench_function("apriltag-c", |b| {
            b.iter(|| {
                std::hint::black_box(
                    DetectorBuilder::new()
                        .add_family_bits(apriltag::Family::tag_standard_52h13(), BITS_CORRECTED)
                        .build()
                        .unwrap(),
                );
            });
        });
    }
}

criterion_group!(benches, bench_tagfamily);
criterion_main!(benches);
