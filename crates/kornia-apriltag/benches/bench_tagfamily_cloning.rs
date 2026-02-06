use criterion::{criterion_group, criterion_main, Criterion};
use kornia_apriltag::family::{TagFamily, TagFamilyKind};
use std::{hint::black_box, sync::Arc};

fn bench_single_clone(c: &mut Criterion) {
    let heavy_family = TagFamily::tag36_h11().unwrap();
    let kind_custom = TagFamilyKind::Custom(Arc::new(heavy_family.clone()));
    let kind_standard = TagFamilyKind::Tag36H11;

    let mut group = c.benchmark_group("Single Clone");

    group.bench_function("Deep Clone (TagFamily)", |b| {
        b.iter(|| {
            black_box(heavy_family.clone());
        });
    });

    group.bench_function("Arc Clone (Custom)", |b| {
        b.iter(|| {
            black_box(kind_custom.clone());
        });
    });

    group.bench_function("Enum Clone (Standard)", |b| {
        b.iter(|| {
            black_box(kind_standard.clone());
        });
    });

    group.finish();
}

fn bench_repeated_clones(c: &mut Criterion) {
    let heavy_family = TagFamily::tag36_h11().unwrap();
    let kind_custom = TagFamilyKind::Custom(Arc::new(heavy_family.clone()));

    let mut group = c.benchmark_group("Repeated Clones (10x)");

    group.bench_function("Deep Clone", |b| {
        b.iter(|| {
            for _ in 0..10 {
                black_box(heavy_family.clone());
            }
        });
    });

    group.bench_function("Arc Clone", |b| {
        b.iter(|| {
            for _ in 0..10 {
                black_box(kind_custom.clone());
            }
        });
    });

    group.finish();
}

fn bench_different_families(c: &mut Criterion) {
    let tag16h5 = TagFamily::tag16_h5().unwrap();
    let tag36h11 = TagFamily::tag36_h11().unwrap();
    let tagstandard52h13 = TagFamily::tagstandard52_h13().unwrap();

    let mut group = c.benchmark_group("Deep Clone by Family");

    group.bench_function("Tag16H5 (Small)", |b| {
        b.iter(|| {
            black_box(tag16h5.clone());
        });
    });

    group.bench_function("Tag36H11 (Medium)", |b| {
        b.iter(|| {
            black_box(tag36h11.clone());
        });
    });

    group.bench_function("TagStandard52H13 (Large)", |b| {
        b.iter(|| {
            black_box(tagstandard52h13.clone());
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_single_clone,
    bench_repeated_clones,
    bench_different_families
);
criterion_main!(benches);
