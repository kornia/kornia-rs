use criterion::{criterion_group, criterion_main, Criterion};
use half::f16;
use kornia_image::{Image, ImageSize};
use kornia_tensor::CpuAllocator;
use std::hint::black_box;

fn sample_image() -> Image<u8, 3, CpuAllocator> {
    Image::from_size_val(
        ImageSize {
            width: 1920,
            height: 1080,
        },
        127,
        CpuAllocator,
    )
    .unwrap()
}

fn bench_image(c: &mut Criterion) {
    let mut group = c.benchmark_group("Image");

    group.bench_function("cast_and_scale_f16", |b| {
        b.iter_batched(
            sample_image,
            |image| {
                black_box(image)
                    .cast_and_scale(f16::from_f32(1. / 255.))
                    .unwrap()
            },
            criterion::BatchSize::LargeInput,
        )
    });

    group.bench_function("cast_and_scale_f32", |b| {
        b.iter_batched(
            sample_image,
            |image| black_box(image).cast_and_scale(1.0f32 / 255.0f32).unwrap(),
            criterion::BatchSize::LargeInput,
        )
    });

    group.bench_function("cast_and_scale_f64", |b| {
        b.iter_batched(
            sample_image,
            |image| black_box(image).cast_and_scale(1.0f64 / 255.0f64).unwrap(),
            criterion::BatchSize::LargeInput,
        )
    });

    group.bench_function("scale_and_cast_f16", |b| {
        b.iter_batched(
            sample_image,
            |image| black_box(image).scale_and_cast::<f16>(1_u8).unwrap(),
            criterion::BatchSize::LargeInput,
        )
    });

    group.bench_function("scale_and_cast_f32", |b| {
        b.iter_batched(
            sample_image,
            |image| black_box(image).scale_and_cast::<f32>(1_u8).unwrap(),
            criterion::BatchSize::LargeInput,
        )
    });

    group.bench_function("scale_and_cast_f64", |b| {
        b.iter_batched(
            sample_image,
            |image| black_box(image).scale_and_cast::<f64>(1_u8).unwrap(),
            criterion::BatchSize::LargeInput,
        )
    });
}

criterion_group!(benches, bench_image);
criterion_main!(benches);
