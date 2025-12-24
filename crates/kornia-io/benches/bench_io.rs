use criterion::{criterion_group, criterion_main, Criterion};
use kornia_io::{jpeg::read_image_jpeg_rgb8, jpegturbo::read_image_jpegturbo_rgb8};

fn bench_read_jpeg(c: &mut Criterion) {
    let mut group = c.benchmark_group("JpegReader");

    let img_path = "../../tests/data/dog.jpeg";

    // NOTE: this is the fastest method
    group.bench_function("jpegturbo", |b| {
        b.iter(|| std::hint::black_box(read_image_jpegturbo_rgb8(img_path)).unwrap())
    });

    group.bench_function("image_any", |b| {
        b.iter(|| std::hint::black_box(read_image_jpeg_rgb8(img_path)).unwrap())
    });

    group.finish();
}

criterion_group!(benches, bench_read_jpeg);
criterion_main!(benches);
