use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn read_turbo_no_mmap(file_path: &std::path::Path) -> kornia_rs::image::Image<u8, 3> {
    let file = std::fs::read(file_path).unwrap();
    kornia_rs::io::jpeg::ImageDecoder::new()
        .unwrap()
        .decode(&file)
        .unwrap()
}

fn read_zune_jpeg(file_path: &std::path::Path) -> kornia_rs::image::Image<u8, 3> {
    let data = std::fs::read(file_path).unwrap();
    let mut decoder = zune_jpeg::JpegDecoder::new(&data);
    decoder.decode_headers().unwrap();
    let image_info = decoder.info().unwrap();
    let image_data = decoder.decode().unwrap();
    kornia_rs::image::Image::new(
        kornia_rs::image::ImageSize {
            width: image_info.width as usize,
            height: image_info.height as usize,
        },
        image_data,
    )
    .unwrap()
}

fn bench_read_jpeg(c: &mut Criterion) {
    let mut group = c.benchmark_group("read_jpeg");

    let img_path = std::path::Path::new("tests/data/dog.jpeg");

    // NOTE: this is the fastest method
    group.bench_function("jpegturbo", |b| {
        b.iter(|| kornia_rs::io::functional::read_image_jpeg(black_box(img_path)).unwrap())
    });

    group.bench_function("image_any", |b| {
        b.iter(|| kornia_rs::io::functional::read_image_any(black_box(img_path)).unwrap())
    });

    group.bench_function("turbo_no_mmap", |b| {
        b.iter(|| read_turbo_no_mmap(black_box(img_path)))
    });

    group.bench_function("zune_jpeg", |b| {
        b.iter(|| read_zune_jpeg(black_box(img_path)))
    });

    group.finish();
}

criterion_group!(benches, bench_read_jpeg);
criterion_main!(benches);
