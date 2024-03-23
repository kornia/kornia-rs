use criterion::{black_box, criterion_group, criterion_main, Criterion};

struct JpegReader {
    decoder: kornia_rs::io::jpeg::ImageDecoder,
}

impl JpegReader {
    fn new() -> Self {
        Self {
            decoder: kornia_rs::io::jpeg::ImageDecoder::new().unwrap(),
        }
    }

    fn read(&mut self, file_path: &std::path::Path) -> kornia_rs::image::Image<u8, 3> {
        let file = std::fs::File::open(file_path).unwrap();
        let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
        self.decoder.decode(&mmap).unwrap()
    }
}

fn read_no_mmap(file_path: &std::path::Path) -> kornia_rs::image::Image<u8, 3> {
    let file = std::fs::read(file_path).unwrap();
    kornia_rs::io::jpeg::ImageDecoder::new()
        .unwrap()
        .decode(&file)
        .unwrap()
}

fn bench_read_jpeg(c: &mut Criterion) {
    let mut group = c.benchmark_group("read_jpeg");

    let img_path = std::path::Path::new("tests/data/dog.jpeg");

    // NOTE: this is the fastest method
    group.bench_function("jpegturbo", |b| {
        b.iter(|| kornia_rs::io::functional::read_image_jpeg(black_box(img_path)).unwrap())
    });

    group.bench_function("image", |b| {
        b.iter(|| kornia_rs::io::functional::read_image_any(black_box(img_path)).unwrap())
    });

    // NOTE: similar to the functional::read_image_jpeg
    group.bench_function("jpeg_reader", |b| {
        b.iter(|| JpegReader::new().read(black_box(img_path)))
    });

    group.bench_function("no_mmap", |b| b.iter(|| read_no_mmap(black_box(img_path))));

    group.finish();
}

criterion_group!(benches, bench_read_jpeg);
criterion_main!(benches);
