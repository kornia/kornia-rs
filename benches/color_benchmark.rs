use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use kornia_rs::color as F;
use kornia_rs::image::{Image, ImageSize};
use ndarray::s;

#[cfg(feature = "candle")]
use candle_core::{DType, Device, Storage, Tensor};

#[cfg(feature = "candle")]
use std::ops::Deref;

// vanilla version
fn gray_iter(image: &Image<f32, 3>) -> Image<u8, 1> {
    let data = vec![0u8; image.size().width * image.size().height];
    let mut gray_image = Image::new(image.size(), data).unwrap();
    for y in 0..image.height() {
        for x in 0..image.width() {
            let r = image.data[[y, x, 0]];
            let g = image.data[[y, x, 1]];
            let b = image.data[[y, x, 2]];
            let gray_pixel = (76. * r + 150. * g + 29. * b) / 255.;
            gray_image.data[[y, x, 0]] = gray_pixel as u8;
        }
    }
    gray_image
}

fn gray_vec(image: &Image<f32, 3>) -> Image<u8, 1> {
    let mut image_f32 = image.data.mapv(|x| x);

    // get channels
    let mut binding = image_f32.view_mut();
    let (r, g, b) = binding.multi_slice_mut((s![.., .., 0], s![.., .., 1], s![.., .., 2]));

    // weighted sum
    // TODO: check data type, for u8 or f32/f64
    let gray_f32 = (&r * 76.0 + &g * 150.0 + &b * 29.0) / 255.0;
    let gray_u8 = gray_f32.mapv(|x| x as u8);

    Image::new(image.size(), gray_u8.into_raw_vec()).unwrap()
}

#[cfg(feature = "candle")]
fn gray_candle(image: Image<f32, 3>) -> Image<u8, 3> {
    let image_data = image.data.as_slice().unwrap();
    let shape = (image.size().height, image.size().width, 3);

    let device = Device::Cpu;
    let image_u8 = Tensor::from_vec(image_data.to_vec(), shape, &device).unwrap();
    //println!("image_t: {:?}", image_u8.shape());

    let image_f32 = image_u8.to_dtype(DType::F32).unwrap();

    let weight = Tensor::from_vec(vec![76.0f32, 150.0, 29.0], (1, 1, 3), &device).unwrap();

    let gray_f32 = image_f32.broadcast_mul(&weight).unwrap();
    let gray_f32 = gray_f32.sum_keepdim(2).unwrap();
    let gray_f32 = (gray_f32 / 255.0).unwrap();
    let gray_f32 = gray_f32.repeat((1, 1, 3)).unwrap();
    //println!("gray_f32: {:?}", gray_f32.shape());

    let gray_u8 = gray_f32.to_dtype(DType::U8).unwrap();

    // https://github.com/huggingface/candle/issues/973
    let (storage, _layout) = gray_u8.storage_and_layout();

    let data = match storage.deref() {
        Storage::Cpu(storage) => {
            let data = storage.as_slice().unwrap();
            data.to_vec()
        }
        Storage::Cuda(_) => {
            panic!("Cuda not implemented yet");
        }
        Storage::Metal(_) => {
            panic!("Metal not implemented yet");
        }
    };

    Image::new(image.size(), data).unwrap()
}

fn gray_image_crate(image: &Image<u8, 3>) -> Image<u8, 1> {
    let image_data = image.data.as_slice().unwrap();
    let rgb = image::RgbImage::from_raw(
        image.size().width as u32,
        image.size().height as u32,
        image_data.to_vec(),
    )
    .unwrap();
    let image_crate = image::DynamicImage::ImageRgb8(rgb);

    let image_gray = image_crate.grayscale();

    Image::new(image.size(), image_gray.into_bytes()).unwrap()
}

fn bench_grayscale(c: &mut Criterion) {
    let mut group = c.benchmark_group("Grayscale");
    let image_sizes = vec![(256, 224), (512, 448), (1024, 896)];

    for (width, height) in image_sizes {
        let id = format!("{}x{}", width, height);
        let image_data = vec![0u8; width * height * 3];
        let image = Image::new(ImageSize { width, height }, image_data).unwrap();
        let image_f32 = image.clone().cast::<f32>().unwrap();
        group.bench_with_input(BenchmarkId::new("zip", &id), &image_f32, |b, i| {
            b.iter(|| F::gray_from_rgb(black_box(i)))
        });
        group.bench_with_input(BenchmarkId::new("iter", &id), &image_f32, |b, i| {
            b.iter(|| gray_iter(black_box(&i.clone())))
        });
        group.bench_with_input(BenchmarkId::new("vec", &id), &image_f32, |b, i| {
            b.iter(|| gray_vec(black_box(&i.clone())))
        });
        group.bench_with_input(BenchmarkId::new("image_crate", &id), &image, |b, i| {
            b.iter(|| gray_image_crate(black_box(&i.clone())))
        });
        #[cfg(feature = "candle")]
        group.bench_with_input(BenchmarkId::new("candle", &id), &image_f32, |b, i| {
            b.iter(|| gray_candle(black_box(i.clone())))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_grayscale);
criterion_main!(benches);
