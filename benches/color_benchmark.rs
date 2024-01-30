use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use kornia_rs::color as F;
use kornia_rs::image::{Image, ImageSize};
use ndarray::{s, stack, Axis};

#[cfg(feature = "candle")]
use candle_core::{DType, Device, Storage, Tensor};

#[cfg(feature = "candle")]
use std::ops::Deref;

// vanilla version
fn gray_iter(image: Image) -> Image {
    let height = image.image_size().height;
    let width = image.image_size().width;
    let mut gray_image = Image::new(ImageSize { width, height }, vec![0; width * height * 3]);
    for y in 0..height {
        for x in 0..width {
            let r = image.data[[y, x, 0]];
            let g = image.data[[y, x, 1]];
            let b = image.data[[y, x, 2]];
            let gray_pixel = (76. * r as f32 + 150. * g as f32 + 29. * b as f32) / 255.;
            gray_image.data[[y, x, 0]] = gray_pixel as u8;
            gray_image.data[[y, x, 1]] = gray_pixel as u8;
            gray_image.data[[y, x, 2]] = gray_pixel as u8;
        }
    }
    gray_image
}

fn gray_vec(image: Image) -> Image {
    // convert to f32
    let mut image_f32 = image.data.mapv(|x| x as f32);

    // get channels
    let mut binding = image_f32.view_mut();
    let (r, g, b) = binding.multi_slice_mut((s![.., .., 0], s![.., .., 1], s![.., .., 2]));

    // weighted sum
    // TODO: check data type, for u8 or f32/f64
    let gray_f32 = (&r * 76.0 + &g * 150.0 + &b * 29.0) / 255.0;
    let gray_u8 = gray_f32.mapv(|x| x as u8);

    // TODO: ideally we stack the channels. Not working yet.
    let gray_stacked = match stack(Axis(2), &[gray_u8.view(), gray_u8.view(), gray_u8.view()]) {
        Ok(gray_stacked) => gray_stacked,
        Err(err) => {
            panic!("Error stacking channels: {}", err);
        }
    };
    Image { data: gray_stacked }
}

#[cfg(feature = "candle")]
fn gray_candle(image: Image) -> Image {
    let image_data = image.data.as_slice().unwrap();
    let shape = (image.image_size().height, image.image_size().width, 3);

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

    Image::from_shape_vec([shape.0, shape.1, shape.2], data)
}

fn bench_grayscale(c: &mut Criterion) {
    let mut group = c.benchmark_group("Grayscale");
    let image_sizes = vec![(256, 224), (512, 448), (1024, 896)];

    for (width, height) in image_sizes {
        let image_size = ImageSize { width, height };
        let id = format!("{}x{}", width, height);
        let image = Image::new(image_size.clone(), vec![0; width * height * 3]);
        group.bench_with_input(BenchmarkId::new("zip", &id), &image, |b, i| {
            b.iter(|| F::gray_from_rgb(black_box(i.clone())))
        });
        group.bench_with_input(BenchmarkId::new("iter", &id), &image, |b, i| {
            b.iter(|| gray_iter(black_box(i.clone())))
        });
        group.bench_with_input(BenchmarkId::new("vec", &id), &image, |b, i| {
            b.iter(|| gray_vec(black_box(i.clone())))
        });
        #[cfg(feature = "candle")]
        group.bench_with_input(BenchmarkId::new("candle", &id), &image, |b, i| {
            b.iter(|| gray_candle(black_box(i.clone())))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_grayscale);
criterion_main!(benches);
