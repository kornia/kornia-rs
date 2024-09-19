use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use kornia_image::Image;
use kornia_imgproc::color::gray_from_rgb;
use rayon::prelude::*;

// vanilla version
fn gray_vanilla_get_unchecked(
    src: &Image<f32, 3>,
    dst: &mut Image<f32, 1>,
) -> Result<(), Box<dyn std::error::Error>> {
    let data = dst.as_slice_mut();
    let (cols, _rows) = (src.cols(), src.rows());

    for y in 0..src.rows() {
        for x in 0..src.cols() {
            let r = src.get_unchecked([y, x, 0]);
            let g = src.get_unchecked([y, x, 1]);
            let b = src.get_unchecked([y, x, 2]);
            let gray_pixel = (76. * r + 150. * g + 29. * b) / 255.;
            data[y * cols + x] = gray_pixel;
        }
    }
    Ok(())
}

fn gray_ndarray_zip_par(
    src: &Image<f32, 3>,
    dst: &mut Image<f32, 1>,
) -> Result<(), Box<dyn std::error::Error>> {
    const RW: f32 = 76. / 255.;
    const GW: f32 = 150. / 255.;
    const BW: f32 = 29. / 255.;

    let src_data = unsafe {
        ndarray::ArrayView3::from_shape_ptr((src.height(), src.width(), 3), src.as_ptr())
    };

    let mut dst_data = unsafe {
        ndarray::ArrayViewMut3::from_shape_ptr((dst.height(), dst.width(), 1), dst.as_mut_ptr())
    };

    ndarray::Zip::from(dst_data.rows_mut())
        .and(src_data.rows())
        .par_for_each(|mut out, inp| {
            assert_eq!(inp.len(), 3);
            let r = inp[0];
            let g = inp[1];
            let b = inp[2];
            out[0] = RW * r + GW * g + BW * b;
        });

    Ok(())
}

fn gray_slice_chunks_pixels(
    src: &Image<f32, 3>,
    dst: &mut Image<f32, 1>,
) -> Result<(), Box<dyn std::error::Error>> {
    src.as_slice()
        .chunks_exact(3)
        .zip(dst.storage.as_mut_slice().chunks_exact_mut(1))
        .for_each(|(src_chunk, dst_chunk)| {
            let r = src_chunk[0];
            let g = src_chunk[1];
            let b = src_chunk[2];
            dst_chunk[0] = (76. * r + 150. * g + 29. * b) / 255.;
        });

    Ok(())
}

fn gray_slice_chunks_pixels_parallel(
    src: &Image<f32, 3>,
    dst: &mut Image<f32, 1>,
) -> Result<(), Box<dyn std::error::Error>> {
    src.as_slice()
        .par_chunks_exact(3)
        .zip(dst.as_slice_mut().par_chunks_exact_mut(1))
        .for_each(|(src_chunk, dst_chunk)| {
            let r = src_chunk[0];
            let g = src_chunk[1];
            let b = src_chunk[2];
            dst_chunk[0] = (76. * r + 150. * g + 29. * b) / 255.;
        });

    Ok(())
}

fn gray_slice_chunks_rows(
    src: &Image<f32, 3>,
    dst: &mut Image<f32, 1>,
) -> Result<(), Box<dyn std::error::Error>> {
    let num_channelsols = src.cols();
    src.as_slice()
        .chunks_exact(3 * num_channelsols)
        .zip(dst.storage.as_mut_slice().chunks_exact_mut(num_channelsols))
        .for_each(|(src_chunk, dst_chunk)| {
            src_chunk
                .chunks_exact(3)
                .zip(dst_chunk.chunks_exact_mut(1))
                .for_each(|(src_pixel, dst_pixel)| {
                    let r = src_pixel[0];
                    let g = src_pixel[1];
                    let b = src_pixel[2];
                    dst_pixel[0] = (76. * r + 150. * g + 29. * b) / 255.;
                });
        });

    Ok(())
}

fn gray_image_crate(image: &Image<u8, 3>) -> Image<u8, 1> {
    let image_data = image.as_slice();
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

    for (width, height) in [(256, 224), (512, 448), (1024, 896)].iter() {
        group.throughput(criterion::Throughput::Elements((*width * *height) as u64));

        let parameter_string = format!("{}x{}", width, height);

        // input image
        let image_data = vec![0u8; width * height * 3];
        let image_size = [*width, *height].into();

        let image = Image::new(image_size, image_data).unwrap();
        let image_f32 = image.clone().cast::<f32>().unwrap();

        // output image
        let gray = Image::from_size_val(image.size(), 0.0).unwrap();

        group.bench_with_input(
            BenchmarkId::new("vanilla_unchecked", &parameter_string),
            &(&image_f32, &gray),
            |b, i| {
                let (src, mut dst) = (i.0, i.1.clone());
                b.iter(|| black_box(gray_vanilla_get_unchecked(src, &mut dst)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("image_crate", &parameter_string),
            &image,
            |b, i| b.iter(|| black_box(gray_image_crate(i))),
        );

        group.bench_with_input(
            BenchmarkId::new("ndarray_zip_par", &parameter_string),
            &(&image_f32, &gray),
            |b, i| {
                let (src, mut dst) = (i.0, i.1.clone());
                b.iter(|| black_box(gray_ndarray_zip_par(src, &mut dst)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("slice_chunks_pixels", &parameter_string),
            &(&image_f32, &gray),
            |b, i| {
                let (src, mut dst) = (i.0, i.1.clone());
                b.iter(|| black_box(gray_slice_chunks_pixels(src, &mut dst)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("slice_chunks_pixels_parallel", &parameter_string),
            &(&image_f32, &gray),
            |b, i| {
                let (src, mut dst) = (i.0, i.1.clone());
                b.iter(|| black_box(gray_slice_chunks_pixels_parallel(src, &mut dst)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("slice_chunks_rows", &parameter_string),
            &(&image_f32, &gray),
            |b, i| {
                let (src, mut dst) = (i.0, i.1.clone());
                b.iter(|| black_box(gray_slice_chunks_rows(src, &mut dst)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("slice_chunks_rows_parallel", &parameter_string),
            &(&image_f32, &gray),
            |b, i| {
                let (src, mut dst) = (i.0, i.1.clone());
                b.iter(|| black_box(gray_from_rgb(src, &mut dst)))
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_grayscale);
criterion_main!(benches);
