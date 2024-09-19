use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use kornia_image::{Image, ImageSize};
use kornia_imgproc::{interpolation::InterpolationMode, resize};

fn resize_image_crate(image: Image<u8, 3>, new_size: ImageSize) -> Image<u8, 3> {
    let image_data = image.as_slice();
    let rgb = image::RgbImage::from_raw(
        image.size().width as u32,
        image.size().height as u32,
        image_data.to_vec(),
    )
    .unwrap();
    let image_crate = image::DynamicImage::ImageRgb8(rgb);

    let image_resized = image_crate.resize_exact(
        new_size.width as u32,
        new_size.height as u32,
        image::imageops::FilterType::Nearest,
    );
    let data = image_resized.into_rgb8().into_raw();
    Image::new(new_size, data).unwrap()
}

fn resize_ndarray_zip(src: &Image<f32, 3>, dst: &mut Image<f32, 3>) {
    // create a grid of x and y coordinates for the output image
    // and interpolate the values from the input image.
    let x = ndarray::Array::linspace(0., (src.width() - 1) as f32, dst.width())
        .insert_axis(ndarray::Axis(0));
    let y = ndarray::Array::linspace(0., (src.height() - 1) as f32, dst.height())
        .insert_axis(ndarray::Axis(0));

    // create the meshgrid of x and y coordinates, arranged in a 2D grid of shape (height, width)
    let nx = x.len_of(ndarray::Axis(1));
    let ny = y.len_of(ndarray::Axis(1));

    // broadcast the x and y coordinates to create a 2D grid, and then transpose the y coordinates
    // to create the meshgrid of x and y coordinates of shape (height, width)
    let xx = x.broadcast((ny, nx)).unwrap().to_owned();
    let yy = y.broadcast((nx, ny)).unwrap().t().to_owned();

    // TODO: benchmark this
    // stack the x and y coordinates into a single array of shape (height, width, 2)
    let xy = ndarray::stack![ndarray::Axis(2), xx, yy];

    // iterate over the output image and interpolate the pixel values

    let mut dst_data = unsafe {
        ndarray::ArrayViewMut3::from_shape_ptr(
            (dst.height(), dst.width(), dst.num_channels()),
            dst.as_mut_ptr(),
        )
    };

    ndarray::Zip::from(xy.rows())
        .and(dst_data.rows_mut())
        .par_for_each(|uv, mut out| {
            assert_eq!(uv.len(), 2);
            let (u, v) = (uv[0], uv[1]);

            // compute the pixel values for each channel
            let pixels = (0..dst.num_channels()).map(|k| {
                kornia_imgproc::interpolation::interpolate_pixel(
                    src,
                    u,
                    v,
                    k,
                    InterpolationMode::Nearest,
                )
            });

            // write the pixel values to the output image
            for (k, pixel) in pixels.enumerate() {
                out[k] = pixel;
            }
        });
}

fn bench_resize(c: &mut Criterion) {
    let mut group = c.benchmark_group("Resize");

    for (width, height) in [(256, 224), (512, 448), (1024, 896)].iter() {
        group.throughput(criterion::Throughput::Elements((*width * *height) as u64));

        let parameter_string = format!("{}x{}", width, height);

        // input image
        let image_size = [*width, *height].into();
        let image = Image::<u8, 3>::new(image_size, vec![0u8; width * height * 3]).unwrap();
        let image_f32 = image.clone().cast::<f32>().unwrap();

        // output image
        let new_size = ImageSize {
            width: width / 2,
            height: height / 2,
        };

        let out_f32 = Image::<f32, 3>::from_size_val(new_size, 0.0).unwrap();
        let out_u8 = Image::<u8, 3>::from_size_val(new_size, 0).unwrap();

        group.bench_with_input(
            BenchmarkId::new("image_rs", &parameter_string),
            &image,
            |b, i| b.iter(|| resize_image_crate(black_box(i.clone()), black_box(new_size))),
        );

        group.bench_with_input(
            BenchmarkId::new("ndarray_zip", &parameter_string),
            &(&image_f32, &out_f32),
            |b, i| {
                let (src, mut dst) = (i.0.clone(), i.1.clone());
                b.iter(|| resize_ndarray_zip(black_box(&src), black_box(&mut dst)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("kornia_par", &parameter_string),
            &(&image_f32, &out_f32),
            |b, i| {
                let (src, mut dst) = (i.0, i.1.clone());
                b.iter(|| {
                    resize::resize_native(
                        black_box(src),
                        black_box(&mut dst),
                        black_box(InterpolationMode::Nearest),
                    )
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("fast_resize_lib", &parameter_string),
            &(image, out_u8),
            |b, i| {
                let (src, mut dst) = (i.0.clone(), i.1.clone());
                b.iter(|| {
                    resize::resize_fast(
                        black_box(&src),
                        black_box(&mut dst),
                        black_box(InterpolationMode::Nearest),
                    )
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_resize);
criterion_main!(benches);
