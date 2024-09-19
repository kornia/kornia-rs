use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use kornia_image::Image;
use kornia_imgproc::flip;

use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::{ParallelSlice, ParallelSliceMut},
};

fn par_par_slicecopy(src: &Image<f32, 3>, dst: &mut Image<f32, 3>) {
    dst.as_slice_mut()
        .par_chunks_exact_mut(src.cols() * 3)
        .zip_eq(src.as_slice().par_chunks_exact(src.cols() * 3))
        .for_each(|(dst_row, src_row)| {
            dst_row
                .par_chunks_exact_mut(3)
                .zip_eq(src_row.par_chunks_exact(3).rev())
                .for_each(|(dst_pixel, src_pixel)| {
                    dst_pixel.copy_from_slice(src_pixel);
                })
        });
}

fn par_loop_loop(src: &Image<f32, 3>, dst: &mut Image<f32, 3>) {
    dst.as_slice_mut()
        .par_chunks_exact_mut(src.cols() * 3)
        .zip_eq(src.as_slice().par_chunks_exact(src.cols() * 3))
        .for_each(|(dst_row, src_row)| {
            let n = src.cols();
            for i in 0..n / 2 {
                for c in 0..3 {
                    let (idx_i, idx_j) = (i * 3 + c, (n - 1 - i) * 3 + c);
                    dst_row[idx_i] = src_row[idx_j];
                    dst_row[idx_j] = src_row[idx_i];
                }
            }
        });
}

fn par_loop_slicecopy(src: &Image<f32, 3>, dst: &mut Image<f32, 3>) {
    dst.as_slice_mut()
        .par_chunks_exact_mut(src.cols() * 3)
        .zip_eq(src.as_slice().par_chunks_exact(src.cols() * 3))
        .for_each(|(dst_row, src_row)| {
            let n = src.cols();
            for i in 0..n / 2 {
                let (idx_i, idx_j) = (i * 3, (n - 1 - i) * 3);
                dst_row[idx_i..idx_i + 3].copy_from_slice(&src_row[idx_j..idx_j + 3]);
                dst_row[idx_j..idx_j + 3].copy_from_slice(&src_row[idx_i..idx_i + 3]);
            }
        });
}

fn par_seq_slicecopy(src: &Image<f32, 3>, dst: &mut Image<f32, 3>) {
    dst.as_slice_mut()
        .par_chunks_exact_mut(src.cols() * 3)
        .zip_eq(src.as_slice().par_chunks_exact(src.cols() * 3))
        .for_each(|(dst_row, src_row)| {
            dst_row
                .chunks_exact_mut(3)
                .zip(src_row.chunks_exact(3).rev())
                .for_each(|(dst_pixel, src_pixel)| {
                    dst_pixel.copy_from_slice(src_pixel);
                })
        });
}

fn bench_flip(c: &mut Criterion) {
    let mut group = c.benchmark_group("Flip");

    for (width, height) in [(256, 224), (512, 448), (1024, 896)].iter() {
        group.throughput(criterion::Throughput::Elements((*width * *height) as u64));

        let parameter_string = format!("{}x{}", width, height);

        // input image
        let image_size = [*width, *height].into();
        let image = Image::<u8, 3>::new(image_size, vec![0u8; width * height * 3]).unwrap();
        let image_f32 = image.clone().cast::<f32>().unwrap();

        // output image
        let output = Image::<f32, 3>::from_size_val(image_size, 0.0).unwrap();

        group.bench_with_input(
            BenchmarkId::new("par_par_slicecopy", &parameter_string),
            &(&image_f32, &output),
            |b, i| {
                let (src, mut dst) = (i.0, i.1.clone());
                b.iter(|| {
                    par_par_slicecopy(black_box(src), black_box(&mut dst));
                    black_box(())
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("par_loop_loop", &parameter_string),
            &(&image_f32, &output),
            |b, i| {
                let (src, mut dst) = (i.0, i.1.clone());
                b.iter(|| {
                    par_loop_loop(black_box(src), black_box(&mut dst));
                    black_box(())
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("par_loop_slicecopy", &parameter_string),
            &(&image_f32, &output),
            |b, i| {
                let (src, mut dst) = (i.0, i.1.clone());
                b.iter(|| {
                    par_loop_slicecopy(black_box(src), black_box(&mut dst));
                    black_box(())
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("par_seq_slicecopy", &parameter_string),
            &(&image_f32, &output),
            |b, i| {
                let (src, mut dst) = (i.0, i.1.clone());
                b.iter(|| {
                    par_seq_slicecopy(black_box(src), black_box(&mut dst));
                    black_box(())
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("kornia", &parameter_string),
            &(&image_f32, &output),
            |b, i| {
                let (src, mut dst) = (i.0, i.1.clone());
                b.iter(|| flip::horizontal_flip(black_box(src), black_box(&mut dst)))
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_flip);
criterion_main!(benches);
