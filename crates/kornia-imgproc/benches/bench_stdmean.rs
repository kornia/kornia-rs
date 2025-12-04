use criterion::{criterion_group, criterion_main, Criterion, black_box};
use kornia_image::{
    allocator::{CpuAllocator, ImageAllocator},
    Image, ImageSize,
};
use rayon::prelude::*;

//Sequential version 
pub fn std_mean_original<A: ImageAllocator>(image: &Image<u8, 3, A>) -> ([f64; 3], [f64; 3]) {
    let (sum, sq_sum) = image.as_slice().chunks_exact(3).fold(
        ([0f64; 3], [0f64; 3]),
        |(mut sum, mut sq_sum), pixel| {
            sum.iter_mut()
                .zip(pixel.iter())
                .for_each(|(s, &p)| *s += p as f64);
            sq_sum
                .iter_mut()
                .zip(pixel.iter())
                .for_each(|(s, &p)| *s += (p as f64).powi(2));
            (sum, sq_sum)
        },
    );

    let n = (image.width() * image.height()) as f64;
    let mean = [sum[0] / n, sum[1] / n, sum[2] / n];

    let variance = [
        (sq_sum[0] / n - mean[0].powi(2)).sqrt(),
        (sq_sum[1] / n - mean[1].powi(2)).sqrt(),
        (sq_sum[2] / n - mean[2].powi(2)).sqrt(),
    ];

    (variance, mean)
}

//Parallel version
pub fn std_mean_parallel<A: ImageAllocator>(image: &Image<u8, 3, A>) -> ([f64; 3], [f64; 3]) {
    let (sum, sq_sum) = image
        .as_slice()
        .par_chunks_exact(3)
        .fold(
            || ([0f64; 3], [0f64; 3]),
            |(mut sum, mut sq_sum), pixel| {
                sum[0] += pixel[0] as f64;
                sum[1] += pixel[1] as f64;
                sum[2] += pixel[2] as f64;

                sq_sum[0] += (pixel[0] as f64).powi(2);
                sq_sum[1] += (pixel[1] as f64).powi(2);
                sq_sum[2] += (pixel[2] as f64).powi(2);

                (sum, sq_sum)
            },
        )
        .reduce(
            || ([0f64; 3], [0f64; 3]),
            |(sa, ssa), (sb, ssb)| {
                (
                    [sa[0] + sb[0], sa[1] + sb[1], sa[2] + sb[2]],
                    [ssa[0] + ssb[0], ssa[1] + ssb[1], ssa[2] + ssb[2]],
                )
            },
        );

    let n = (image.width() * image.height()) as f64;
    let mean = [sum[0] / n, sum[1] / n, sum[2] / n];

    let std = [
        (sq_sum[0] / n - mean[0] * mean[0]).sqrt(),
        (sq_sum[1] / n - mean[1] * mean[1]).sqrt(),
        (sq_sum[2] / n - mean[2] * mean[2]).sqrt(),
    ];

    (std, mean)
}


// BENCHMARK HELPER

fn bench_size(c: &mut Criterion, width: usize, height: usize) {
    let mut data = vec![0u8; width * height * 3];
    for (i, px) in data.iter_mut().enumerate() {
        *px = (i % 255) as u8;
    }

    let image: Image<u8, 3, _> =
        Image::new(ImageSize { width, height }, data, CpuAllocator).unwrap();

    // Sequential
    let name_seq = format!("std_mean_seq_{}x{}", width, height);
    c.bench_function(&name_seq, |b| {
        b.iter(|| black_box(std_mean_original(&image)))
    });

    // Parallel
    let name_par = format!("std_mean_par_{}x{}", width, height);
    c.bench_function(&name_par, |b| {
        b.iter(|| black_box(std_mean_parallel(&image)))
    });
}


// BENCHMARK ENTRY POINT

fn criterion_benchmark(c: &mut Criterion) {
    bench_size(c, 128, 128);
    bench_size(c, 256, 256);
    bench_size(c, 512, 512);
    bench_size(c, 1024, 1024);
    bench_size(c, 4096, 4096);
    
    
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
