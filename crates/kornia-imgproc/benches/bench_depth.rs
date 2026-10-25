//! Depth ops: valid-aware median blur and the flying-pixel mask, u16 depth,
//! CPU and (with `--features cuda`) device-resident CUDA.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kornia_image::{Image, ImageSize};
use kornia_imgproc::depth::{flying_pixel_mask, median_blur_depth};

const SIZES: [(usize, usize); 2] = [(640, 480), (1280, 720)];

/// Two planar surfaces with noise and ~10 % holes.
fn depth_u16(w: usize, h: usize) -> Vec<u16> {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    (0..w * h)
        .map(|i| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let r = (state >> 33) as u32;
            let (x, y) = (i % w, i / w);
            if r.is_multiple_of(10) {
                0
            } else {
                let base = if x < w / 2 { 900 } else { 2400 };
                base + (x / 4 + y / 4) as u16 + (r >> 28) as u16
            }
        })
        .collect()
}

fn bench_depth(c: &mut Criterion) {
    let mut group = c.benchmark_group("Depth");
    for (w, h) in SIZES {
        let size = ImageSize {
            width: w,
            height: h,
        };
        let src = Image::<u16, 1>::new(size, depth_u16(w, h)).unwrap();
        let id = format!("{w}x{h}");

        for ksize in [3usize, 5] {
            let mut dst = Image::<u16, 1>::from_size_val(size, 0).unwrap();
            group.bench_function(
                BenchmarkId::new(format!("median_u16_k{ksize}_cpu"), &id),
                |b| b.iter(|| std::hint::black_box(median_blur_depth(&src, &mut dst, ksize))),
            );
        }
        let mut mask = Image::<u8, 1>::from_size_val(size, 0).unwrap();
        group.bench_function(BenchmarkId::new("flying_mask_u16_cpu", &id), |b| {
            b.iter(|| std::hint::black_box(flying_pixel_mask(&src, &mut mask, 30.0, 0.02, 4)))
        });

        #[cfg(feature = "cuda")]
        {
            let ctx = cudarc::driver::CudaContext::new(0).unwrap();
            let stream = ctx.default_stream();
            let d_src = src.to_cuda(&stream).unwrap();
            // Kernel time only: images stay device-resident, the stream is
            // synchronized inside the timed loop so launches are not queued.
            for ksize in [3usize, 5] {
                let mut d_dst = Image::<u16, 1>::zeros_cuda(size, &stream).unwrap();
                group.bench_function(
                    BenchmarkId::new(format!("median_u16_k{ksize}_cuda"), &id),
                    |b| {
                        b.iter(|| {
                            median_blur_depth(&d_src, &mut d_dst, ksize).unwrap();
                            stream.synchronize().unwrap();
                        })
                    },
                );
            }
            let mut d_mask = Image::<u8, 1>::zeros_cuda(size, &stream).unwrap();
            group.bench_function(BenchmarkId::new("flying_mask_u16_cuda", &id), |b| {
                b.iter(|| {
                    flying_pixel_mask(&d_src, &mut d_mask, 30.0, 0.02, 4).unwrap();
                    stream.synchronize().unwrap();
                })
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_depth);
criterion_main!(benches);
