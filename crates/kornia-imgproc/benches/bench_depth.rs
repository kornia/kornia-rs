//! Depth ops: valid-aware median blur and the flying-pixel mask, u16 and
//! f32, CPU and (with `--features cuda`) device-resident CUDA.

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

/// The u16 fixture in metres, holes as NaN.
fn depth_f32(depth: &[u16]) -> Vec<f32> {
    depth
        .iter()
        .map(|&d| {
            if d == 0 {
                f32::NAN
            } else {
                f32::from(d) * 1e-3
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
        let depth = depth_u16(w, h);
        let src_f32 = Image::<f32, 1>::new(size, depth_f32(&depth)).unwrap();
        let src = Image::<u16, 1>::new(size, depth).unwrap();
        let id = format!("{w}x{h}");

        for ksize in [3usize, 5] {
            let mut dst = Image::<u16, 1>::from_size_val(size, 0).unwrap();
            group.bench_function(
                BenchmarkId::new(format!("median_u16_k{ksize}_cpu"), &id),
                |b| b.iter(|| std::hint::black_box(median_blur_depth(&src, &mut dst, ksize))),
            );
            let mut dst_f32 = Image::<f32, 1>::from_size_val(size, 0.0).unwrap();
            group.bench_function(
                BenchmarkId::new(format!("median_f32_k{ksize}_cpu"), &id),
                |b| {
                    b.iter(|| {
                        std::hint::black_box(median_blur_depth(&src_f32, &mut dst_f32, ksize))
                    })
                },
            );
        }
        // Hole-free input: the workload `cv2.medianBlur` sees (it has no
        // invalid marker), so the two compare like for like.
        let dense =
            Image::<u16, 1>::new(size, src.as_slice().iter().map(|&d| d.max(1)).collect()).unwrap();
        let dense_f32 = Image::<f32, 1>::new(size, depth_f32(dense.as_slice())).unwrap();
        for ksize in [3usize, 5] {
            let mut dst = Image::<u16, 1>::from_size_val(size, 0).unwrap();
            group.bench_function(
                BenchmarkId::new(format!("median_u16_k{ksize}_dense_cpu"), &id),
                |b| b.iter(|| std::hint::black_box(median_blur_depth(&dense, &mut dst, ksize))),
            );
            let mut dst_f32 = Image::<f32, 1>::from_size_val(size, 0.0).unwrap();
            group.bench_function(
                BenchmarkId::new(format!("median_f32_k{ksize}_dense_cpu"), &id),
                |b| {
                    b.iter(|| {
                        std::hint::black_box(median_blur_depth(&dense_f32, &mut dst_f32, ksize))
                    })
                },
            );
        }
        let mut mask = Image::<u8, 1>::from_size_val(size, 0).unwrap();
        group.bench_function(BenchmarkId::new("flying_mask_u16_cpu", &id), |b| {
            b.iter(|| std::hint::black_box(flying_pixel_mask(&src, &mut mask, 30.0, 0.02, 4)))
        });
        group.bench_function(BenchmarkId::new("flying_mask_f32_cpu", &id), |b| {
            b.iter(|| std::hint::black_box(flying_pixel_mask(&src_f32, &mut mask, 0.025, 0.02, 4)))
        });

        #[cfg(feature = "cuda")]
        {
            let ctx = cudarc::driver::CudaContext::new(0).unwrap();
            let stream = ctx.default_stream();
            // Device throughput: images stay device-resident, `iters`
            // launches are queued and the stream is synchronized once, so
            // per-launch host overhead and the sync round trip are not
            // counted per op.
            let queued = |b: &mut criterion::Bencher, op: &mut dyn FnMut()| {
                b.iter_custom(|iters| {
                    stream.synchronize().unwrap();
                    let t = std::time::Instant::now();
                    for _ in 0..iters {
                        op();
                    }
                    stream.synchronize().unwrap();
                    t.elapsed()
                })
            };
            let inputs = [
                (
                    "",
                    src.to_cuda(&stream).unwrap(),
                    src_f32.to_cuda(&stream).unwrap(),
                ),
                (
                    "_dense",
                    dense.to_cuda(&stream).unwrap(),
                    dense_f32.to_cuda(&stream).unwrap(),
                ),
            ];
            for (tag, d_src, d_src_f32) in &inputs {
                for ksize in [3usize, 5] {
                    let mut d_dst = Image::<u16, 1>::zeros_cuda(size, &stream).unwrap();
                    group.bench_function(
                        BenchmarkId::new(format!("median_u16_k{ksize}{tag}_cuda"), &id),
                        |b| {
                            queued(b, &mut || {
                                median_blur_depth(d_src, &mut d_dst, ksize).unwrap()
                            })
                        },
                    );
                    let mut d_dst_f32 = Image::<f32, 1>::zeros_cuda(size, &stream).unwrap();
                    group.bench_function(
                        BenchmarkId::new(format!("median_f32_k{ksize}{tag}_cuda"), &id),
                        |b| {
                            queued(b, &mut || {
                                median_blur_depth(d_src_f32, &mut d_dst_f32, ksize).unwrap()
                            })
                        },
                    );
                }
            }
            let (_, d_src, d_src_f32) = &inputs[0];
            let mut d_mask = Image::<u8, 1>::zeros_cuda(size, &stream).unwrap();
            group.bench_function(BenchmarkId::new("flying_mask_u16_cuda", &id), |b| {
                queued(b, &mut || {
                    flying_pixel_mask(d_src, &mut d_mask, 30.0, 0.02, 4).unwrap()
                })
            });
            group.bench_function(BenchmarkId::new("flying_mask_f32_cuda", &id), |b| {
                queued(b, &mut || {
                    flying_pixel_mask(d_src_f32, &mut d_mask, 0.025, 0.02, 4).unwrap()
                })
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_depth);
criterion_main!(benches);
