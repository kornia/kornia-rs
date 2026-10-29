//! Label-boundary erosion: u8 and u16 label maps, both connectivities,
//! several radii, CPU and (with `--features cuda`) device-resident CUDA.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kornia_image::{Image, ImageSize};
use kornia_imgproc::connected_components::Connectivity;
use kornia_imgproc::morphology::erode_label_boundaries;

const SIZES: [(usize, usize); 2] = [(640, 480), (1280, 720)];
/// 1, 2, 5: direct window (`t <= 4`); 8: CPU scan, CUDA fused launch;
/// 12: CPU scan, CUDA two-launch path (`radius >= 10`).
const RADII: [usize; 5] = [1, 2, 5, 8, 12];
const CONNECTIVITIES: [(Connectivity, &str); 2] =
    [(Connectivity::Four, "c4"), (Connectivity::Eight, "c8")];

/// A Voronoi-like segmentation: each pixel takes the label of the nearest
/// of a fixed set of pseudo-random sites (~2 sites per 10k pixels).
fn labels_u16(w: usize, h: usize) -> Vec<u16> {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as usize
    };
    let sites: Vec<(usize, usize)> = (0..(w * h / 5000).max(2))
        .map(|_| (next() % w, next() % h))
        .collect();
    (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            let mut best = (usize::MAX, 0u16);
            for (k, &(sx, sy)) in sites.iter().enumerate() {
                let d = x.abs_diff(sx).pow(2) + y.abs_diff(sy).pow(2);
                if d < best.0 {
                    best = (d, 1 + k as u16);
                }
            }
            best.1
        })
        .collect()
}

fn bench_label_boundaries(c: &mut Criterion) {
    let mut group = c.benchmark_group("LabelBoundaries");
    for (w, h) in SIZES {
        let size = ImageSize {
            width: w,
            height: h,
        };
        let labels = labels_u16(w, h);
        let src_u8 = Image::<u8, 1>::new(size, labels.iter().map(|&v| v as u8).collect()).unwrap();
        let src_u16 = Image::<u16, 1>::new(size, labels).unwrap();
        let mut dst_u8 = Image::<u8, 1>::from_size_val(size, 0).unwrap();
        let mut dst_u16 = Image::<u16, 1>::from_size_val(size, 0).unwrap();
        let id = format!("{w}x{h}");

        for (conn, tag) in CONNECTIVITIES {
            for radius in RADII {
                group.bench_function(
                    BenchmarkId::new(format!("u8_{tag}_r{radius}_cpu"), &id),
                    |b| {
                        b.iter(|| {
                            std::hint::black_box(erode_label_boundaries(
                                &src_u8,
                                &mut dst_u8,
                                radius,
                                conn,
                                0,
                            ))
                        })
                    },
                );
                group.bench_function(
                    BenchmarkId::new(format!("u16_{tag}_r{radius}_cpu"), &id),
                    |b| {
                        b.iter(|| {
                            std::hint::black_box(erode_label_boundaries(
                                &src_u16,
                                &mut dst_u16,
                                radius,
                                conn,
                                0,
                            ))
                        })
                    },
                );
            }
        }

        #[cfg(feature = "cuda")]
        {
            let ctx = cudarc::driver::CudaContext::new(0).unwrap();
            // Owned (non-legacy) stream: the context default stream
            // serializes globally and inflates timings.
            let stream = ctx.new_stream().unwrap();
            let d_src_u8 = src_u8.to_cuda(&stream).unwrap();
            let d_src_u16 = src_u16.to_cuda(&stream).unwrap();
            let mut d_dst_u8 = Image::<u8, 1>::zeros_cuda(size, &stream).unwrap();
            let mut d_dst_u16 = Image::<u16, 1>::zeros_cuda(size, &stream).unwrap();
            // Images stay device-resident. `_cuda` synchronizes after every
            // call (latency, including the launch and sync round trip);
            // `_cuda_queued` queues `iters` calls and synchronizes once
            // (device throughput).
            let timed = |b: &mut criterion::Bencher, queued: bool, op: &mut dyn FnMut()| {
                b.iter_custom(|iters| {
                    stream.synchronize().unwrap();
                    let t = std::time::Instant::now();
                    for _ in 0..iters {
                        op();
                        if !queued {
                            stream.synchronize().unwrap();
                        }
                    }
                    stream.synchronize().unwrap();
                    t.elapsed()
                })
            };
            for (conn, tag) in CONNECTIVITIES {
                for radius in RADII {
                    for (queued, mode) in [(false, "cuda"), (true, "cuda_queued")] {
                        group.bench_function(
                            BenchmarkId::new(format!("u8_{tag}_r{radius}_{mode}"), &id),
                            |b| {
                                timed(b, queued, &mut || {
                                    erode_label_boundaries(
                                        &d_src_u8,
                                        &mut d_dst_u8,
                                        radius,
                                        conn,
                                        0,
                                    )
                                    .unwrap()
                                })
                            },
                        );
                        group.bench_function(
                            BenchmarkId::new(format!("u16_{tag}_r{radius}_{mode}"), &id),
                            |b| {
                                timed(b, queued, &mut || {
                                    erode_label_boundaries(
                                        &d_src_u16,
                                        &mut d_dst_u16,
                                        radius,
                                        conn,
                                        0,
                                    )
                                    .unwrap()
                                })
                            },
                        );
                    }
                }
            }
        }
    }
    group.finish();
}

criterion_group!(benches, bench_label_boundaries);
criterion_main!(benches);
