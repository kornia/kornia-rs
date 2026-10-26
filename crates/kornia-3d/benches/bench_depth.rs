//! Depth unprojection and registration on synthetic u16 / f32 depth images
//! (640x480 and 1280x720): CPU, and with `--features cuda` the device twins
//! on device-resident buffers, synchronized per call (latency) and queued
//! (`_queued`, throughput).
//!
//! `cargo bench -p kornia-3d --bench bench_depth [--features cuda]`
//! Baselines on the same inputs: `benches/bench_depth_baselines.py`.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kornia_3d::camera::PinholeCamera;
use kornia_3d::depth::{register_depth, unproject_depth_into};
use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, Vec3F64};
use kornia_image::{Image, ImageSize};

const SIZES: [(usize, usize); 2] = [(640, 480), (1280, 720)];
const MM: f32 = 0.001;

fn camera(w: usize, h: usize) -> PinholeCamera {
    let f = 0.9 * w as f64;
    PinholeCamera {
        fx: f,
        fy: f,
        cx: 0.5 * w as f64 - 0.5,
        cy: 0.5 * h as f64 - 0.5,
        k1: 0.0,
        k2: 0.0,
        p1: 0.0,
        p2: 0.0,
    }
}

/// A tilted plane with a box in front and ~5% missing samples, in mm.
/// The same generator is used by the Python baselines.
fn depth_u16(w: usize, h: usize) -> Vec<u16> {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            if state.is_multiple_of(20) {
                return 0;
            }
            let plane = 1500 + (x * 1000 / w) as u16 + (y * 500 / h) as u16;
            let in_box = x > w / 3 && x < w / 2 && y > h / 3 && y < 2 * h / 3;
            if in_box {
                800
            } else {
                plane
            }
        })
        .collect()
}

/// A small rotation and a 5 cm baseline between the two cameras.
fn dst_from_src() -> Pose3d {
    let a = 0.02f64;
    let r =
        Mat3F64::from_cols_array(&[a.cos(), 0.0, -a.sin(), 0.0, 1.0, 0.0, a.sin(), 0.0, a.cos()]);
    Pose3d::new(r, Vec3F64::new(0.05, 0.0, 0.0))
}

/// A 3 degree roll about the optical axis: rows of the source land on
/// slanted destination rows, the hard case for the banded CPU scatter.
fn dst_from_src_roll() -> Pose3d {
    let a = 0.05f64;
    let r =
        Mat3F64::from_cols_array(&[a.cos(), a.sin(), 0.0, -a.sin(), a.cos(), 0.0, 0.0, 0.0, 1.0]);
    Pose3d::new(r, Vec3F64::new(0.05, 0.0, 0.0))
}

fn bench_depth(c: &mut Criterion) {
    let mut group = c.benchmark_group("Depth");
    for (w, h) in SIZES {
        let size = ImageSize {
            width: w,
            height: h,
        };
        let id = format!("{w}x{h}");
        let cam = camera(w, h);
        let mm = depth_u16(w, h);
        let src_u16 = Image::<u16, 1>::new(size, mm.clone()).unwrap();
        let src_f32 =
            Image::<f32, 1>::new(size, mm.iter().map(|&d| f32::from(d) * MM).collect()).unwrap();
        let pose = dst_from_src();

        let mut points = Vec::with_capacity(w * h);
        let mut index = Vec::with_capacity(w * h);
        group.bench_function(BenchmarkId::new("unproject_u16_cpu", &id), |b| {
            b.iter(|| {
                unproject_depth_into(&src_u16, MM, &cam, None, &mut points, None).unwrap();
                std::hint::black_box(points.len())
            })
        });
        group.bench_function(BenchmarkId::new("unproject_u16_index_cpu", &id), |b| {
            b.iter(|| {
                unproject_depth_into(&src_u16, MM, &cam, None, &mut points, Some(&mut index))
                    .unwrap();
                std::hint::black_box(points.len())
            })
        });
        group.bench_function(BenchmarkId::new("unproject_f32_cpu", &id), |b| {
            b.iter(|| {
                unproject_depth_into(&src_f32, 1.0, &cam, None, &mut points, None).unwrap();
                std::hint::black_box(points.len())
            })
        });

        let mut dst_u16 = Image::<u16, 1>::from_size_val(size, 0).unwrap();
        let mut dst_f32 = Image::<f32, 1>::from_size_val(size, 0.0).unwrap();
        for radius in [0usize, 1] {
            group.bench_function(
                BenchmarkId::new(format!("register_u16_r{radius}_cpu"), &id),
                |b| {
                    b.iter(|| {
                        register_depth(&src_u16, MM, &cam, &pose, &cam, radius, &mut dst_u16)
                            .unwrap()
                    })
                },
            );
            group.bench_function(
                BenchmarkId::new(format!("register_f32_r{radius}_cpu"), &id),
                |b| {
                    b.iter(|| {
                        register_depth(&src_f32, 1.0, &cam, &pose, &cam, radius, &mut dst_f32)
                            .unwrap()
                    })
                },
            );
        }

        let roll = dst_from_src_roll();
        group.bench_function(BenchmarkId::new("register_u16_r0_roll_cpu", &id), |b| {
            b.iter(|| register_depth(&src_u16, MM, &cam, &roll, &cam, 0, &mut dst_u16).unwrap())
        });

        #[cfg(feature = "cuda")]
        {
            use cudarc::driver::CudaContext;
            use kornia_3d::depth::CudaDepthProjector;

            let ctx = CudaContext::new(0).unwrap();
            let st = ctx.default_stream();
            let mut proj = CudaDepthProjector::new(&st).unwrap();
            let src_u16_dev = src_u16.to_cuda(&st).unwrap();
            let src_f32_dev = src_f32.to_cuda(&st).unwrap();
            let mut cloud = proj.alloc_point_cloud(w * h, false).unwrap();
            let mut cloud_ix = proj.alloc_point_cloud(w * h, true).unwrap();
            let mut dst_u16_dev = Image::<u16, 1>::zeros_cuda(size, &st).unwrap();
            let mut dst_f32_dev = Image::<f32, 1>::zeros_cuda(size, &st).unwrap();

            // Device throughput: `iters` calls are queued and the stream is
            // synchronized once, so the per-call sync round trip (and the
            // GPU clock dropping between synced calls) is not counted.
            let queued = |b: &mut criterion::Bencher, op: &mut dyn FnMut()| {
                b.iter_custom(|iters| {
                    st.synchronize().unwrap();
                    let t = std::time::Instant::now();
                    for _ in 0..iters {
                        op();
                    }
                    st.synchronize().unwrap();
                    t.elapsed()
                })
            };
            group.bench_function(BenchmarkId::new("unproject_u16_cuda_queued", &id), |b| {
                queued(b, &mut || {
                    proj.unproject_depth(&src_u16_dev, MM, &cam, None, &mut cloud)
                        .unwrap()
                })
            });
            for radius in [0usize, 1] {
                group.bench_function(
                    BenchmarkId::new(format!("register_u16_r{radius}_cuda_queued"), &id),
                    |b| {
                        queued(b, &mut || {
                            proj.register_depth(
                                &src_u16_dev,
                                MM,
                                &cam,
                                &pose,
                                &cam,
                                radius,
                                &mut dst_u16_dev,
                            )
                            .unwrap()
                        })
                    },
                );
            }
            group.bench_function(BenchmarkId::new("unproject_u16_cuda", &id), |b| {
                b.iter(|| {
                    proj.unproject_depth(&src_u16_dev, MM, &cam, None, &mut cloud)
                        .unwrap();
                    st.synchronize().unwrap();
                })
            });
            group.bench_function(BenchmarkId::new("unproject_u16_index_cuda", &id), |b| {
                b.iter(|| {
                    proj.unproject_depth(&src_u16_dev, MM, &cam, None, &mut cloud_ix)
                        .unwrap();
                    st.synchronize().unwrap();
                })
            });
            group.bench_function(BenchmarkId::new("unproject_f32_cuda", &id), |b| {
                b.iter(|| {
                    proj.unproject_depth(&src_f32_dev, 1.0, &cam, None, &mut cloud)
                        .unwrap();
                    st.synchronize().unwrap();
                })
            });
            for radius in [0usize, 1] {
                group.bench_function(
                    BenchmarkId::new(format!("register_u16_r{radius}_cuda"), &id),
                    |b| {
                        b.iter(|| {
                            proj.register_depth(
                                &src_u16_dev,
                                MM,
                                &cam,
                                &pose,
                                &cam,
                                radius,
                                &mut dst_u16_dev,
                            )
                            .unwrap();
                            st.synchronize().unwrap();
                        })
                    },
                );
                group.bench_function(
                    BenchmarkId::new(format!("register_f32_r{radius}_cuda"), &id),
                    |b| {
                        b.iter(|| {
                            proj.register_depth(
                                &src_f32_dev,
                                1.0,
                                &cam,
                                &pose,
                                &cam,
                                radius,
                                &mut dst_f32_dev,
                            )
                            .unwrap();
                            st.synchronize().unwrap();
                        })
                    },
                );
            }
        }
    }
    group.finish();
}

criterion_group!(benches, bench_depth);
criterion_main!(benches);
