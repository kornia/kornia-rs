//! Depth unprojection and registration on synthetic u16 / f32 depth images
//! (640x480 and 1280x720) on the CPU.
//!
//! `cargo bench -p kornia-3d --bench bench_depth`
//! Baselines on the same inputs: `benches/bench_depth_baselines.py`.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kornia_3d::camera::PinholeCamera;
use kornia_3d::depth::DepthImage;
use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, Vec3F64};
use kornia_image::{Image, ImageSize};

const SIZES: [(usize, usize); 2] = [(640, 480), (1280, 720)];
/// `meter` of millimetre depth.
const MM: f32 = 1000.0;

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
        let src_u16 = DepthImage::new(Image::<u16, 1>::new(size, mm.clone()).unwrap(), MM).unwrap();
        let metres = mm.iter().map(|&d| f32::from(d) * (1.0 / MM)).collect();
        let src_f32 = DepthImage::new(Image::<f32, 1>::new(size, metres).unwrap(), 1.0).unwrap();
        let pose = dst_from_src();

        let mut points = Vec::with_capacity(w * h);
        let mut index = Vec::with_capacity(w * h);
        group.bench_function(BenchmarkId::new("unproject_u16_cpu", &id), |b| {
            b.iter(|| {
                src_u16
                    .to_pointcloud_into(&cam, None, &mut points, None)
                    .unwrap();
                std::hint::black_box(points.len())
            })
        });
        group.bench_function(BenchmarkId::new("unproject_u16_index_cpu", &id), |b| {
            b.iter(|| {
                src_u16
                    .to_pointcloud_into(&cam, None, &mut points, Some(&mut index))
                    .unwrap();
                std::hint::black_box(points.len())
            })
        });
        group.bench_function(BenchmarkId::new("unproject_f32_cpu", &id), |b| {
            b.iter(|| {
                src_f32
                    .to_pointcloud_into(&cam, None, &mut points, None)
                    .unwrap();
                std::hint::black_box(points.len())
            })
        });

        let mut dst_u16 = DepthImage::<u16>::from_size_val(size, 0, MM).unwrap();
        let mut dst_f32 = DepthImage::<f32>::from_size_val(size, 0.0, 1.0).unwrap();
        for radius in [0usize, 1] {
            group.bench_function(
                BenchmarkId::new(format!("register_u16_r{radius}_cpu"), &id),
                |b| {
                    b.iter(|| {
                        src_u16
                            .register_into(&cam, &pose, &cam, radius, &mut dst_u16)
                            .unwrap()
                    })
                },
            );
            group.bench_function(
                BenchmarkId::new(format!("register_f32_r{radius}_cpu"), &id),
                |b| {
                    b.iter(|| {
                        src_f32
                            .register_into(&cam, &pose, &cam, radius, &mut dst_f32)
                            .unwrap()
                    })
                },
            );
        }

        let roll = dst_from_src_roll();
        group.bench_function(BenchmarkId::new("register_u16_r0_roll_cpu", &id), |b| {
            b.iter(|| {
                src_u16
                    .register_into(&cam, &roll, &cam, 0, &mut dst_u16)
                    .unwrap()
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_depth);
criterion_main!(benches);
