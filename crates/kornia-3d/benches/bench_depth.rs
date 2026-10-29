//! Depth unprojection on synthetic u16 / f32 depth images
//! (640x480 and 1280x720) on the CPU.
//!
//! `cargo bench -p kornia-3d --bench bench_depth`
//! Baselines on the same inputs: `benches/bench_depth_baselines.py`.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kornia_3d::camera::PinholeCamera;
use kornia_3d::depth::unproject_depth_into;
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
    }
    group.finish();
}

criterion_group!(benches, bench_depth);
criterion_main!(benches);
