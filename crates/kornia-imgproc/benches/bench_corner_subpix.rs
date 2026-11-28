//! `features::corner_sub_pix` and, with `opencv_bench`,
//! OpenCV's `cornerSubPix`. One refill: 48 corners of a 640×480 image,
//! an 11×11 window (`half_win` 5), at most 30 steps to 0.01 px.
//!
//! `RAYON_NUM_THREADS=1 taskset -c 8 cargo bench -j2 -p kornia-imgproc
//! --features opencv_bench --bench bench_corner_subpix`

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use kornia_image::{Image, ImageSize};
use kornia_imgproc::features::corner_sub_pix;

const WIDTH: usize = 640;
const HEIGHT: usize = 480;
const HALF_WIN: usize = 5;
const MAX_ITER: usize = 30;
const EPSILON: f32 = 0.01;
/// Squares of this pitch; their corners are the features.
const PITCH: usize = 40;

/// A checkerboard with soft edges, so corners refine to sub-pixel positions.
fn scene() -> Image<u8, 1> {
    let edge = |v: f32| 1.0 / (1.0 + (-1.4 * v).exp());
    let data = (0..WIDTH * HEIGHT)
        .map(|i| {
            let (x, y) = ((i % WIDTH) as f32 + 0.37, (i / WIDTH) as f32 - 0.21);
            let (u, v) = (
                x % PITCH as f32 - PITCH as f32 / 2.0,
                y % PITCH as f32 - PITCH as f32 / 2.0,
            );
            let (a, b) = (edge(u), edge(v));
            (40.0 + 180.0 * (a * (1.0 - b) + b * (1.0 - a))).round() as u8
        })
        .collect();
    Image::new(
        ImageSize {
            width: WIDTH,
            height: HEIGHT,
        },
        data,
    )
    .unwrap()
}

/// 48 checkerboard corners on an 8×6 grid, each started about a pixel off.
fn corners() -> Vec<[f32; 2]> {
    (1..7)
        .flat_map(|j| {
            (1..9).map(move |i| {
                [
                    (i * PITCH + PITCH / 2) as f32 - 0.8,
                    (j * PITCH + PITCH / 2) as f32 + 0.5,
                ]
            })
        })
        .collect()
}

fn bench_corner_subpix(c: &mut Criterion) {
    let img = scene();
    let start = corners();
    let mut group = c.benchmark_group("corner_subpix");
    group.bench_function("refine_48_win11", |b| {
        let mut pts = start.clone();
        b.iter(|| {
            pts.copy_from_slice(&start);
            corner_sub_pix(&img, black_box(&mut pts), HALF_WIN, MAX_ITER, EPSILON);
            black_box(&pts);
        });
    });

    #[cfg(feature = "opencv_bench")]
    opencv_arm::bench(&mut group, &img, &start);

    group.finish();
}

#[cfg(feature = "opencv_bench")]
mod opencv_arm {
    use super::*;
    use criterion::measurement::WallTime;
    use criterion::BenchmarkGroup;
    use opencv::{core, imgproc, prelude::*};

    fn mat(img: &Image<u8, 1>) -> opencv::Result<core::Mat> {
        let mut mat =
            core::Mat::zeros(img.height() as i32, img.width() as i32, core::CV_8UC1)?.to_mat()?;
        for (i, &v) in img.as_slice().iter().enumerate() {
            *mat.at_2d_mut::<u8>((i / img.width()) as i32, (i % img.width()) as i32)? = v;
        }
        Ok(mat)
    }

    fn points(pts: &[[f32; 2]]) -> core::Vector<core::Point2f> {
        let mut out = core::Vector::<core::Point2f>::new();
        for p in pts {
            out.push(core::Point2f::new(p[0], p[1]));
        }
        out
    }

    pub(super) fn bench(
        group: &mut BenchmarkGroup<'_, WallTime>,
        img: &Image<u8, 1>,
        start: &[[f32; 2]],
    ) {
        let cv_img = mat(img).unwrap();
        let term_type = core::TermCriteria_Type::COUNT as i32 + core::TermCriteria_Type::EPS as i32;
        group.bench_function("opencv_refine_48_win11", |b| {
            // Allocated once, as the Rust arm's buffer is.
            let mut pts = points(start);
            b.iter(|| {
                // In/out; refilling the 48 starts is timed too.
                for (i, p) in start.iter().enumerate() {
                    pts.set(i, core::Point2f::new(p[0], p[1])).unwrap();
                }
                let term = core::TermCriteria::new(term_type, MAX_ITER as i32, f64::from(EPSILON))
                    .unwrap();
                imgproc::corner_sub_pix(
                    &cv_img,
                    &mut pts,
                    core::Size::new(HALF_WIN as i32, HALF_WIN as i32),
                    core::Size::new(-1, -1),
                    term,
                )
                .unwrap();
                black_box(&pts);
            });
        });
    }
}

criterion_group!(benches, bench_corner_subpix);
criterion_main!(benches);
