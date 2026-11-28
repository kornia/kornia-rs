//! `optical_flow_pyr_lk::{extract_template, track_template}` against the full-pyramid fixed-point LK
//! and, with `opencv_bench`, OpenCV's `calcOpticalFlowPyrLK` on prebuilt
//! pyramids. One rig camera's flow step: 48 tracks, 11×11, level 0, at most
//! 8 iterations from a start about a pixel off.
//!
//! `RAYON_NUM_THREADS=1 taskset -c 8 cargo bench -j2 -p kornia-imgproc
//! --features opencv_bench --bench bench_lk_window`

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use kornia_image::{Image, ImageSize};
use kornia_imgproc::optical_flow_pyr_lk::{calc_optical_flow_pyr_lk_u8, LkPyramidU8, PyrLKParams};
use kornia_imgproc::optical_flow_pyr_lk::{extract_template, track_template, LkFrame, LkTemplate};

const WIDTH: usize = 640;
const HEIGHT: usize = 480;
const WIN: usize = 11;
const MAX_ITER: usize = 8;
const EPSILON: f32 = 0.01;
const TRACKS: usize = 48;
/// The scene moves by this between the frames; starts are off by `MISS`.
const SHIFT: [f32; 2] = [1.3, -0.7];
const MISS: [f32; 2] = [0.8, -0.5];

/// A smooth texture shifted by `(dx, dy)`.
fn scene(dx: f32, dy: f32) -> Image<u8, 1> {
    let data = (0..WIDTH * HEIGHT)
        .map(|i| {
            let (x, y) = ((i % WIDTH) as f32 - dx, (i / WIDTH) as f32 - dy);
            let v = 128.0
                + 50.0 * (0.31 * x).sin() * (0.27 * y).cos()
                + 35.0 * (0.13 * (x + 2.0 * y)).sin()
                + 20.0 * (0.71 * x - 0.53 * y).cos();
            v.round().clamp(0.0, 255.0) as u8
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

/// `TRACKS` points on an 8×6 grid, 23 px or more from the edges.
fn points() -> Vec<[f32; 2]> {
    (0..6)
        .flat_map(|j| {
            (0..8).map(move |i| {
                [
                    40.3 + i as f32 * (WIDTH as f32 - 80.0) / 7.0,
                    30.7 + j as f32 * (HEIGHT as f32 - 60.0) / 5.0,
                ]
            })
        })
        .collect()
}

fn starts(pts: &[[f32; 2]]) -> Vec<[f32; 2]> {
    pts.iter()
        .map(|p| [p[0] + SHIFT[0] - MISS[0], p[1] + SHIFT[1] - MISS[1]])
        .collect()
}

fn bench_lk_window(c: &mut Criterion) {
    let (prev_img, next_img) = (scene(0.0, 0.0), scene(SHIFT[0], SHIFT[1]));
    let (pts, starts) = (points(), starts(&points()));
    let prev = LkFrame::new(&prev_img, WIN, 0).unwrap();
    let next = LkFrame::new(&next_img, WIN, 1).unwrap();
    let mut templates = vec![LkTemplate::default(); TRACKS];
    for (t, &p) in templates.iter_mut().zip(&pts) {
        assert!(extract_template(&prev, p, 0, WIN, t));
    }

    let mut group = c.benchmark_group("lk_window");
    group.throughput(Throughput::Elements(TRACKS as u64));
    group.bench_function("frame_rebuild_640x480", |b| {
        let mut frame = LkFrame::new(&prev_img, WIN, 0).unwrap();
        let mut id = 0;
        b.iter(|| {
            id += 1;
            frame.rebuild(black_box(&next_img), id).unwrap();
        });
    });
    group.bench_function("build_coarse_level_1", |b| {
        let mut frame = LkFrame::new(&prev_img, WIN, 0).unwrap();
        b.iter(|| {
            frame.rebuild(&prev_img, 0).unwrap();
            frame.build_coarse(black_box(1)).unwrap();
        });
    });
    group.bench_function("templates_48", |b| {
        let mut scratch = vec![LkTemplate::default(); TRACKS];
        b.iter(|| {
            for (t, &p) in scratch.iter_mut().zip(&pts) {
                black_box(extract_template(&prev, black_box(p), 0, WIN, t));
            }
        });
    });
    group.bench_function("forward_48", |b| {
        b.iter(|| {
            for (t, &s) in templates.iter().zip(&starts) {
                black_box(track_template(t, &next, black_box(s), MAX_ITER, EPSILON));
            }
        });
    });
    group.bench_function("templates_and_forward_48", |b| {
        let mut scratch = LkTemplate::default();
        b.iter(|| {
            for (&p, &s) in pts.iter().zip(&starts) {
                extract_template(&prev, p, 0, WIN, &mut scratch);
                black_box(track_template(
                    &scratch,
                    &next,
                    black_box(s),
                    MAX_ITER,
                    EPSILON,
                ));
            }
        });
    });

    // kornia-rs's u8 LK from prebuilt pyramids: whole-image Scharr is not
    // timed, so this is the per-track kernel only.
    let params = PyrLKParams {
        win_size: WIN,
        max_level: 0,
        max_iter: MAX_ITER,
        epsilon: EPSILON,
        use_initial_flow: true,
        ..PyrLKParams::default()
    };
    let p = LkPyramidU8::new(&prev_img, WIN, 0, true).unwrap();
    let q = LkPyramidU8::new(&next_img, WIN, 0, false).unwrap();
    group.bench_function("kornia_u8_prebuilt_48", |b| {
        b.iter(|| {
            black_box(
                calc_optical_flow_pyr_lk_u8(&p, &q, &pts, Some(starts.as_slice()), &params)
                    .unwrap(),
            );
        });
    });

    #[cfg(feature = "opencv_bench")]
    opencv_arm::bench(&mut group, &prev_img, &next_img, &pts, &starts);

    group.finish();
}

#[cfg(feature = "opencv_bench")]
mod opencv_arm {
    use super::*;
    use criterion::measurement::WallTime;
    use criterion::BenchmarkGroup;
    use opencv::{core, prelude::*, video};

    pub(super) fn mat(img: &Image<u8, 1>) -> opencv::Result<core::Mat> {
        let mut mat =
            core::Mat::zeros(img.height() as i32, img.width() as i32, core::CV_8UC1)?.to_mat()?;
        for (i, &v) in img.as_slice().iter().enumerate() {
            *mat.at_2d_mut::<u8>((i / img.width()) as i32, (i % img.width()) as i32)? = v;
        }
        Ok(mat)
    }

    fn pyramid(img: &Image<u8, 1>) -> opencv::Result<core::Vector<core::Mat>> {
        let mut pyr = core::Vector::<core::Mat>::new();
        video::build_optical_flow_pyramid(
            &mat(img)?,
            &mut pyr,
            core::Size::new(WIN as i32, WIN as i32),
            0,
            true,
            core::BORDER_REFLECT_101,
            core::BORDER_CONSTANT,
            true,
        )?;
        Ok(pyr)
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
        prev_img: &Image<u8, 1>,
        next_img: &Image<u8, 1>,
        pts: &[[f32; 2]],
        starts: &[[f32; 2]],
    ) {
        let (prev, next) = (pyramid(prev_img).unwrap(), pyramid(next_img).unwrap());
        let prev_pts = points(pts);
        let term_type = core::TermCriteria_Type::COUNT as i32 + core::TermCriteria_Type::EPS as i32;
        let (mut status, mut err) = (core::Vector::<u8>::new(), core::Vector::<f32>::new());
        group.bench_function("opencv_prebuilt_48", |b| {
            // Allocated once, as the Rust arm's buffer is.
            let mut next_pts = points(starts);
            b.iter(|| {
                // In/out with OPTFLOW_USE_INITIAL_FLOW; refilling the 48 starts is timed too.
                for (i, p) in starts.iter().enumerate() {
                    next_pts.set(i, core::Point2f::new(p[0], p[1])).unwrap();
                }
                let term = core::TermCriteria::new(term_type, MAX_ITER as i32, f64::from(EPSILON))
                    .unwrap();
                video::calc_optical_flow_pyr_lk(
                    &prev,
                    &next,
                    &prev_pts,
                    &mut next_pts,
                    &mut status,
                    &mut err,
                    core::Size::new(WIN as i32, WIN as i32),
                    0,
                    term,
                    video::OPTFLOW_USE_INITIAL_FLOW,
                    1e-4,
                )
                .unwrap();
                black_box(&next_pts);
            });
        });
    }
}

criterion_group!(benches, bench_lk_window);
criterion_main!(benches);
