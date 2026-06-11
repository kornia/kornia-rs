use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use kornia_image::{allocator::CpuAllocator, Image, ImageSize};
#[cfg(feature = "opencv_bench")]
use kornia_imgproc::optical_flow_pyr_lk::{
    build_lk_precomputed, calc_optical_flow_pyr_lk_with_precomputed,
};
use kornia_imgproc::optical_flow_pyr_lk::{calc_optical_flow_pyr_lk, PyrLKParams};
#[cfg(feature = "opencv_bench")]
use std::hint::black_box;

#[cfg(feature = "opencv_bench")]
use opencv::{core, prelude::*, video};

type GrayImage = Image<f32, 1, CpuAllocator>;

fn make_synthetic_pair(
    size: usize,
    dx: f32,
    dy: f32,
    n_points: usize,
) -> (GrayImage, GrayImage, Vec<[f32; 2]>) {
    let r = 10.0;
    let img_size = ImageSize {
        width: size,
        height: size,
    };
    let mut img1 = GrayImage::from_size_val(img_size, 0.0, CpuAllocator).unwrap();
    for y in 0..size {
        for x in 0..size {
            let cx = size as f32 / 2.0;
            let cy = size as f32 / 2.0;
            let dx_ = x as f32 - cx;
            let dy_ = y as f32 - cy;
            if (dx_ * dx_ + dy_ * dy_).sqrt() < r {
                img1.set_pixel(x, y, 0, 1.0).unwrap();
            }
        }
    }

    let mut img2 = GrayImage::from_size_val(img_size, 0.0, CpuAllocator).unwrap();
    for y in 0..size {
        for x in 0..size {
            let cx = size as f32 / 2.0 + dx;
            let cy = size as f32 / 2.0 + dy;
            let dx_ = x as f32 - cx;
            let dy_ = y as f32 - cy;
            if (dx_ * dx_ + dy_ * dy_).sqrt() < r {
                img2.set_pixel(x, y, 0, 1.0).unwrap();
            }
        }
    }

    // Generate grid of points
    let mut pts = Vec::new();
    let step = ((size as f32 * size as f32) / n_points as f32).sqrt();
    let mut y = r + 1.0;
    while y < (size as f32 - r - 1.0) {
        let mut x = r + 1.0;
        while x < (size as f32 - r - 1.0) {
            pts.push([x, y]);
            if pts.len() >= n_points {
                break;
            }
            x += step;
        }
        if pts.len() >= n_points {
            break;
        }
        y += step;
    }

    (img1, img2, pts)
}

fn bench_optical_flow(c: &mut Criterion) {
    let mut group = c.benchmark_group("OpticalFlowPyrLK");

    // Baseline synthetic case.
    {
        let size = 256usize;
        let n_points = 100usize;
        let (img1, img2, pts) = make_synthetic_pair(size, 5.0, -3.0, n_points);
        let params = PyrLKParams::default();
        group.throughput(Throughput::Elements(n_points as u64));
        group.bench_with_input(
            BenchmarkId::new("kornia_cpu/synthetic", format!("{size}x{size}")),
            &pts,
            |b, pts| {
                b.iter(|| {
                    let _ = calc_optical_flow_pyr_lk(&img1, &img2, pts, None, &params).unwrap();
                });
            },
        );
    }

    // Scale over feature count.
    for n_points in [50usize, 100, 200, 500] {
        let size = 256usize;
        let (img1, img2, pts) = make_synthetic_pair(size, 5.0, -3.0, n_points);
        let params = PyrLKParams::default();
        group.throughput(Throughput::Elements(n_points as u64));
        group.bench_with_input(
            BenchmarkId::new("kornia_cpu/points", format!("{size}x{size}/{n_points}")),
            &pts,
            |b, pts| {
                b.iter(|| {
                    let _ = calc_optical_flow_pyr_lk(&img1, &img2, pts, None, &params).unwrap();
                });
            },
        );
    }

    // 512x512 800 points case.
    {
        let size = 512usize;
        let n_points = 800usize;
        let (img1, img2, pts) = make_synthetic_pair(size, 5.0, -3.0, n_points);
        let params = PyrLKParams::default();
        group.throughput(Throughput::Elements(n_points as u64));
        group.bench_with_input(
            BenchmarkId::new("kornia_cpu/points", format!("{size}x{size}/{n_points}")),
            &pts,
            |b, pts| {
                b.iter(|| {
                    let _ = calc_optical_flow_pyr_lk(&img1, &img2, pts, None, &params).unwrap();
                });
            },
        );
    }

    // Scale over pyramid levels.
    for max_level in [0usize, 1, 2, 3] {
        let size = 256usize;
        let n_points = 100usize;
        let (img1, img2, pts) = make_synthetic_pair(size, 8.0, -6.0, n_points);
        let params = PyrLKParams {
            max_level,
            ..PyrLKParams::default()
        };
        group.throughput(Throughput::Elements(n_points as u64));
        group.bench_with_input(
            BenchmarkId::new(
                "kornia_cpu/pyr_levels",
                format!("{size}x{size}/L{max_level}"),
            ),
            &pts,
            |b, pts| {
                b.iter(|| {
                    let _ = calc_optical_flow_pyr_lk(&img1, &img2, pts, None, &params).unwrap();
                });
            },
        );
    }

    #[cfg(feature = "opencv_bench")]
    {
        let size = 512usize;
        let n_points = 800usize;
        let params = PyrLKParams::default();
        let (img1, img2, pts) = make_synthetic_pair(size, 7.0, -5.0, n_points);
        let precomputed = build_lk_precomputed(&img1, &img2, params.max_level).unwrap();
        let cv_prev = image_to_cv_mat(&img1).unwrap();
        let cv_next = image_to_cv_mat(&img2).unwrap();
        let cv_prev_pts = pts_to_cv_vec(&pts);
        let param = format!("{size}x{size}/{n_points}");
        group.throughput(Throughput::Elements(n_points as u64));

        group.bench_with_input(
            BenchmarkId::new("kornia_cpu/opencv_compare", &param),
            &pts,
            |b, pts| {
                b.iter(|| {
                    let _ = black_box(calc_optical_flow_pyr_lk(&img1, &img2, pts, None, &params))
                        .unwrap();
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("kornia_cpu/opencv_compare_precomputed", &param),
            &pts,
            |b, pts| {
                b.iter(|| {
                    let _ = black_box(calc_optical_flow_pyr_lk_with_precomputed(
                        &precomputed,
                        pts,
                        None,
                        &params,
                    ))
                    .unwrap();
                });
            },
        );

        group.bench_function(BenchmarkId::new("opencv/opencv_compare", &param), |b| {
            b.iter(|| {
                let _ = black_box(
                    run_opencv_lk_once(&cv_prev, &cv_next, &cv_prev_pts, &params).unwrap(),
                );
            });
        });
    }

    group.finish();
}

#[cfg(feature = "opencv_bench")]
fn image_to_cv_mat(img: &GrayImage) -> opencv::Result<core::Mat> {
    let mut mat =
        core::Mat::zeros(img.rows() as i32, img.cols() as i32, core::CV_8UC1)?.to_mat()?;
    for y in 0..img.rows() {
        for x in 0..img.cols() {
            let v = (*img.get([y, x, 0]).unwrap()).clamp(0.0, 1.0);
            *mat.at_2d_mut::<u8>(y as i32, x as i32)? = (v * 255.0).round() as u8;
        }
    }
    Ok(mat)
}

#[cfg(feature = "opencv_bench")]
fn pts_to_cv_vec(pts: &[[f32; 2]]) -> core::Vector<core::Point2f> {
    let mut out = core::Vector::<core::Point2f>::new();
    out.reserve(pts.len());
    for p in pts {
        out.push(core::Point2f::new(p[0], p[1]));
    }
    out
}

#[cfg(feature = "opencv_bench")]
fn run_opencv_lk_once(
    prev: &core::Mat,
    next: &core::Mat,
    prev_pts: &core::Vector<core::Point2f>,
    params: &PyrLKParams,
) -> opencv::Result<(
    core::Vector<core::Point2f>,
    core::Vector<u8>,
    core::Vector<f32>,
)> {
    let mut next_pts = core::Vector::<core::Point2f>::new();
    let mut status = core::Vector::<u8>::new();
    let mut err = core::Vector::<f32>::new();

    let term_type = core::TermCriteria_Type::COUNT as i32 + core::TermCriteria_Type::EPS as i32;
    let term = core::TermCriteria::new(term_type, params.max_iter as i32, params.epsilon as f64)?;

    video::calc_optical_flow_pyr_lk(
        prev,
        next,
        prev_pts,
        &mut next_pts,
        &mut status,
        &mut err,
        core::Size::new(params.win_size as i32, params.win_size as i32),
        params.max_level as i32,
        term,
        0,
        params.min_eigen_threshold as f64,
    )?;

    Ok((next_pts, status, err))
}

criterion_group!(benches, bench_optical_flow);
criterion_main!(benches);
