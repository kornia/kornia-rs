use criterion::{criterion_group, criterion_main, Criterion};
use rand::Rng;
use std::hint::black_box;
use kornia_imgproc::calibration::{
    distortion::{undistort_points, brown_conrady_distort_normalized, PolynomialDistortion, TermCriteria},
    CameraIntrinsic,
};
use kornia_tensor::{CpuAllocator, Tensor};

#[cfg(feature = "opencv_bench")]
use opencv::{
    calib3d,
    core::Mat,
    prelude::*,
};

fn gen_normalized_points(n: usize) -> Vec<(f64, f64)> {
    let mut rng = rand::rng();
    (0..n)
        .map(|_| {
            (
                rng.random_range(-0.4..0.4),
                rng.random_range(-0.3..0.3),
            )
        })
        .collect()
}

fn distort_and_project(
    pts: &[(f64, f64)],
    intr: &CameraIntrinsic,
    dist: &PolynomialDistortion,
) -> Tensor<f64, 2, CpuAllocator> {
    let mut data = Vec::with_capacity(pts.len() * 2);

    for &(x, y) in pts {
        let (xd, yd) = brown_conrady_distort_normalized(x, y, dist);

        let u = xd * intr.fx + intr.cx;
        let v = yd * intr.fy + intr.cy;

        data.push(u);
        data.push(v);
    }

    Tensor::from_shape_vec([pts.len(), 2], data, CpuAllocator).unwrap()
}

fn bench_rust(c: &mut Criterion) {
    let intr = CameraIntrinsic {
        fx: 800.0,
        fy: 800.0,
        cx: 320.0,
        cy: 240.0,
    };

    let dist = PolynomialDistortion {
        k1: -0.2,
        k2: 0.05,
        k3: 0.0,
        k4: 0.0,
        k5: 0.0,
        k6: 0.0,
        p1: 0.001,
        p2: -0.001,
    };

    let criteria = TermCriteria {
        max_iter: 20,
        eps: 1e-9,
    };

    let pts = gen_normalized_points(10_000);
    let src = distort_and_project(&pts, &intr, &dist);
    let mut dst = Tensor::<f64, 2, _>::zeros([10_000, 2], CpuAllocator);

    c.bench_function("undistort_rust", |b| {
        b.iter(|| {
            undistort_points(
                black_box(&src),
                black_box(&mut dst),
                &intr,
                &dist,
                None,
                None,
                None,
                criteria,
            )
            .unwrap();
        })
    });
}

#[cfg(feature = "opencv_bench")]
fn bench_opencv(c: &mut Criterion) {
    let intr =
        Mat::from_slice_2d(&[[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]).unwrap();

    let dist = Mat::from_slice(&[-0.2, 0.05, 0.001, -0.001, 0.0, 0.0, 0.0, 0.0]).unwrap();

    let pts = gen_normalized_points(10_000);
    let intr_rust = CameraIntrinsic {
        fx: 800.0,
        fy: 800.0,
        cx: 320.0,
        cy: 240.0,
    };
    let dist_rust = PolynomialDistortion {
        k1: -0.2,
        k2: 0.05,
        k3: 0.0,
        k4: 0.0,
        k5: 0.0,
        k6: 0.0,
        p1: 0.001,
        p2: -0.001,
    };

    let src = distort_and_project(&pts, &intr_rust, &dist_rust);

    let src_cv_base = Mat::from_slice(src.as_slice()).unwrap();
    let src_cv = src_cv_base.reshape(2, 10_000).unwrap();

    let mut dst = Mat::default();

    c.bench_function("undistort_opencv", |b| {
        b.iter(|| {
            calib3d::undistort_points(
                black_box(&src_cv),
                black_box(&mut dst),
                &intr,
                &dist,
                &Mat::default(),
                &intr,
            )
            .unwrap();
        })
    });
}

fn bench_all(c: &mut Criterion) {
    bench_rust(c);

    #[cfg(feature = "opencv_bench")]
    bench_opencv(c);
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
