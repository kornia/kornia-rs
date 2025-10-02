use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use kornia_pnp as kpnp;
use rand::seq::SliceRandom;
use rand::{rngs::StdRng, Rng, SeedableRng};

type PnpDataset = (Vec<[f32; 3]>, Vec<[f32; 2]>, [[f32; 3]; 3]);
const NUM_SEEDS: usize = 1;

// Removed unseeded generator; use seeded variant for reproducibility across benches

fn generate_cube_dataset_with_seed(num_points: usize, noise_px: f32, seed: u64) -> PnpDataset {
    // Camera intrinsics, assumes no distortion
    let k = [[800.0, 0.0, 640.0], [0.0, 800.0, 480.0], [0.0, 0.0, 1.0]];

    // Simple cube-like distribution in front of the camera
    let mut world = Vec::with_capacity(num_points);
    let mut rng = StdRng::seed_from_u64(seed);
    for _ in 0..num_points {
        // points in a 1m cube around z in [3,6]
        world.push([
            rng.random_range(-0.5..0.5),
            rng.random_range(-0.5..0.5),
            rng.random_range(3.0..6.0),
        ]);
    }

    // Ground-truth pose (mild rotation/translation)
    let r = [
        [0.96, -0.10, 0.26],
        [0.12, 0.99, -0.04],
        [-0.25, 0.07, 0.97],
    ];
    let t = [0.2, -0.1, 0.3];

    let mut image = Vec::with_capacity(num_points);
    for p in &world {
        let xc = r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2] + t[0];
        let yc = r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2] + t[1];
        let zc = r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2] + t[2];
        let u = k[0][0] * xc / zc + k[0][2] + rng.random_range(-noise_px..noise_px);
        let v = k[1][1] * yc / zc + k[1][2] + rng.random_range(-noise_px..noise_px);
        image.push([u, v]);
    }

    (world, image, k)
}

fn inject_outliers_random(image: &mut [[f32; 2]], fraction: f32, seed: u64) {
    let num_out = (fraction.clamp(0.0, 1.0) * image.len() as f32) as usize;
    if num_out == 0 {
        return;
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let mut idxs: Vec<usize> = (0..image.len()).collect();
    idxs.shuffle(&mut rng);
    for &i in idxs.iter().take(num_out) {
        let angle = rng.random_range(0.0..(2.0 * std::f32::consts::PI));
        let radius = rng.random_range(300.0..800.0);
        image[i][0] += radius * angle.cos();
        image[i][1] += radius * angle.sin();
    }
}

fn bench_epnp(c: &mut Criterion) {
    let mut group = c.benchmark_group("pnp_epnp");
    for &n in &[8usize, 32, 128, 512, 2048] {
        let (world, image, k) = generate_cube_dataset_with_seed(n, 0.5, 42);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let res =
                    kpnp::solve_pnp(&world, &image, &k, kpnp::PnPMethod::EPnPDefault).unwrap();
                std::hint::black_box(res);
            });
        });
    }
    group.finish();
}

fn bench_ransac(c: &mut Criterion) {
    let mut group = c.benchmark_group("pnp_ransac");
    for &n in &[32usize, 128, 512, 2048] {
        // Run multiple seeds to observe distribution while keeping identical data across libraries
        let seeds: Vec<u64> = (0..NUM_SEEDS).map(|i| 10_000u64 + i as u64).collect();
        group.throughput(Throughput::Elements(n as u64));
        for &seed in &seeds {
            let (world, mut image, k) = generate_cube_dataset_with_seed(n, 0.5, seed);
            inject_outliers_random(&mut image, 0.20, seed.wrapping_add(12345));

            let params = kpnp::RansacParams {
                max_iterations: 200,
                reproj_threshold_px: 3.0,
                confidence: 0.99,
                random_seed: Some(seed),
                refine: true,
            };

            group.bench_with_input(
                BenchmarkId::new("n", format!("{}_s{}", n, seed)),
                &seed,
                |b, _| {
                    b.iter(|| {
                        let res = kpnp::solve_pnp_ransac(
                            &world,
                            &image,
                            &k,
                            kpnp::PnPMethod::EPnPDefault,
                            &params,
                        )
                        .unwrap();
                        std::hint::black_box(res);
                    });
                },
            );
            // keep image mutable in scope
            let _ = image.len();
        }
    }
    group.finish();
}

#[cfg(feature = "opencv_bench")]
mod opencv_cmp {
    use super::*;
    use opencv::{calib3d, core, prelude::*};
    use std::sync::Once;

    static INIT_OPENCV: Once = Once::new();

    fn init_opencv_runtime() {
        INIT_OPENCV.call_once(|| {
            let _ = core::set_num_threads(1); // force single-threaded
            let _ = core::set_use_optimized(true); // keep SIMD optimizations
            let _ = core::set_use_opencl(false); // avoid OpenCL/GPU variability
        });
    }

    fn to_mat_object_points(world: &[[f32; 3]]) -> opencv::Result<Mat> {
        let mut m = Mat::zeros(world.len() as i32, 1, core::CV_32FC3)?.to_mat()?;
        for (i, p) in world.iter().enumerate() {
            m.at_mut::<core::Vec3f>(i as i32)?.0 = [p[0], p[1], p[2]];
        }
        Ok(m)
    }

    fn to_mat_image_points(image: &[[f32; 2]]) -> opencv::Result<Mat> {
        let mut m = Mat::zeros(image.len() as i32, 1, core::CV_32FC2)?.to_mat()?;
        for (i, p) in image.iter().enumerate() {
            m.at_mut::<core::Vec2f>(i as i32)?.0 = [p[0], p[1]];
        }
        Ok(m)
    }

    fn k_to_mat(k: &[[f32; 3]; 3]) -> opencv::Result<Mat> {
        let data = vec![
            k[0][0] as f64,
            k[0][1] as f64,
            k[0][2] as f64,
            k[1][0] as f64,
            k[1][1] as f64,
            k[1][2] as f64,
            k[2][0] as f64,
            k[2][1] as f64,
            k[2][2] as f64,
        ];
        Mat::from_slice_2d(&[&data[0..3], &data[3..6], &data[6..9]])
    }

    pub fn bench_opencv_epnp(c: &mut Criterion) {
        init_opencv_runtime();
        let mut group = c.benchmark_group("opencv_epnp");
        for &n in &[8usize, 32, 128, 512, 2048] {
            let (world, image, k) = generate_cube_dataset_with_seed(n, 0.5, 42);
            let obj = to_mat_object_points(&world).unwrap();
            let img = to_mat_image_points(&image).unwrap();
            let kmat = k_to_mat(&k).unwrap();
            let dist = Mat::zeros(4, 1, core::CV_64F).unwrap().to_mat().unwrap();
            group.throughput(Throughput::Elements(n as u64));
            let mut rvec = Mat::default();
            let mut tvec = Mat::default();
            group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
                b.iter(|| {
                    calib3d::solve_pnp(
                        &obj,
                        &img,
                        &kmat,
                        &dist,
                        &mut rvec,
                        &mut tvec,
                        false,
                        calib3d::SOLVEPNP_EPNP,
                    )
                    .unwrap();
                    std::hint::black_box((&rvec, &tvec));
                });
            });
        }
        group.finish();
    }

    pub fn bench_opencv_ransac(c: &mut Criterion) {
        init_opencv_runtime();
        let mut group = c.benchmark_group("opencv_ransac");
        for &n in &[32usize, 128, 512, 2048] {
            let seeds: Vec<u64> = (0..NUM_SEEDS).map(|i| 10_000u64 + i as u64).collect();
            group.throughput(Throughput::Elements(n as u64));
            for &seed in &seeds {
                let (world, mut image, k) = generate_cube_dataset_with_seed(n, 0.5, seed);
                inject_outliers_random(&mut image, 0.20, seed.wrapping_add(12345));

                let obj = to_mat_object_points(&world).unwrap();
                let img = to_mat_image_points(&image).unwrap();
                let kmat = k_to_mat(&k).unwrap();
                let dist = Mat::zeros(4, 1, core::CV_64F).unwrap().to_mat().unwrap();
                // Preallocate once per (n, seed) input to avoid allocation in inner loop
                let mut rvec = Mat::default();
                let mut tvec = Mat::default();
                let mut inliers = Mat::default();
                group.bench_with_input(
                    BenchmarkId::new("n", format!("{}_s{}", n, seed)),
                    &seed,
                    |b, _| {
                        b.iter(|| {
                            calib3d::solve_pnp_ransac(
                                &obj,
                                &img,
                                &kmat,
                                &dist,
                                &mut rvec,
                                &mut tvec,
                                false,
                                200,
                                3.0,
                                0.99,
                                &mut inliers,
                                calib3d::SOLVEPNP_EPNP,
                            )
                            .unwrap();
                            std::hint::black_box((&rvec, &tvec, &inliers));
                        });
                    },
                );
            }
        }
        group.finish();
    }
}

#[cfg(feature = "opencv_bench")]
criterion_group!(
    benches,
    bench_epnp,
    bench_ransac,
    opencv_cmp::bench_opencv_epnp,
    opencv_cmp::bench_opencv_ransac,
);

#[cfg(not(feature = "opencv_bench"))]
criterion_group!(benches, bench_epnp, bench_ransac,);

criterion_main!(benches);
