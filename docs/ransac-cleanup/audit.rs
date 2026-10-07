//! Frozen-binary output audit for behavior-preserving RANSAC refactors.

use kornia_3d::pose::{
    ransac_essential_5pt, ransac_fundamental, ransac_fundamental_8point, ransac_homography,
    EssentialNister5ptSolver, RansacParams, RansacResult, TwoViewError, TwoViewEstimator,
};
use kornia_algebra::{Mat3F64, Vec2F64, Vec3F64};

fn matrix(m: Mat3F64) -> Vec<u64> {
    m.to_cols_array().iter().map(|x| x.to_bits()).collect()
}

fn emit(result: Result<RansacResult<Mat3F64>, TwoViewError>) {
    match result {
        Ok(r) => println!(
            "{:?} {:?} {} {:x}",
            matrix(r.model),
            r.inliers,
            r.inlier_count,
            r.score.to_bits()
        ),
        Err(e) => println!("error: {e}"),
    }
}

fn scene(n: usize, planar: bool, outliers: usize) -> (Vec<Vec2F64>, Vec<Vec2F64>, Mat3F64) {
    let angle = 5.0_f64.to_radians();
    let (s, c) = angle.sin_cos();
    let r = Mat3F64::from_cols(
        Vec3F64::new(c, 0.0, -s),
        Vec3F64::new(0.0, 1.0, 0.0),
        Vec3F64::new(s, 0.0, c),
    );
    let t = Vec3F64::new(0.5, 0.1, 0.2);
    let k = Mat3F64::from_cols(
        Vec3F64::new(500.0, 0.0, 0.0),
        Vec3F64::new(0.0, 500.0, 0.0),
        Vec3F64::new(320.0, 240.0, 1.0),
    );
    let mut state = 42_u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 32) as f64 / 4_294_967_296.0
    };
    let mut x1 = Vec::with_capacity(n);
    let mut x2 = Vec::with_capacity(n);
    for i in 0..n {
        let x = (next() - 0.5) * 3.0;
        let y = (next() - 0.5) * 2.0;
        let depth = next() * 3.0 + 3.0;
        let p1 = Vec3F64::new(x, y, if planar { 4.0 } else { depth });
        let p2 = r * p1 + t;
        x1.push(Vec2F64::new(
            500.0 * p1.x / p1.z + 320.0,
            500.0 * p1.y / p1.z + 240.0,
        ));
        let noise = (next() - 0.5) * 0.1;
        x2.push(if i < outliers {
            Vec2F64::new(next() * 640.0, next() * 480.0)
        } else {
            Vec2F64::new(
                500.0 * p2.x / p2.z + 320.0 + noise,
                500.0 * p2.y / p2.z + 240.0 - noise,
            )
        });
    }
    (x1, x2, k)
}

fn main() {
    for n in [0, 4, 7, 8, 40, 129] {
        for planar in [false, true] {
            for outliers in [0, n / 2] {
                let (x1, x2, k) = scene(n, planar, outliers);
                for seed in [0, 42, 1004] {
                    for budget in [1, 128] {
                        for refit in [false, true] {
                            let p = RansacParams {
                                max_iterations: budget,
                                threshold: 0.5,
                                min_inliers: 7,
                                random_seed: Some(seed),
                                refit,
                                ..Default::default()
                            };
                            println!("case {n} {planar} {outliers} {seed} {budget} {refit}");
                            emit(ransac_fundamental(&x1, &x2, &p));
                            emit(ransac_fundamental_8point(&x1, &x2, &p));
                            emit(ransac_essential_5pt(&x1, &x2, &k, &k, &p));
                            emit(ransac_homography(&x1, &x2, &p));
                        }
                    }
                }
                // Default LM/annealing/cheirality pipeline and E5 adapter.
                if n >= 40 {
                    for essential in [false, true] {
                        let estimator = if essential {
                            TwoViewEstimator::builder()
                                .epipolar_solver(EssentialNister5ptSolver::default())
                                .build()
                        } else {
                            TwoViewEstimator::default()
                        };
                        match estimator.estimate(&x1, &x2, &k, &k) {
                            Ok(r) => println!(
                                "pipeline {essential} {:?} {:?} {:?} {:?} {:?}",
                                matrix(r.rotation),
                                r.translation.to_array().map(f64::to_bits),
                                r.inliers,
                                r.inlier_indices,
                                r.points3d
                                    .iter()
                                    .map(|p| p.to_array().map(f64::to_bits))
                                    .collect::<Vec<_>>()
                            ),
                            Err(e) => println!("pipeline {essential} error: {e}"),
                        }
                    }
                }
            }
        }
    }
}
