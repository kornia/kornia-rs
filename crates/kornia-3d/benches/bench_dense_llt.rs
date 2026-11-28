//! `DenseLlt` against faer's `Mat::llt` + `Llt::solve` on one 150×150 system, sequential.

use criterion::{criterion_group, criterion_main, Criterion};
use faer::prelude::Solve;
use faer::{Mat, Par, Side};
use kornia_3d::linalg::dense_llt::DenseLlt;
use std::hint::black_box;

const DIM: usize = 150;

/// `BᵀB + DIM·I` from a fixed LCG: symmetric positive definite.
fn spd() -> Mat<f64> {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let b = Mat::<f64>::from_fn(DIM, DIM, |_, _| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    });
    let mut a = b.transpose() * &b;
    for i in 0..DIM {
        a[(i, i)] += DIM as f64;
    }
    a
}

fn bench_dense_llt(c: &mut Criterion) {
    faer::set_global_parallelism(Par::Seq);
    let a = spd();
    let rhs = Mat::<f64>::from_fn(DIM, 1, |i, _| (i as f64 * 0.37).sin());
    let mut group = c.benchmark_group("dense_llt_150");
    group.bench_function("faer_mat_llt", |bench| {
        bench.iter(|| {
            let llt = black_box(&a).llt(Side::Lower).ok();
            llt.map(|llt| llt.solve(black_box(&rhs)))
        })
    });
    let mut dense = DenseLlt::new(DIM, Par::Seq);
    group.bench_function("dense_llt", |bench| {
        bench.iter(|| {
            dense.factor(black_box(&a)).ok()?;
            dense
                .solve_column(black_box(rhs.col_as_slice(0)))
                .ok()
                .map(|x| x[0])
        })
    });
    group.finish();
}

criterion_group!(benches, bench_dense_llt);
criterion_main!(benches);
