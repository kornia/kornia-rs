use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kornia_algebra::{Vec3AF32, SE3F32, SO3F32};
// nalgebra 0.32 — used for kornia and nalgebra benchmarks
use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};
// nalgebra 0.33 — required by sophus_lie; aliased to avoid version conflict
use nalgebra33::SVector as SVec64;
use rand::Rng;
use sophus_lie::{Isometry3F64, Rotation3F64};

fn bench_so3(c: &mut Criterion) {
    println!("Starting SO3 Benchmark...");
    let mut group = c.benchmark_group("so3");

    let data_size = 1000;
    let mut rng = rand::rng();

    // Input tangent vectors — one set per library's scalar type
    let omegas: Vec<Vec3AF32> = (0..data_size)
        .map(|_| Vec3AF32::new(rng.random(), rng.random(), rng.random()))
        .collect();

    let omegas_na: Vec<Vector3<f32>> =
        omegas.iter().map(|v| Vector3::new(v.x, v.y, v.z)).collect();

    // Sophus uses nalgebra 0.33 SVectors (f64); generated independently
    let omegas_sp: Vec<SVec64<f64, 3>> = (0..data_size)
        .map(|_| SVec64::<f64, 3>::new(rng.random(), rng.random(), rng.random()))
        .collect();

    // Pre-computed rotations for log / inverse benchmarks
    let rots_kornia: Vec<SO3F32> = omegas.iter().map(|&v| SO3F32::exp(v)).collect();
    let rots_na: Vec<UnitQuaternion<f32>> = omegas_na
        .iter()
        .map(|&v| UnitQuaternion::from_scaled_axis(v))
        .collect();
    let rots_sp: Vec<Rotation3F64> =
        omegas_sp.iter().map(|&v| Rotation3F64::exp(v)).collect();

    // exp 
    group.bench_function(BenchmarkId::new("exp_kornia", ""), |b| {
        b.iter(|| {
            for omega in omegas.iter() {
                std::hint::black_box(SO3F32::exp(std::hint::black_box(*omega)));
            }
        })
    });

    group.bench_function(BenchmarkId::new("exp_nalgebra", ""), |b| {
        b.iter(|| {
            for omega in omegas_na.iter() {
                std::hint::black_box(UnitQuaternion::from_scaled_axis(
                    std::hint::black_box(*omega),
                ));
            }
        })
    });

    group.bench_function(BenchmarkId::new("exp_sophus", ""), |b| {
        b.iter(|| {
            for omega in omegas_sp.iter() {
                std::hint::black_box(Rotation3F64::exp(std::hint::black_box(*omega)));
            }
        })
    });

    // log 
    group.bench_function(BenchmarkId::new("log_kornia", ""), |b| {
        b.iter(|| {
            for rot in rots_kornia.iter() {
                std::hint::black_box(std::hint::black_box(*rot).log());
            }
        })
    });

    group.bench_function(BenchmarkId::new("log_nalgebra", ""), |b| {
        b.iter(|| {
            for quat in rots_na.iter() {
                std::hint::black_box(std::hint::black_box(*quat).scaled_axis());
            }
        })
    });

    group.bench_function(BenchmarkId::new("log_sophus", ""), |b| {
        b.iter(|| {
            for rot in rots_sp.iter() {
                std::hint::black_box(rot.log());
            }
        })
    });

    // inverse 
    group.bench_function(BenchmarkId::new("inverse_kornia", ""), |b| {
        b.iter(|| {
            for rot in rots_kornia.iter() {
                std::hint::black_box(std::hint::black_box(*rot).inverse());
            }
        })
    });

    group.bench_function(BenchmarkId::new("inverse_nalgebra", ""), |b| {
        b.iter(|| {
            for quat in rots_na.iter() {
                std::hint::black_box(std::hint::black_box(*quat).inverse());
            }
        })
    });

    group.bench_function(BenchmarkId::new("inverse_sophus", ""), |b| {
        b.iter(|| {
            for rot in rots_sp.iter() {
                std::hint::black_box(rot.inverse());
            }
        })
    });

    group.finish();
}

fn bench_se3(c: &mut Criterion) {
    println!("Starting SE3 Benchmark...");
    let mut group = c.benchmark_group("se3");

    let data_size = 1000;
    let poses_kornia: Vec<SE3F32> = (0..data_size).map(|_| SE3F32::from_random()).collect();
    let poses2_kornia: Vec<SE3F32> = (0..data_size).map(|_| SE3F32::from_random()).collect();

    let poses_na: Vec<Isometry3<f32>> = poses_kornia
        .iter()
        .map(|p| {
            let t = Translation3::new(p.t.x, p.t.y, p.t.z);
            let q = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                p.r.q.w, p.r.q.x, p.r.q.y, p.r.q.z,
            ));
            Isometry3::from_parts(t, q)
        })
        .collect();

    let poses2_na: Vec<Isometry3<f32>> = poses2_kornia
        .iter()
        .map(|p| {
            let t = Translation3::new(p.t.x, p.t.y, p.t.z);
            let q = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                p.r.q.w, p.r.q.x, p.r.q.y, p.r.q.z,
            ));
            Isometry3::from_parts(t, q)
        })
        .collect();

    // Build sophus Isometry3F64 independently from random axis-angle vectors.
    // Feeding quaternion vector parts (q.x, q.y, q.z) into Rotation3F64::exp() is wrong:
    // those are NOT the SO(3) Lie algebra vector. exp() expects an axis-angle vector.
    // Generate fresh random poses so all three libraries use valid, representative inputs.
    let mut rng2 = rand::rng();
    let poses_sp: Vec<Isometry3F64> = (0..data_size)
        .map(|_| {
            let rot = Rotation3F64::exp(SVec64::<f64, 3>::new(
                rng2.random::<f64>() * 2.0 - 1.0,
                rng2.random::<f64>() * 2.0 - 1.0,
                rng2.random::<f64>() * 2.0 - 1.0,
            ));
            Isometry3F64::from_rotation_and_translation(
                rot,
                SVec64::<f64, 3>::new(
                    rng2.random::<f64>() * 2.0 - 1.0,
                    rng2.random::<f64>() * 2.0 - 1.0,
                    rng2.random::<f64>() * 2.0 - 1.0,
                ),
            )
        })
        .collect();

    let poses2_sp: Vec<Isometry3F64> = (0..data_size)
        .map(|_| {
            let rot = Rotation3F64::exp(SVec64::<f64, 3>::new(
                rng2.random::<f64>() * 2.0 - 1.0,
                rng2.random::<f64>() * 2.0 - 1.0,
                rng2.random::<f64>() * 2.0 - 1.0,
            ));
            Isometry3F64::from_rotation_and_translation(
                rot,
                SVec64::<f64, 3>::new(
                    rng2.random::<f64>() * 2.0 - 1.0,
                    rng2.random::<f64>() * 2.0 - 1.0,
                    rng2.random::<f64>() * 2.0 - 1.0,
                ),
            )
        })
        .collect();

    // composition 
    group.bench_function(BenchmarkId::new("composition_kornia", ""), |b| {
        b.iter(|| {
            for (p1, p2) in poses_kornia.iter().zip(poses2_kornia.iter()) {
                std::hint::black_box(std::hint::black_box(*p1) * std::hint::black_box(*p2));
            }
        })
    });

    group.bench_function(BenchmarkId::new("composition_nalgebra", ""), |b| {
        b.iter(|| {
            for (i1, i2) in poses_na.iter().zip(poses2_na.iter()) {
                std::hint::black_box(std::hint::black_box(*i1) * std::hint::black_box(*i2));
            }
        })
    });

    group.bench_function(BenchmarkId::new("composition_sophus", ""), |b| {
        b.iter(|| {
            for (p1, p2) in poses_sp.iter().zip(poses2_sp.iter()) {
                std::hint::black_box(p1.group_mul(*p2));
            }
        })
    });

    // inverse 
    group.bench_function(BenchmarkId::new("inverse_kornia", ""), |b| {
        b.iter(|| {
            for pose in poses_kornia.iter() {
                std::hint::black_box(std::hint::black_box(*pose).inverse());
            }
        })
    });

    group.bench_function(BenchmarkId::new("inverse_nalgebra", ""), |b| {
        b.iter(|| {
            for iso in poses_na.iter() {
                std::hint::black_box(std::hint::black_box(*iso).inverse());
            }
        })
    });

    group.bench_function(BenchmarkId::new("inverse_sophus", ""), |b| {
        b.iter(|| {
            for pose in poses_sp.iter() {
                std::hint::black_box(pose.inverse());
            }
        })
    });

    group.finish();
}

criterion_group!(benches, bench_so3, bench_se3);
criterion_main!(benches);