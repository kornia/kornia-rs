use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kornia_algebra::{Vec3AF32, SE3F32, SO3F32};
use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};
use rand::Rng;

fn bench_so3(c: &mut Criterion) {
    println!("Starting SO3 Benchmark...");
    let mut group = c.benchmark_group("so3");

    let data_size = 1000;
    let omegas: Vec<Vec3AF32> = (0..data_size)
        .map(|_| {
            let mut rng = rand::rng();
            Vec3AF32::new(rng.random(), rng.random(), rng.random())
        })
        .collect();

    let omegas_na: Vec<Vector3<f32>> = omegas.iter().map(|v| Vector3::new(v.x, v.y, v.z)).collect();

    let rots: Vec<SO3F32> = omegas.iter().map(|&v| SO3F32::exp(v)).collect();
    let quats_na: Vec<UnitQuaternion<f32>> = omegas_na
        .iter()
        .map(|&v| UnitQuaternion::from_scaled_axis(v))
        .collect();

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
                std::hint::black_box(UnitQuaternion::from_scaled_axis(std::hint::black_box(
                    *omega,
                )));
            }
        })
    });

    group.bench_function(BenchmarkId::new("log_kornia", ""), |b| {
        b.iter(|| {
            for rot in rots.iter() {
                std::hint::black_box(std::hint::black_box(*rot).log());
            }
        })
    });

    group.bench_function(BenchmarkId::new("log_nalgebra", ""), |b| {
        b.iter(|| {
            for quat in quats_na.iter() {
                std::hint::black_box(std::hint::black_box(*quat).scaled_axis());
            }
        })
    });

    group.bench_function(BenchmarkId::new("inverse_kornia", ""), |b| {
        b.iter(|| {
            for rot in rots.iter() {
                std::hint::black_box(std::hint::black_box(*rot).inverse());
            }
        })
    });

    group.bench_function(BenchmarkId::new("inverse_nalgebra", ""), |b| {
        b.iter(|| {
            for quat in quats_na.iter() {
                std::hint::black_box(std::hint::black_box(*quat).inverse());
            }
        })
    });

    let omegas_s: Vec<sophus::nalgebra::Vector3<f64>> = omegas
        .iter()
        .map(|v| sophus::nalgebra::Vector3::new(v.x as f64, v.y as f64, v.z as f64))
        .collect();

    let rots_s: Vec<sophus::lie::Rotation3F64> = omegas_s
        .iter()
        .map(|&v| sophus::lie::Rotation3F64::exp(v))
        .collect();

    group.bench_function(BenchmarkId::new("exp_sophus", ""), |b| {
        b.iter(|| {
            for omega in omegas_s.iter() {
                std::hint::black_box(sophus::lie::Rotation3F64::exp(std::hint::black_box(*omega)));
            }
        })
    });

    group.bench_function(BenchmarkId::new("log_sophus", ""), |b| {
        b.iter(|| {
            for rot in rots_s.iter() {
                std::hint::black_box(std::hint::black_box(*rot).log());
            }
        })
    });

    group.bench_function(BenchmarkId::new("inverse_sophus", ""), |b| {
        b.iter(|| {
            for rot in rots_s.iter() {
                std::hint::black_box(std::hint::black_box(*rot).inverse());
            }
        })
    });
}

fn bench_se3(c: &mut Criterion) {
    let mut group = c.benchmark_group("se3");

    let data_size = 1000;
    let poses: Vec<SE3F32> = (0..data_size).map(|_| SE3F32::from_random()).collect();
    let poses2: Vec<SE3F32> = (0..data_size).map(|_| SE3F32::from_random()).collect();

    let iso_na: Vec<Isometry3<f32>> = poses
        .iter()
        .map(|p| {
            let t = Translation3::new(p.t.x, p.t.y, p.t.z);
            let q = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                p.r.q.w, p.r.q.x, p.r.q.y, p.r.q.z,
            ));
            Isometry3::from_parts(t, q)
        })
        .collect();

    let iso_na2: Vec<Isometry3<f32>> = poses2
        .iter()
        .map(|p| {
            let t = Translation3::new(p.t.x, p.t.y, p.t.z);
            let q = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                p.r.q.w, p.r.q.x, p.r.q.y, p.r.q.z,
            ));
            Isometry3::from_parts(t, q)
        })
        .collect();

    group.bench_function(BenchmarkId::new("inverse_kornia", ""), |b| {
        b.iter(|| {
            for pose in poses.iter() {
                std::hint::black_box(std::hint::black_box(*pose).inverse());
            }
        })
    });

    group.bench_function(BenchmarkId::new("inverse_nalgebra", ""), |b| {
        b.iter(|| {
            for iso in iso_na.iter() {
                std::hint::black_box(std::hint::black_box(*iso).inverse());
            }
        })
    });

    group.bench_function(BenchmarkId::new("mul_kornia", ""), |b| {
        b.iter(|| {
            for (p1, p2) in poses.iter().zip(poses2.iter()) {
                std::hint::black_box(std::hint::black_box(*p1) * std::hint::black_box(*p2));
            }
        })
    });

    group.bench_function(BenchmarkId::new("mul_nalgebra", ""), |b| {
        b.iter(|| {
            for (i1, i2) in iso_na.iter().zip(iso_na2.iter()) {
                std::hint::black_box(std::hint::black_box(*i1) * std::hint::black_box(*i2));
            }
        })
    });

    let s_poses: Vec<sophus::lie::Isometry3F64> = iso_na
        .iter()
        .map(|iso| {
            let t = sophus::nalgebra::Vector3::new(
                iso.translation.vector.x as f64,
                iso.translation.vector.y as f64,
                iso.translation.vector.z as f64,
            );
            let q = sophus::nalgebra::Quaternion::new(
                iso.rotation.quaternion().w as f64,
                iso.rotation.quaternion().coords.x as f64,
                iso.rotation.quaternion().coords.y as f64,
                iso.rotation.quaternion().coords.z as f64,
            );
            sophus::lie::Isometry3F64::from_rotation_and_translation(
                sophus::nalgebra::UnitQuaternion::from_quaternion(q).into(),
                t,
            )
        })
        .collect();

    let s_poses2: Vec<sophus::lie::Isometry3F64> = iso_na2
        .iter()
        .map(|iso| {
            let t = sophus::nalgebra::Vector3::new(
                iso.translation.vector.x as f64,
                iso.translation.vector.y as f64,
                iso.translation.vector.z as f64,
            );
            let q = sophus::nalgebra::Quaternion::new(
                iso.rotation.quaternion().w as f64,
                iso.rotation.quaternion().coords.x as f64,
                iso.rotation.quaternion().coords.y as f64,
                iso.rotation.quaternion().coords.z as f64,
            );
            sophus::lie::Isometry3F64::from_rotation_and_translation(
                sophus::nalgebra::UnitQuaternion::from_quaternion(q).into(),
                t,
            )
        })
        .collect();

    group.bench_function(BenchmarkId::new("mul_sophus_f64", ""), |b| {
        b.iter(|| {
            for (p1, p2) in s_poses.iter().zip(s_poses2.iter()) {
                std::hint::black_box(std::hint::black_box(*p1) * std::hint::black_box(*p2));
            }
        })
    });

    group.bench_function(BenchmarkId::new("inverse_sophus", ""), |b| {
        b.iter(|| {
            for pose in s_poses.iter() {
                std::hint::black_box(std::hint::black_box(*pose).inverse());
            }
        })
    });
}
criterion_group!(benches, bench_so3, bench_se3);
criterion_main!(benches);
