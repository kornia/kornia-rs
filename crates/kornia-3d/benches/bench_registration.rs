use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kornia_3d::{
    linalg::transform_points3d,
    pointcloud::PointCloud,
    registration::{icp_point_to_plane, icp_vanilla, ICPConvergenceCriteria},
    transforms::axis_angle_to_rotation_matrix,
};

/// Generate a flat plane point cloud.
fn make_plane(n: usize) -> Vec<[f64; 3]> {
    let side = (n as f64).sqrt().ceil() as usize;
    let mut points = Vec::with_capacity(side * side);
    for i in 0..side {
        for j in 0..side {
            points.push([
                i as f64 - side as f64 / 2.0,
                j as f64 - side as f64 / 2.0,
                0.0,
            ]);
        }
    }
    points
}

fn bench_icp_registration(c: &mut Criterion) {
    let mut group = c.benchmark_group("icp_registration");

    for &n in &[100, 400, 900] {
        let source_points = make_plane(n);
        let source_pcl = PointCloud::new(source_points, None, None);

        let axis = [0.0, 0.0, 1.0];
        let angle = 0.1745; // 10 degrees
        let r_known = axis_angle_to_rotation_matrix(&axis, angle).unwrap();
        let t_known = [0.2, 0.1, 0.0];

        let mut target_points = vec![[0.0; 3]; source_pcl.points().len()];
        transform_points3d(source_pcl.points(), &r_known, &t_known, &mut target_points).unwrap();
        let target_pcl = PointCloud::new(target_points, None, None);

        let criteria = ICPConvergenceCriteria {
            max_iterations: 100,
            tolerance: 1e-6,
        };

        let init_rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let init_trans = [0.0, 0.0, 0.0];

        group.bench_with_input(BenchmarkId::new("icp_point_to_point", n), &n, |b, _| {
            b.iter(|| {
                let _ = std::hint::black_box(icp_vanilla(
                    &source_pcl,
                    &target_pcl,
                    init_rot,
                    init_trans,
                    criteria.clone(),
                ));
            });
        });

        group.bench_with_input(BenchmarkId::new("icp_point_to_plane", n), &n, |b, _| {
            b.iter(|| {
                let _ = std::hint::black_box(icp_point_to_plane(
                    &source_pcl,
                    &target_pcl,
                    init_rot,
                    init_trans,
                    criteria.clone(),
                ));
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_icp_registration);
criterion_main!(benches);