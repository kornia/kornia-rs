use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use kornia_3d::linalg;

// transform_points3d_col using faer with cols point by point
fn transform_points3d_col(
    src_points: &[[f64; 3]],
    dst_r_src: &[[f64; 3]; 3],
    dst_t_src: &[f64; 3],
    dst_points: &mut [[f64; 3]],
) {
    assert_eq!(src_points.len(), dst_points.len());

    // create views of the rotation and translation matrices
    let dst_r_src_mat = faer::Mat::<f64>::from_fn(3, 3, |i, j| dst_r_src[i][j]);
    let dst_t_src_col = faer::col![dst_t_src[0], dst_t_src[1], dst_t_src[2]];

    for (point_dst, point_src) in dst_points.iter_mut().zip(src_points.iter()) {
        let point_src_col = faer::col![point_src[0], point_src[1], point_src[2]];
        let point_dst_col = &dst_r_src_mat * point_src_col + &dst_t_src_col;
        for (i, point_dst_col_val) in point_dst_col.iter().enumerate().take(3) {
            point_dst[i] = *point_dst_col_val;
        }
    }
}

// transform_points3d_matmul using faer with matmul
fn transform_points3d_matmul(
    src_points: &[[f64; 3]],
    dst_r_src: &[[f64; 3]; 3],
    dst_t_src: &[f64; 3],
    dst_points: &mut [[f64; 3]],
) {
    // create views of the rotation and translation matrices
    let dst_r_src_mat = {
        let dst_r_src_slice = unsafe {
            std::slice::from_raw_parts(dst_r_src.as_ptr() as *const f64, dst_r_src.len() * 3)
        };
        faer::mat::from_row_major_slice(dst_r_src_slice, 3, 3)
    };
    let dst_t_src_col = faer::col![dst_t_src[0], dst_t_src[1], dst_t_src[2]];

    // create view of the source points
    let points_in_src: faer::MatRef<'_, f64> = {
        let src_points_slice = unsafe {
            std::slice::from_raw_parts(src_points.as_ptr() as *const f64, src_points.len() * 3)
        };
        // SAFETY: src_points_slice is a 3xN matrix where each column represents a 3D point
        faer::mat::from_row_major_slice(src_points_slice, 3, src_points.len())
    };

    // create a mutable view of the destination points
    let mut points_in_dst = {
        let dst_points_slice = unsafe {
            std::slice::from_raw_parts_mut(
                dst_points.as_mut_ptr() as *mut f64,
                dst_points.len() * 3,
            )
        };
        // SAFETY: dst_points_slice is a 3xN matrix where each column represents a 3D point
        faer::mat::from_column_major_slice_mut(dst_points_slice, 3, dst_points.len())
    };

    // perform the matrix multiplication
    faer::linalg::matmul::matmul(
        &mut points_in_dst,
        dst_r_src_mat,
        points_in_src,
        None,
        1.0,
        faer::Parallelism::Rayon(4),
    );

    // apply translation to each point
    for mut col_mut in points_in_dst.col_iter_mut() {
        let sum = &dst_t_src_col + col_mut.to_owned();
        col_mut.copy_from(&sum);
    }
}

fn bench_transform_points3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("transform_points3d");

    for num_points in [1000, 10000, 100000, 200000, 500000].iter() {
        group.throughput(criterion::Throughput::Elements(*num_points as u64));
        let parameter_string = format!("{}", num_points);

        let src_points = vec![[2.0, 2.0, 2.0]; *num_points];
        let rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let translation = [0.0, 0.0, 0.0];
        let mut dst_points = vec![[0.0; 3]; src_points.len()];

        group.bench_with_input(
            BenchmarkId::new("transform_points3d", &parameter_string),
            &(&src_points, &rotation, &translation, &mut dst_points),
            |b, i| {
                let (src, rot, trans, mut dst) = (i.0, i.1, i.2, i.3.clone());
                b.iter(|| {
                    linalg::transform_points3d(src, rot, trans, &mut dst).unwrap();
                    black_box(());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("transform_points3d_col", &parameter_string),
            &(&src_points, &rotation, &translation, &mut dst_points),
            |b, i| {
                let (src, rot, trans, mut dst) = (i.0, i.1, i.2, i.3.clone());
                b.iter(|| {
                    transform_points3d_col(src, rot, trans, &mut dst);
                    black_box(());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("transform_points3d_matmul", &parameter_string),
            &(&src_points, &rotation, &translation, &mut dst_points),
            |b, i| {
                let (src, rot, trans, mut dst) = (i.0, i.1, i.2, i.3.clone());
                b.iter(|| {
                    transform_points3d_matmul(src, rot, trans, &mut dst);
                    black_box(());
                });
            },
        );
    }
}

fn matmul33_dot(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3], m: &mut [[f64; 3]; 3]) {
    let row0 = &a[0];
    let row1 = &a[1];
    let row2 = &a[2];

    let col0 = &[b[0][0], b[1][0], b[2][0]];
    let col1 = &[b[0][1], b[1][1], b[2][1]];
    let col2 = &[b[0][2], b[1][2], b[2][2]];

    m[0][0] = linalg::dot_product3(row0, col0);
    m[0][1] = linalg::dot_product3(row0, col1);
    m[0][2] = linalg::dot_product3(row0, col2);

    m[1][0] = linalg::dot_product3(row1, col0);
    m[1][1] = linalg::dot_product3(row1, col1);
    m[1][2] = linalg::dot_product3(row1, col2);

    m[2][0] = linalg::dot_product3(row2, col0);
    m[2][1] = linalg::dot_product3(row2, col1);
    m[2][2] = linalg::dot_product3(row2, col2);
}

fn bench_matmul33(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul33");

    let a_mat = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    let b_mat = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    let mut m_mat = [[0.0; 3]; 3];

    group.bench_function(BenchmarkId::new("matmul33", ""), |b| {
        b.iter(|| {
            linalg::matmul33(&a_mat, &b_mat, &mut m_mat);
            black_box(());
        });
    });

    group.bench_function(BenchmarkId::new("matmul33_dot", ""), |b| {
        b.iter(|| {
            matmul33_dot(&a_mat, &b_mat, &mut m_mat);
            black_box(());
        });
    });
}

criterion_group!(benches, bench_transform_points3d, bench_matmul33);
criterion_main!(benches);
