/// Transform a set of points using a rotation and translation.
///
/// # Arguments
///
/// * `src_points` - A set of points to be transformed.
/// * `rotation` - A rotation matrix.
/// * `translation` - A translation vector.
/// * `dst_points` - A pre-allocated vector to store the transformed points.
///
/// PRECONDITION: dst_points is a pre-allocated vector of the same size as source.
///
/// Example:
///
/// ```no_run
/// use kornia_3d::linalg::transform_points3d;
///
/// let src_points = vec![[2.0, 2.0, 2.0], [3.0, 4.0, 5.0]];
/// let rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
/// let translation = [0.0, 0.0, 0.0];
/// let mut dst_points = vec![[0.0; 3]; src_points.len()];
/// transform_points3d(&src_points, &rotation, &translation, &mut dst_points);
/// ```
pub fn transform_points3d(
    src_points: &[[f64; 3]],
    dst_r_src: &[[f64; 3]; 3],
    dst_t_src: &[f64; 3],
    dst_points: &mut [[f64; 3]],
) {
    for (point_dst, point_src) in dst_points.iter_mut().zip(src_points.iter()) {
        point_dst[0] = dot3_product(&dst_r_src[0], point_src) + dst_t_src[0];
        point_dst[1] = dot3_product(&dst_r_src[1], point_src) + dst_t_src[1];
        point_dst[2] = dot3_product(&dst_r_src[2], point_src) + dst_t_src[2];
    }
}

pub fn dot3_product(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn sub_vec3(a: &[f64; 3], b: &[f64; 3], out: &mut [f64; 3]) {
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];
}

// NOTE: not very performant
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
        for i in 0..3 {
            point_dst[i] = point_dst_col.read(i);
        }
    }
}

// NOTE: less performant than transform_points3d
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
        faer::Parallelism::None,
    );

    // apply translation to each point
    for mut col_mut in points_in_dst.col_iter_mut() {
        let sum = &dst_t_src_col + col_mut.to_owned();
        col_mut.copy_from(&sum);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transform_points_identity() {
        let src_points = vec![[2.0, 2.0, 2.0], [3.0, 4.0, 5.0]];
        let rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let translation = [0.0, 0.0, 0.0];
        let mut dst_points = vec![[0.0; 3]; src_points.len()];
        transform_points3d(&src_points, &rotation, &translation, &mut dst_points);

        assert_eq!(dst_points, src_points);
    }

    #[test]
    fn test_transform_points_roundtrip() {
        let src_points = vec![[2.0, 2.0, 2.0], [3.0, 4.0, 5.0]];
        let rotation = [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]];
        let translation = [1.0, 2.0, 3.0];

        let mut dst_points = vec![[0.0; 3]; src_points.len()];
        transform_points3d(&src_points, &rotation, &translation, &mut dst_points);

        // invert the transformation
        let dst_r_src = {
            let rotation_slice = unsafe {
                std::slice::from_raw_parts(rotation.as_ptr() as *const f64, rotation.len() * 3)
            };
            faer::mat::from_row_major_slice(rotation_slice, 3, 3)
        };
        let dst_t_src = faer::col![translation[0], translation[1], translation[2]];
        // R' = R^T
        let src_r_dst = dst_r_src.transpose();
        // t' = -R^T * t
        let src_t_dst = -src_r_dst * dst_t_src;
        let (rotation_inv, translation_inv) = {
            let mut rotation_inv = [[0.0; 3]; 3];
            for (i, row) in rotation_inv.iter_mut().enumerate() {
                for (j, val) in row.iter_mut().enumerate() {
                    *val = src_r_dst.read(i, j);
                }
            }
            let mut translation_inv = [0.0; 3];
            for (i, val) in translation_inv.iter_mut().enumerate() {
                *val = src_t_dst.read(i);
            }
            (rotation_inv, translation_inv)
        };

        // transform dst_points back to src_points
        let mut dst_points_src = vec![[0.0; 3]; dst_points.len()];
        transform_points3d(
            &dst_points,
            &rotation_inv,
            &translation_inv,
            &mut dst_points_src,
        );

        assert_eq!(dst_points_src, src_points);
    }

    #[test]
    fn test_transform_points_matmul_time() {
        let src_points = vec![[0.0; 3]; 200000];
        let rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let translation = [0.0, 0.0, 0.0];
        let mut dst_points = vec![[0.0; 3]; src_points.len()];

        let num_tests = 10;

        let mut times_matmul = Vec::with_capacity(num_tests);
        let mut times_column = Vec::with_capacity(num_tests);
        let mut times_native = Vec::with_capacity(num_tests);

        for i in 0..num_tests {
            println!("Running matmul test {}", i);
            let now = std::time::Instant::now();
            transform_points3d_matmul(&src_points, &rotation, &translation, &mut dst_points);
            let elapsed = now.elapsed();
            times_matmul.push(elapsed);
        }
        let avg_time_matmul = times_matmul.iter().sum::<std::time::Duration>() / num_tests as u32;
        println!("Average time for matmul: {:?}", avg_time_matmul);

        for i in 0..num_tests {
            println!("Running column test {}", i);
            let now = std::time::Instant::now();
            transform_points3d_col(&src_points, &rotation, &translation, &mut dst_points);
            let elapsed = now.elapsed();
            times_column.push(elapsed);
        }
        let avg_time_column = times_column.iter().sum::<std::time::Duration>() / num_tests as u32;
        println!("Average time for column: {:?}", avg_time_column);

        for i in 0..num_tests {
            println!("Running native test {}", i);
            let now = std::time::Instant::now();
            transform_points3d(&src_points, &rotation, &translation, &mut dst_points);
            let elapsed = now.elapsed();
            times_native.push(elapsed);
        }
        let avg_time_native = times_native.iter().sum::<std::time::Duration>() / num_tests as u32;
        println!("Average time for native: {:?}", avg_time_native);
    }
}
