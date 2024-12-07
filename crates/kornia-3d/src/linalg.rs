use crate::utils;

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
/// use kornia_3d::linalg::transform_points;
///
/// let src_points = vec![[2.0, 2.0, 2.0], [3.0, 4.0, 5.0]];
/// let rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
/// let translation = [0.0, 0.0, 0.0];
/// let mut dst_points = vec![[0.0; 3]; src_points.len()];
/// transform_points(&src_points, &rotation, &translation, &mut dst_points);
/// ```
pub fn transform_points(
    src_points: &[[f64; 3]],
    dst_r_src: &[[f64; 3]; 3],
    dst_t_src: &[f64; 3],
    dst_points: &mut [[f64; 3]],
) {
    assert_eq!(src_points.len(), dst_points.len());

    // create views of the rotation and translation matrices
    let dst_r_src_mat = utils::array33_to_faer_mat33(dst_r_src);
    let dst_t_src_col = utils::array3_to_faer_col(dst_t_src);

    // create view of the source points
    let points_in_src = {
        let src_points_slice = unsafe {
            std::slice::from_raw_parts(src_points.as_ptr() as *const f64, src_points.len() * 3)
        };
        // SAFETY: src_points_slice is a 3xN matrix where each column represents a 3D point
        faer::mat::from_row_major_slice(src_points_slice, src_points.len(), 3)
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
        points_in_src.transpose(),
        None,
        1.0,
        faer::Parallelism::None,
    );

    // SAFETY: dst_t_src is guaranteed to be length 3 by construction
    let (tx, ty, tz) = unsafe {
        (
            dst_t_src_col.read_unchecked(0),
            dst_t_src_col.read_unchecked(1),
            dst_t_src_col.read_unchecked(2),
        )
    };

    // SAFETY: points_in_dst is a 3xN matrix where each column represents a 3D point
    // The unchecked reads/writes are within bounds as we're only accessing indices 0,1,2
    for mut col in points_in_dst.col_iter_mut() {
        unsafe {
            col.write_unchecked(0, col.read_unchecked(0) + tx);
            col.write_unchecked(1, col.read_unchecked(1) + ty);
            col.write_unchecked(2, col.read_unchecked(2) + tz);
        }
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
        transform_points(&src_points, &rotation, &translation, &mut dst_points);

        assert_eq!(dst_points, src_points);
    }

    #[test]
    fn test_transform_points_roundtrip() {
        let src_points = vec![[2.0, 2.0, 2.0], [3.0, 4.0, 5.0]];
        let rotation = [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]];
        let translation = [1.0, 2.0, 3.0];

        let mut dst_points = vec![[0.0; 3]; src_points.len()];
        transform_points(&src_points, &rotation, &translation, &mut dst_points);

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
        transform_points(
            &dst_points,
            &rotation_inv,
            &translation_inv,
            &mut dst_points_src,
        );

        assert_eq!(dst_points_src, src_points);
    }
}
