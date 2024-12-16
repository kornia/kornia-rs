/// Transform a set of 3D points using a rotation and translation.
///
/// # Arguments
///
/// * `src_points` - A set of 3D points to be transformed.
/// * `rotation` - A 3x3 rotation matrix.
/// * `translation` - A 3D translation vector.
/// * `dst_points` - A pre-allocated vector to store the transformed 3D points.
///
/// PRECONDITION: dst_points is a pre-allocated vector of the same size as source.
///
/// Example:
///
/// ```
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
        point_dst[0] = dot_product3(&dst_r_src[0], point_src) + dst_t_src[0];
        point_dst[1] = dot_product3(&dst_r_src[1], point_src) + dst_t_src[1];
        point_dst[2] = dot_product3(&dst_r_src[2], point_src) + dst_t_src[2];
    }
}

/// Compute the dot product of two 3D vectors.
///
/// # Arguments
///
/// * `a` - The first 3D vector.
/// * `b` - The second 3D vector.
///
/// # Returns
///
/// The dot product of the two vectors.
///
/// # Panics
///
/// Panics if the vectors are not of the same length or if the length is not 3.
///
/// # Example
///
/// ```
/// use kornia_3d::linalg::dot_product3;
///
/// let a = [1.0, 2.0, 3.0];
/// let b = [4.0, 5.0, 6.0];
/// let result = dot_product3(&a, &b);
/// assert_eq!(result, 32.0);
/// ```
pub fn dot_product3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Multiply two 3x3 matrices.
///
/// # Arguments
///
/// * `a` - The left hand side 3x3 matrix.
/// * `b` - The right hand side 3x3 matrix.
/// * `m` - A pre-allocated 3x3 matrix to store the result.
///
/// PRECONDITION: m is a pre-allocated 3x3 matrix.
///
/// # Example
///
/// ```
/// use kornia_3d::linalg::matmul33;
///
/// let a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
/// let b = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
/// let mut m = [[0.0; 3]; 3];
/// matmul33(&a, &b, &mut m);
/// ```
pub fn matmul33(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3], m: &mut [[f64; 3]; 3]) {
    m[0][0] = a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0];
    m[0][1] = a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1];
    m[0][2] = a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2];

    m[1][0] = a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0];
    m[1][1] = a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1];
    m[1][2] = a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2];

    m[2][0] = a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0];
    m[2][1] = a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1];
    m[2][2] = a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2];
}

/// Transpose a 3x3 matrix.
///
/// # Arguments
///
/// * `a` - The 3x3 matrix to transpose.
/// * `m` - A pre-allocated 3x3 matrix to store the result.
///
/// PRECONDITION: m is a pre-allocated 3x3 matrix.
///
/// # Example
///
/// ```
/// use kornia_3d::linalg::transpose33;
///
/// let a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
/// let mut m = [[0.0; 3]; 3];
/// transpose33(&a, &mut m);
/// ```
pub fn transpose33(a: &[[f64; 3]; 3], m: &mut [[f64; 3]; 3]) {
    m[0] = [a[0][0], a[1][0], a[2][0]]; // First column
    m[1] = [a[0][1], a[1][1], a[2][1]]; // Second column
    m[2] = [a[0][2], a[1][2], a[2][2]]; // Third column
}

/// Multiply a 3x3 matrix by a 3D vector.
///
/// # Arguments
///
/// * `a` - The 3x3 matrix.
/// * `b` - The 3D vector.
/// * `m` - A pre-allocated 3D vector to store the result.
///
/// PRECONDITION: m is a pre-allocated 3D vector.
///
/// # Example
///
/// ```
/// use kornia_3d::linalg::mat_vec_mul33;
///
/// let a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
/// let b = [1.0, 2.0, 3.0];
/// let mut m = [0.0; 3];
/// mat_vec_mul33(&a, &b, &mut m);
/// ```
pub fn mat_vec_mul33(a: &[[f64; 3]; 3], b: &[f64; 3], m: &mut [f64; 3]) {
    m[0] = dot_product3(&a[0], b);
    m[1] = dot_product3(&a[1], b);
    m[2] = dot_product3(&a[2], b);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product3() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let result = dot_product3(&a, &b);
        assert_eq!(result, 32.0);
    }

    #[test]
    fn test_matmul33() {
        let a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let b = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let mut m = [[0.0; 3]; 3];
        matmul33(&a, &b, &mut m);
        assert_eq!(
            m,
            [
                [30.0, 36.0, 42.0],
                [66.0, 81.0, 96.0],
                [102.0, 126.0, 150.0]
            ]
        );
    }

    #[test]
    fn test_transpose33() {
        let a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let mut m = [[0.0; 3]; 3];
        transpose33(&a, &mut m);
        assert_eq!(m, [[1.0, 4.0, 7.0], [2.0, 5.0, 8.0], [3.0, 6.0, 9.0]]);
    }

    #[test]
    fn test_mat_vec_mul33() {
        let a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let b = [1.0, 2.0, 3.0];
        let mut m = [0.0; 3];
        mat_vec_mul33(&a, &b, &mut m);
        assert_eq!(m, [14.0, 32.0, 50.0]);
    }

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
}
