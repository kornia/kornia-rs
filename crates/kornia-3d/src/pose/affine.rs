use faer::prelude::SpSolverLstsq;

/// Computes the 2D affine transformation matrix from 4 point correspondences.
///
/// * `x1` - The source points with shape (4, 2).
/// * `x2` - The destination points with shape (4, 2).
/// * `affine` - The output 2D affine transformation matrix with shape (3, 3).
///
/// # Returns
///
/// The output 2D affine transformation matrix with shape (2, 3).
pub fn affine_4pt2d(x1: &[[f64; 2]; 4], x2: &[[f64; 2]; 4], affine: &mut [[f64; 3]; 2]) {
    // construct matrix A
    let mut mat_a = faer::Mat::<f64>::zeros(8, 6);
    let mut mat_b = faer::Mat::<f64>::zeros(8, 1);

    for i in 0..4 {
        let (x1_0, x1_1) = (x1[i][0], x1[i][1]);
        let (x2_0, x2_1) = (x2[i][0], x2[i][1]);
        unsafe {
            mat_a.write_unchecked(2 * i, 0, x1_0);
            mat_a.write_unchecked(2 * i, 1, x1_1);
            mat_a.write_unchecked(2 * i, 2, 1.0);
            mat_a.write_unchecked(2 * i + 1, 3, x1_0);
            mat_a.write_unchecked(2 * i + 1, 4, x1_1);
            mat_a.write_unchecked(2 * i + 1, 5, 1.0);
            mat_b.write_unchecked(2 * i, 0, x2_0);
            mat_b.write_unchecked(2 * i + 1, 0, x2_1);
        }
    }

    let params = mat_a.qr().solve_lstsq(mat_b);
    let aff = params.col(0);

    // copy to affine matrix
    affine[0] = [aff[0], aff[1], aff[2]];
    affine[1] = [aff[3], aff[4], aff[5]];
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_affine_4pt2d_identity() {
        let x1 = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let x2 = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let expected = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let mut affine = [[0.0; 3]; 2];
        affine_4pt2d(&x1, &x2, &mut affine);

        for i in 0..2 {
            for j in 0..3 {
                assert_relative_eq!(affine[i][j], expected[i][j], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_affine_4pt2d_translation() {
        let x1 = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let x2 = [[1.0, 1.0], [2.0, 1.0], [1.0, 2.0], [2.0, 2.0]];
        let expected = [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]];
        let mut affine = [[0.0; 3]; 2];
        affine_4pt2d(&x1, &x2, &mut affine);

        for i in 0..2 {
            for j in 0..3 {
                assert_relative_eq!(affine[i][j], expected[i][j], epsilon = 1e-6);
            }
        }
    }
}
