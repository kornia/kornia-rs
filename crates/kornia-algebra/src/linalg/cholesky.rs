use crate::{Mat3F32, Vec3F32};

/// Computes the Cholesky decomposition of a symmetric positive definite 3x3 matrix.
///
/// Returns a lower triangular matrix L such that A = L * L^T.
///
/// # Arguments
///
/// * `mat` - The symmetric positive definite matrix to decompose.
///
/// # Returns
///
/// An `Option<Mat3F32>` containing the lower triangular matrix L if regular, or `None` if the matrix is not positive definite.
pub fn cholesky_3x3(mat: &Mat3F32) -> Option<Mat3F32> {
    let mut l = Mat3F32::ZERO;
    let a = mat;

    if a.x_axis.x <= 0.0 {
        return None;
    }
    l.x_axis.x = a.x_axis.x.sqrt();

    l.x_axis.y = a.x_axis.y / l.x_axis.x;
    l.x_axis.z = a.x_axis.z / l.x_axis.x;

    let val_l22 = a.y_axis.y - l.x_axis.y * l.x_axis.y;
    if val_l22 <= 0.0 {
        return None;
    }
    l.y_axis.y = val_l22.sqrt();

    l.y_axis.z = (a.y_axis.z - l.x_axis.z * l.x_axis.y) / l.y_axis.y;

    let val_l33 = a.z_axis.z - l.x_axis.z * l.x_axis.z - l.y_axis.z * l.y_axis.z;
    if val_l33 <= 0.0 {
        return None;
    }
    l.z_axis.z = val_l33.sqrt();

    Some(l)
}

/// Solves the system Lx = b where L is a lower triangular 3x3 matrix.
///
/// # Arguments
///
/// * `l` - The lower triangular matrix L.
/// * `b` - The right-hand side vector b.
///
/// # Returns
///
/// The solution vector x.
pub fn solve_lower_triangular_3x3(l: &Mat3F32, b: &Vec3F32) -> Vec3F32 {
    let mut x = Vec3F32::ZERO;

    // x1 = b1 / L11
    x.x = b.x / l.x_axis.x;

    // x2 = (b2 - L21*x1) / L22
    x.y = (b.y - l.x_axis.y * x.x) / l.y_axis.y;

    // x3 = (b3 - L31*x1 - L32*x2) / L33
    x.z = (b.z - l.x_axis.z * x.x - l.y_axis.z * x.y) / l.z_axis.z;

    x
}

/// Solves the system L^T x = b where L is a lower triangular 3x3 matrix.
///
/// # Arguments
///
/// * `l` - The lower triangular matrix L.
/// * `b` - The right-hand side vector b.
///
/// # Returns
///
/// The solution vector x.
pub fn solve_transpose_lower_triangular_3x3(l: &Mat3F32, b: &Vec3F32) -> Vec3F32 {
    let mut x = Vec3F32::ZERO;

    // x3 = b3 / L33
    x.z = b.z / l.z_axis.z;

    // x2 = (b2 - L32*x3) / L22
    x.y = (b.y - l.y_axis.z * x.z) / l.y_axis.y;

    // x1 = (b1 - L21*x2 - L31*x3) / L11
    x.x = (b.x - l.x_axis.y * x.y - l.x_axis.z * x.z) / l.x_axis.x;

    x
}

/// Solves the system A x = b where A is a symmetric positive definite matrix with Cholesky decomposition L.
///
/// This uses an explicit inverse of L to solve the system, which can be more stable for
/// ill-conditioned matrices compared to substitution.
pub fn cholesky_solve_3x3(l: &Mat3F32, b: &Vec3F32) -> Vec3F32 {
    // We solve L * L^T * x = b by computing x = (L^-T * L^-1) * b = M^T * M * b
    // where M = L^-1.

    //Compute M = L^-1 explicitly (M is lower triangular)
    let l11 = l.x_axis.x;
    let l21 = l.x_axis.y;
    let l31 = l.x_axis.z;
    let l22 = l.y_axis.y;
    let l32 = l.y_axis.z;
    let l33 = l.z_axis.z;

    let m11 = 1.0 / l11;
    let m21 = -l21 * m11 / l22;
    let m22 = 1.0 / l22;
    let m31 = (-l31 * m11 - l32 * m21) / l33;
    let m32 = -l32 * m22 / l33;
    let m33 = 1.0 / l33;

    //Compute tmp = M * b
    let tmp0 = m11 * b.x;
    let tmp1 = m21 * b.x + m22 * b.y;
    let tmp2 = m31 * b.x + m32 * b.y + m33 * b.z;

    //Compute x = M^T * tmp
    let mut x = Vec3F32::ZERO;
    x.x = m11 * tmp0 + m21 * tmp1 + m31 * tmp2;
    x.y = m22 * tmp1 + m32 * tmp2;
    x.z = m33 * tmp2;

    x
}

// TODO: Implement f64 versions if needed.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cholesky_3x3() {
        let a = Mat3F32::from_cols(
            Vec3F32::new(6.0, 15.0, 55.0),
            Vec3F32::new(15.0, 55.0, 225.0),
            Vec3F32::new(55.0, 225.0, 979.0),
        );

        let l = cholesky_3x3(&a).unwrap();

        let expected_l = Mat3F32::from_cols(
            Vec3F32::new(2.4494898, 6.1237245, 22.453665),
            Vec3F32::new(0.0, 4.183_3, 20.916_5),
            Vec3F32::new(0.0, 0.0, 6.110_1),
        );

        assert!((l.x_axis.x - expected_l.x_axis.x).abs() < 1e-4);
        assert!((l.x_axis.y - expected_l.x_axis.y).abs() < 1e-4);
        assert!((l.x_axis.z - expected_l.x_axis.z).abs() < 1e-4);

        assert!((l.y_axis.y - expected_l.y_axis.y).abs() < 1e-4);
        assert!((l.y_axis.z - expected_l.y_axis.z).abs() < 1e-4);

        assert!((l.z_axis.z - expected_l.z_axis.z).abs() < 1e-3);
    }

    #[test]
    fn test_solve_lower_triangular() {
        let l = Mat3F32::from_cols(
            Vec3F32::new(1.0, 2.0, 4.0),
            Vec3F32::new(0.0, 3.0, 5.0),
            Vec3F32::new(0.0, 0.0, 6.0),
        );

        let b = Vec3F32::new(1.0, 5.0, 32.0);

        let x = solve_lower_triangular_3x3(&l, &b);

        assert!((x.x - 1.0).abs() < 1e-6);
        assert!((x.y - 1.0).abs() < 1e-6);
        assert!((x.z - 3.833333).abs() < 1e-6);
    }

    #[test]
    fn test_matrix_3x3_cholesky_apriltag_data() {
        // Data from kornia-apriltag utils.rs test
        let a = Mat3F32::from_cols(
            Vec3F32::new(6.0, 15.0, 55.0),
            Vec3F32::new(15.0, 55.0, 225.0),
            Vec3F32::new(55.0, 225.0, 979.0),
        );

        let l = cholesky_3x3(&a).unwrap();

        // Expected results from apriltag test (transposed to standard L form)
        // r = [2.4495, 0.0, 0.0, 6.1237, 4.1833, 0.0, 22.4537, 20.9165, 6.1101]
        // L11=r[0], L21=r[3], L31=r[6]
        // L22=r[4], L32=r[7]
        // L33=r[8]

        assert!((l.x_axis.x - 2.4495).abs() < 1e-4);
        assert!((l.x_axis.y - 6.1237).abs() < 1e-4);
        assert!((l.x_axis.z - 22.4537).abs() < 1e-4);

        assert!((l.y_axis.y - 4.1833).abs() < 1e-4);
        assert!((l.y_axis.z - 20.9165).abs() < 1e-4);

        assert!((l.z_axis.z - 6.1101).abs() < 1e-4);
    }

    #[test]
    fn test_cholesky_solve_3x3() {
        let l = Mat3F32::from_cols(
            Vec3F32::new(12.206555, 16.794254, 1.3926942),
            Vec3F32::new(0.0, 6.5538564, 0.24576962),
            Vec3F32::new(0.0, 0.0, 0.00021143198),
        );

        let b = Vec3F32::new(71.0, 95.0, 8.0);

        let x = cholesky_solve_3x3(&l, &b);

        assert!((x.x - 0.562500).abs() < 1e-4);
        assert!((x.y - -0.062500).abs() < 1e-4);
        assert!(x.z.abs() < 1e-4);
    }
}
