//! Unconstrained affine projection, including its world-to-image transform.
//!
//! Projection follows homogeneous matrix multiplication, equivalent to NumPy
//! `matrix @ np.r_[point, 1.0]`; Jacobians differentiate its eight coefficients.
use kornia_algebra::{Vec2F64, Vec3F64};

/// Float64 affine camera `pixel = matrix * [world_point; 1]`.
///
/// Unlike a central camera, this includes the extrinsics and has no unique optical centre.
/// The eight coefficients are ordered by row for optimization. No metric constraints
/// (orthogonal rows, equal scale, or a prescribed vertical direction) are imposed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AffineCamera {
    /// Row-major 2×4 world-to-pixel matrix.
    pub matrix: [[f64; 4]; 2],
}

impl AffineCamera {
    /// Project a world point into pixels.
    ///
    /// # Arguments
    /// * `point` - Point in the camera matrix's world coordinate frame.
    /// # Returns
    /// The two image coordinates; there is no perspective depth division.
    /// # Errors
    /// Infallible; nonfinite inputs propagate to the output.
    /// # Example
    /// ```
    /// use kornia_3d::camera::AffineCamera;
    /// use kornia_algebra::{Vec2F64, Vec3F64};
    /// let c = AffineCamera { matrix: [[1., 0., 0., 2.], [0., 1., 0., 3.]] };
    /// assert_eq!(c.project(&Vec3F64::new(4., 5., 6.)), Vec2F64::new(6., 8.));
    /// ```
    pub fn project(&self, point: &Vec3F64) -> Vec2F64 {
        let x = kornia_algebra::Vec4F64::new(point.x, point.y, point.z, 1.0);
        Vec2F64::from_array(
            self.matrix
                .map(|r| kornia_algebra::Vec4F64::from_array(r).dot(x)),
        )
    }

    /// Compute analytical projection Jacobians.
    ///
    /// # Arguments
    /// * `point` - Linearization point in world coordinates.
    /// # Returns
    /// Row-major `(2×8 camera, 2×3 point)` Jacobians. Camera increments are additive.
    /// # Errors
    /// Infallible; nonfinite inputs propagate to the output.
    /// # Example
    /// ```
    /// use kornia_3d::camera::AffineCamera;
    /// use kornia_algebra::Vec3F64;
    /// let camera = AffineCamera { matrix: [[2., 0., 0., 1.], [0., 3., 0., 2.]] };
    /// let (camera_jacobian, point_jacobian) = camera.jacobians(&Vec3F64::new(4., 5., 6.));
    /// assert_eq!(camera_jacobian[0], [4., 5., 6., 1., 0., 0., 0., 0.]);
    /// assert_eq!(point_jacobian, [[2., 0., 0.], [0., 3., 0.]]);
    /// ```
    pub fn jacobians(&self, point: &Vec3F64) -> ([[f64; 8]; 2], [[f64; 3]; 2]) {
        let x = [point.x, point.y, point.z, 1.0];
        let mut camera = [[0.; 8]; 2];
        camera[0][..4].copy_from_slice(&x);
        camera[1][4..].copy_from_slice(&x);
        (camera, self.matrix.map(|r| [r[0], r[1], r[2]]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn projection_matches_numpy_reference() {
        // NumPy: A @ np.array([2., -4., 1., 1.]), with A as below.
        let camera = AffineCamera {
            matrix: [[1.2, -0.3, 0.4, 12.], [0.2, 2., -0.7, -3.]],
        };
        let point = Vec3F64::new(2., -4., 1.);
        let pixel = camera.project(&point);
        assert!((pixel.x - 16.).abs() < 1e-12);
        assert!((pixel.y + 11.3).abs() < 1e-12);
        let (camera_jacobian, point_jacobian) = camera.jacobians(&point);
        assert_eq!(
            camera_jacobian,
            [
                [2., -4., 1., 1., 0., 0., 0., 0.],
                [0., 0., 0., 0., 2., -4., 1., 1.],
            ]
        );
        assert_eq!(point_jacobian, [[1.2, -0.3, 0.4], [0.2, 2., -0.7]]);
    }

    #[test]
    fn jacobians_match_finite_differences() {
        let c = AffineCamera {
            matrix: [[1.2, -0.3, 0.4, 12.], [0.2, 2., -0.7, -3.]],
        };
        let x = Vec3F64::new(2., -4., 1.);
        let (jc, jx) = c.jacobians(&x);
        let eps = 1e-6;
        for (k, _) in jc[0].iter().enumerate() {
            let mut plus = c;
            let mut minus = c;
            plus.matrix[k / 4][k % 4] += eps;
            minus.matrix[k / 4][k % 4] -= eps;
            let diff = (plus.project(&x) - minus.project(&x)) / (2. * eps);
            for (r, row) in jc.iter().enumerate() {
                assert!((diff.to_array()[r] - row[k]).abs() < 1e-8);
            }
        }
        for k in 0..3 {
            let mut plus = x.to_array();
            let mut minus = x.to_array();
            plus[k] += eps;
            minus[k] -= eps;
            let diff = (c.project(&Vec3F64::from_array(plus))
                - c.project(&Vec3F64::from_array(minus)))
                / (2. * eps);
            for (r, row) in jx.iter().enumerate() {
                assert!((diff.to_array()[r] - row[k]).abs() < 1e-8);
            }
        }
    }
}
