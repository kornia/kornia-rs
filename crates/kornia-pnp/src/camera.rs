//! Camera models and distortion handling for PnP solvers.
use thiserror::Error;

/// Error types for camera operations.
#[derive(Debug, Error)]
pub enum CameraError {
    /// Invalid camera intrinsics matrix
    #[error("Invalid camera intrinsics matrix: {0}")]
    InvalidIntrinsics(String),
    
    /// Invalid distortion parameters
    #[error("Invalid distortion parameters: {0}")]
    InvalidDistortion(String),
    
    /// Failed to undistort point
    #[error("Failed to undistort point: {0}")]
    UndistortFailed(String),
}

/// Result type for camera operations.
pub type CameraResult<T> = Result<T, CameraError>;

/// Represents the intrinsic parameters of a pinhole camera.
#[derive(Debug, Clone, PartialEq)]
pub struct CameraIntrinsics {
    /// Focal length in x direction
    pub fx: f32,
    /// Focal length in y direction
    pub fy: f32,
    /// Principal point x coordinate
    pub cx: f32,
    /// Principal point y coordinate
    pub cy: f32,
}

impl CameraIntrinsics {
    /// Create camera intrinsics from focal lengths and principal point.
    pub fn new(fx: f32, fy: f32, cx: f32, cy: f32) -> Self {
        Self { fx, fy, cx, cy }
    }

    /// Create camera intrinsics from a 3x3 intrinsics matrix.
    pub fn from_matrix(k: &[[f32; 3]; 3]) -> CameraResult<Self> {
        // Check that the matrix has the expected form
        if k[0][1] != 0.0 || k[1][0] != 0.0 || k[2][0] != 0.0 || k[2][1] != 0.0 || k[2][2] != 1.0 {
            return Err(CameraError::InvalidIntrinsics(
                "Intrinsics matrix must have form [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]".to_string(),
            ));
        }
        
        Ok(Self {
            fx: k[0][0],
            fy: k[1][1],
            cx: k[0][2],
            cy: k[1][2],
        })
    }

    /// Convert to 3x3 intrinsics matrix.
    pub fn to_matrix(&self) -> [[f32; 3]; 3] {
        [
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ]
    }
}

/// Represents polynomial distortion parameters using the Brown-Conrady model.
#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)]
pub struct PolynomialDistortion {
    /// Radial distortion coefficients
    pub k1: f32,
    pub k2: f32,
    pub k3: f32,
    pub k4: f32,
    pub k5: f32,
    pub k6: f32,
    /// Tangential distortion coefficients
    pub p1: f32,
    pub p2: f32,
}

impl PolynomialDistortion {
    /// Create distortion parameters with all coefficients set to zero (no distortion).
    pub fn none() -> Self {
        Self {
            k1: 0.0, k2: 0.0, k3: 0.0, k4: 0.0, k5: 0.0, k6: 0.0,
            p1: 0.0, p2: 0.0,
        }
    }

    /// Create distortion parameters with only first two radial coefficients.
    pub fn radial(k1: f32, k2: f32) -> Self {
        Self {
            k1, k2, k3: 0.0, k4: 0.0, k5: 0.0, k6: 0.0,
            p1: 0.0, p2: 0.0,
        }
    }

    /// Create distortion parameters with radial and tangential coefficients.
    pub fn radial_tangential(k1: f32, k2: f32, p1: f32, p2: f32) -> Self {
        Self {
            k1, k2, k3: 0.0, k4: 0.0, k5: 0.0, k6: 0.0,
            p1, p2,
        }
    }

    /// Check if there is any distortion.
    pub fn has_distortion(&self) -> bool {
        self.k1 != 0.0 || self.k2 != 0.0 || self.k3 != 0.0 || 
        self.k4 != 0.0 || self.k5 != 0.0 || self.k6 != 0.0 ||
        self.p1 != 0.0 || self.p2 != 0.0
    }
}

/// A complete camera model with intrinsics and optional distortion.
#[derive(Debug, Clone)]
pub struct CameraModel {
    /// Camera intrinsics
    pub intrinsics: CameraIntrinsics,
    /// Distortion parameters (None for no distortion)
    pub distortion: Option<PolynomialDistortion>,
}

impl CameraModel {
    /// Create a camera model without distortion.
    pub fn pinhole(intrinsics: CameraIntrinsics) -> Self {
        Self {
            intrinsics,
            distortion: None,
        }
    }

    /// Create a camera model with distortion.
    pub fn with_distortion(intrinsics: CameraIntrinsics, distortion: PolynomialDistortion) -> Self {
        Self {
            intrinsics,
            distortion: Some(distortion),
        }
    }

    /// Check if the camera has distortion.
    pub fn has_distortion(&self) -> bool {
        self.distortion.as_ref().is_some_and(|d| d.has_distortion())
    }

    /// Undistort a point using iterative method.
    pub fn undistort_point(&self, x: f32, y: f32) -> CameraResult<(f32, f32)> {
        if let Some(distortion) = &self.distortion {
            self.undistort_point_iterative(x, y, distortion)
        } else {
            Ok((x, y))
        }
    }

    /// Undistort multiple points.
    pub fn undistort_points(&self, points: &[[f32; 2]]) -> CameraResult<Vec<[f32; 2]>> {
        points
            .iter()
            .map(|&[x, y]| self.undistort_point(x, y).map(|(ux, uy)| [ux, uy]))
            .collect()
    }

    /// Apply distortion to a point.
    pub fn distort_point(&self, x: f32, y: f32) -> CameraResult<(f32, f32)> {
        if let Some(distortion) = &self.distortion {
            self.distort_point_polynomial(x, y, distortion)
        } else {
            Ok((x, y))
        }
    }

    /// Distort multiple points.
    pub fn distort_points(&self, points: &[[f32; 2]]) -> CameraResult<Vec<[f32; 2]>> {
        points
            .iter()
            .map(|&[x, y]| self.distort_point(x, y).map(|(dx, dy)| [dx, dy]))
            .collect()
    }

    /// Get the intrinsics matrix for use with existing PnP solvers.
    pub fn intrinsics_matrix(&self) -> [[f32; 3]; 3] {
        self.intrinsics.to_matrix()
    }

    /// Iterative undistortion using the Brown-Conrady model.
    fn undistort_point_iterative(
        &self,
        x_distorted: f32,
        y_distorted: f32,
        distortion: &PolynomialDistortion,
    ) -> CameraResult<(f32, f32)> {
        let fx = self.intrinsics.fx;
        let fy = self.intrinsics.fy;
        let cx = self.intrinsics.cx;
        let cy = self.intrinsics.cy;

        // Normalize coordinates
        let x = (x_distorted - cx) / fx;
        let y = (y_distorted - cy) / fy;

        // Initial guess: assume no distortion
        let mut x_undistorted = x;
        let mut y_undistorted = y;

        // Iterative refinement (typically 5-10 iterations is sufficient)
        const MAX_ITERATIONS: usize = 10;
        const EPSILON: f32 = 1e-6;

        for _ in 0..MAX_ITERATIONS {
            let r2 = x_undistorted * x_undistorted + y_undistorted * y_undistorted;
            let r4 = r2 * r2;
            let r6 = r4 * r2;

            // Radial distortion
            let kr = (1.0 + distortion.k1 * r2 + distortion.k2 * r4 + distortion.k3 * r6)
                / (1.0 + distortion.k4 * r2 + distortion.k5 * r4 + distortion.k6 * r6);

            // Tangential distortion
            let x_2 = 2.0 * x_undistorted;
            let y_2 = 2.0 * y_undistorted;
            let xy_2 = x_2 * y_undistorted;
            let x_distorted_pred = x_undistorted * kr + xy_2 * distortion.p1 + distortion.p2 * (r2 + x_2 * x_undistorted);
            let y_distorted_pred = y_undistorted * kr + distortion.p1 * (r2 + y_2 * y_undistorted) + xy_2 * distortion.p2;

            // Update undistorted coordinates
            let dx = x - x_distorted_pred;
            let dy = y - y_distorted_pred;

            x_undistorted += dx;
            y_undistorted += dy;

            // Check convergence
            if dx.abs() < EPSILON && dy.abs() < EPSILON {
                break;
            }
        }

        // Denormalize coordinates
        let x_final = fx * x_undistorted + cx;
        let y_final = fy * y_undistorted + cy;

        Ok((x_final, y_final))
    }

    /// Apply polynomial distortion to a point.
    fn distort_point_polynomial(
        &self,
        x: f32,
        y: f32,
        distortion: &PolynomialDistortion,
    ) -> CameraResult<(f32, f32)> {
        let fx = self.intrinsics.fx;
        let fy = self.intrinsics.fy;
        let cx = self.intrinsics.cx;
        let cy = self.intrinsics.cy;

        // Normalize coordinates
        let x_norm = (x - cx) / fx;
        let y_norm = (y - cy) / fy;

        // Calculate radial distance
        let r2 = x_norm * x_norm + y_norm * y_norm;
        let r4 = r2 * r2;
        let r6 = r4 * r2;

        // Radial distortion
        let kr = (1.0 + distortion.k1 * r2 + distortion.k2 * r4 + distortion.k3 * r6)
            / (1.0 + distortion.k4 * r2 + distortion.k5 * r4 + distortion.k6 * r6);

        // Tangential distortion
        let x_2 = 2.0 * x_norm;
        let y_2 = 2.0 * y_norm;
        let xy_2 = x_2 * y_norm;
        let x_distorted = x_norm * kr + xy_2 * distortion.p1 + distortion.p2 * (r2 + x_2 * x_norm);
        let y_distorted = y_norm * kr + distortion.p1 * (r2 + y_2 * y_norm) + xy_2 * distortion.p2;

        // Denormalize coordinates
        let x_final = fx * x_distorted + cx;
        let y_final = fy * y_distorted + cy;

        Ok((x_final, y_final))
    }
}

impl Default for CameraModel {
    fn default() -> Self {
        Self::pinhole(CameraIntrinsics::new(1000.0, 1000.0, 640.0, 480.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_camera_intrinsics_from_matrix() {
        let k = [[1000.0, 0.0, 640.0], [0.0, 1000.0, 480.0], [0.0, 0.0, 1.0]];
        let intrinsics = CameraIntrinsics::from_matrix(&k).unwrap();
        assert_eq!(intrinsics.fx, 1000.0);
        assert_eq!(intrinsics.fy, 1000.0);
        assert_eq!(intrinsics.cx, 640.0);
        assert_eq!(intrinsics.cy, 480.0);
    }

    #[test]
    fn test_camera_intrinsics_to_matrix() {
        let intrinsics = CameraIntrinsics::new(1000.0, 1000.0, 640.0, 480.0);
        let k = intrinsics.to_matrix();
        assert_eq!(k[0][0], 1000.0);
        assert_eq!(k[1][1], 1000.0);
        assert_eq!(k[0][2], 640.0);
        assert_eq!(k[1][2], 480.0);
    }

    #[test]
    fn test_distortion_none() {
        let distortion = PolynomialDistortion::none();
        assert!(!distortion.has_distortion());
    }

    #[test]
    fn test_distortion_radial() {
        let distortion = PolynomialDistortion::radial(0.1, 0.01);
        assert!(distortion.has_distortion());
        assert_eq!(distortion.k1, 0.1);
        assert_eq!(distortion.k2, 0.01);
        assert_eq!(distortion.p1, 0.0);
    }

    #[test]
    fn test_camera_model_pinhole() {
        let intrinsics = CameraIntrinsics::new(1000.0, 1000.0, 640.0, 480.0);
        let camera = CameraModel::pinhole(intrinsics);
        assert!(!camera.has_distortion());
    }

    #[test]
    fn test_camera_model_with_distortion() {
        let intrinsics = CameraIntrinsics::new(1000.0, 1000.0, 640.0, 480.0);
        let distortion = PolynomialDistortion::radial(0.1, 0.01);
        let camera = CameraModel::with_distortion(intrinsics, distortion);
        assert!(camera.has_distortion());
    }

    #[test]
    fn test_distort_undistort_roundtrip() {
        let intrinsics = CameraIntrinsics::new(1000.0, 1000.0, 640.0, 480.0);
        let distortion = PolynomialDistortion::radial(0.1, 0.01);
        let camera = CameraModel::with_distortion(intrinsics, distortion);

        let original_point = [100.0, 200.0];
        let distorted = camera.distort_point(original_point[0], original_point[1]).unwrap();
        let undistorted = camera.undistort_point(distorted.0, distorted.1).unwrap();

        // Should be close to original (within numerical precision)
        assert!((original_point[0] - undistorted.0).abs() < 1e-3);
        assert!((original_point[1] - undistorted.1).abs() < 1e-3);
    }
}

