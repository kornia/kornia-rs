use ndarray::Array2;

/// A struct representing the intrinsic parameters of a pinhole camera.
#[derive(Debug, Clone)]
pub struct PinholeCameraIntrinsic {
    /// The focal length in pixels (fx, fy)
    pub focal_length: (f64, f64),
    /// The principal point in pixels (cx, cy)
    pub principal_point: (f64, f64),
    /// The image dimensions (width, height)
    pub image_size: (u32, u32),
}

impl PinholeCameraIntrinsic {
    /// Creates a new PinholeCameraIntrinsic with the given parameters.
    pub fn new(focal_length: (f64, f64), principal_point: (f64, f64), image_size: (u32, u32)) -> Self {
        Self {
            focal_length,
            principal_point,
            image_size,
        }
    }

    /// Returns the camera matrix as a 3x3 array.
    pub fn camera_matrix(&self) -> Array2<f64> {
        Array2::from_shape_vec((3, 3), vec![
            self.focal_length.0, 0.0, self.principal_point.0,
            0.0, self.focal_length.1, self.principal_point.1,
            0.0, 0.0, 1.0,
        ]).unwrap()
    }
} 