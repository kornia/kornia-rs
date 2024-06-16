pub mod distortion;

/// Represents the instrinsic parameters of a pinhole camera
///
/// # Fields
///
/// * `fx` - The focal length in the x direction
/// * `fy` - The focal length in the y direction
/// * `cx` - The x coordinate of the principal point
/// * `cy` - The y coordinate of the principal point
pub struct CameraIntrinsic {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
}

/// Represents the extrinsic parameters of a pinhole camera
///
/// # Fields
///
/// * `rotation` - The rotation matrix of the camera 3x3
/// * `translation` - The translation vector of the camera 3x1
pub struct CameraExtrinsic {
    pub rotation: [[f64; 3]; 3],
    pub translation: [f64; 3],
}
