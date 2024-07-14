/// image distortion module.
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
    /// The focal length in the x direction
    pub fx: f64,
    /// The focal length in the y direction
    pub fy: f64,
    /// The x coordinate of the principal point
    pub cx: f64,
    /// The y coordinate of the principal point
    pub cy: f64,
}

/// Represents the extrinsic parameters of a pinhole camera
///
/// # Fields
///
/// * `rotation` - The rotation matrix of the camera 3x3
/// * `translation` - The translation vector of the camera 3x1
pub struct CameraExtrinsic {
    /// The rotation matrix of the camera 3x3
    pub rotation: [[f64; 3]; 3],
    /// The translation vector of the camera 3x1
    pub translation: [f64; 3],
}
