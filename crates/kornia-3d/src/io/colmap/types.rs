/// Represents a 2D vector.
pub struct Vector2d {
    /// x coordinate
    pub x: f64,
    /// y coordinate
    pub y: f64,
}

/// Represents a Colmap camera model id.
#[derive(Debug)]
pub enum CameraModelId {
    /// Invalid camera model
    CameraModelInvalid = -1,
    /// Simple pinhole camera model
    CameraModelSimplePinhole = 0,
    /// Pinhole camera model
    CameraModelPinhole = 1,
    /// Simplified radial camera model
    CameraModelSimplifiedRadial = 2,
    /// Radial camera model
    CameraModelRadial = 3,
    /// OpenCV camera model
    CameraModelOpenCV = 4,
    /// OpenCV fisheye camera model
    CameraModelOpenCVFisheye = 5,
    /// Full OpenCV camera model
    CameraModelFullOpenCV = 6,
    /// Field of view camera model
    CameraModelFOV = 7,
    /// Simple radial fisheye camera model
    CameraModelSimpleRadialFisheye = 8,
    /// Radial fisheye camera model
    CameraModelRadialFisheye = 9,
    /// Thin prism fisheye camera model
    CameraModelThinPrismFisheye = 10,
    /// Camera model count
    CameraModelCount = 11,
}

/// Represents a camera in the Colmap system.
#[derive(Debug)]
pub struct ColmapCamera {
    /// Camera id
    pub camera_id: u32,
    /// Camera model id
    pub model_id: CameraModelId,
    /// Image width
    pub width: usize,
    /// Image height
    pub height: usize,
    /// Camera parameters
    pub params: Vec<f64>,
}

/// Represents an image in the Colmap system.
#[derive(Debug)]
pub struct ColmapImage {
    /// Image name
    pub name: String,
    /// Image id
    pub image_id: u32,
    /// Camera id
    pub camera_id: u32,
    /// Rotation
    pub rotation: [f64; 4], // qw, qx, qy, qz
    /// Translation
    pub translation: [f64; 3], // x, y, z
    /// Points2d
    pub points2d: Vec<(f64, f64, i64)>,
}

/// Represents a 3D point in the Colmap system.
#[derive(Debug)]
pub struct ColmapPoint3d {
    /// Point3d id
    pub point3d_id: u64,
    /// x, y, z coordinates
    pub xyz: [f64; 3],
    /// rgb color
    pub rgb: [u8; 3],
    /// Error
    pub error: f64,
    /// Track
    pub track: Vec<(u32, u32)>,
}
