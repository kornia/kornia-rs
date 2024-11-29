/// Represents a 2D vector.
pub struct Vector2d {
    pub x: f64,
    pub y: f64,
}

/// Represents a Colmap camera model id.
#[derive(Debug)]
pub enum CameraModelId {
    CameraModelInvalid = -1,
    CameraModelSimplePinhole = 0,
    CameraModelPinhole = 1,
    CameraModelSimplifiedRadial = 2,
    CameraModelRadial = 3,
    CameraModelOpenCV = 4,
    CameraModelOpenCVFisheye = 5,
    CameraModelFullOpenCV = 6,
    CameraModelFOV = 7,
    CameraModelSimpleRadialFisheye = 8,
    CameraModelRadialFisheye = 9,
    CameraModelThinPrismFisheye = 10,
    CameraModelCount = 11,
}

/// Represents a camera in the Colmap system.
#[derive(Debug)]
pub struct ColmapCamera {
    pub camera_id: u32,
    pub model_id: CameraModelId,
    pub width: usize,
    pub height: usize,
    pub params: Vec<f64>,
}

/// Represents an image in the Colmap system.
#[derive(Debug)]
pub struct ColmapImage {
    pub name: String,
    pub image_id: u32,
    pub camera_id: u32,
    pub rotation: [f64; 4],    // qw, qx, qy, qz
    pub translation: [f64; 3], // x, y, z
    pub points2d: Vec<(f64, f64, i64)>,
}

/// Represents a 3D point in the Colmap system.
#[derive(Debug)]
pub struct ColmapPoint3d {
    pub point3d_id: u64,
    pub xyz: [f64; 3],
    pub rgb: [u8; 3],
    pub error: f64,
    pub track: Vec<(u32, u32)>,
}
