pub mod distortion;

pub struct CameraIntrinsic {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
}

pub struct CameraExtrinsic {
    pub rotation: [[f64; 3]; 3],
    pub translation: [f64; 3],
}
