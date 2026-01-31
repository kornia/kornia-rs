use kornia_algebra::{Mat3F32, Mat3F64, Vec3F32, Vec3F64};

// TODO(issue #451): Temporary conversion until f64 SVD is supported.
pub(crate) fn mat3f64_to_mat3f32(m: &Mat3F64) -> Mat3F32 {
    let c0 = m.x_axis();
    let c1 = m.y_axis();
    let c2 = m.z_axis();
    Mat3F32::from_cols(
        Vec3F32::new(c0.x as f32, c0.y as f32, c0.z as f32),
        Vec3F32::new(c1.x as f32, c1.y as f32, c1.z as f32),
        Vec3F32::new(c2.x as f32, c2.y as f32, c2.z as f32),
    )
}

// TODO(issue #451): Temporary conversion until f64 SVD is supported.
pub(crate) fn mat3f32_to_mat3f64(m: &Mat3F32) -> Mat3F64 {
    let c0 = m.x_axis();
    let c1 = m.y_axis();
    let c2 = m.z_axis();
    Mat3F64::from_cols(
        Vec3F64::new(c0.x as f64, c0.y as f64, c0.z as f64),
        Vec3F64::new(c1.x as f64, c1.y as f64, c1.z as f64),
        Vec3F64::new(c2.x as f64, c2.y as f64, c2.z as f64),
    )
}

pub(crate) fn vec3f32_to_vec3f64(v: &Vec3F32) -> Vec3F64 {
    Vec3F64::new(v.x as f64, v.y as f64, v.z as f64)
}
