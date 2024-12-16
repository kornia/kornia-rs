/// Compute the rotation matrix from an axis and angle.
///
/// # Arguments
///
/// * `axis` - The axis of rotation.
/// * `angle` - The angle of rotation.
///
/// # Returns
///
/// The rotation matrix.
///
/// PRECONDITION: axis is a unit vector.
///
/// Example:
///
/// ```no_run
/// use kornia_3d::transforms::axis_angle_to_rotation_matrix;
///
/// let axis = [1.0, 0.0, 0.0];
/// let angle = std::f64::consts::PI / 2.0;
/// let rotation = axis_angle_to_rotation_matrix(&axis, angle).unwrap();
/// assert_eq!(rotation, [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]);
/// ```
pub fn axis_angle_to_rotation_matrix(
    axis: &[f64; 3],
    angle: f64,
) -> Result<[[f64; 3]; 3], &'static str> {
    // normalize the vector
    let axis_norm = {
        let magnitude = (axis[0].powi(2) + axis[1].powi(2) + axis[2].powi(2)).sqrt();
        match magnitude < 1e-10 {
            true => return Err("cannot compute rotation matrix from a zero vector"),
            false => [
                axis[0] / magnitude,
                axis[1] / magnitude,
                axis[2] / magnitude,
            ],
        }
    };

    let x = axis_norm[0];
    let y = axis_norm[1];
    let z = axis_norm[2];

    let c = angle.cos();
    let s = angle.sin();
    let t = 1.0 - c;

    let m00 = c + x * x * t;
    let m11 = c + y * y * t;
    let m22 = c + z * z * t;

    let tmp1 = x * y * t;
    let tmp2 = z * s;

    let m10 = tmp1 + tmp2;
    let m01 = tmp1 - tmp2;

    let tmp3 = x * z * t;
    let tmp4 = y * s;

    let m20 = tmp3 - tmp4;
    let m02 = tmp3 + tmp4;

    let tmp5 = y * z * t;
    let tmp6 = x * s;

    let m12 = tmp5 - tmp6;
    let m21 = tmp5 + tmp6;

    Ok([[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]])
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_axis_angle_to_rotation_matrix_identity() -> Result<(), Box<dyn std::error::Error>> {
        let axis = [1.0, 0.0, 0.0];
        let angle = std::f64::consts::PI / 2.0;
        let rotation = axis_angle_to_rotation_matrix(&axis, angle)?;
        let expected = [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]];
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(rotation[i][j], expected[i][j]);
            }
        }
        Ok(())
    }
}
