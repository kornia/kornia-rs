use super::CameraIntrinsic;

/// Distortion coefficients for the Kannala-Brandt equidistant fisheye model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EquidistantDistortion {
    /// Third-order coefficient.
    pub k1: f64,
    /// Fifth-order coefficient.
    pub k2: f64,
    /// Seventh-order coefficient.
    pub k3: f64,
    /// Ninth-order coefficient.
    pub k4: f64,
}

/// Projects a 3D camera-frame point to pixel coordinates using equidistant fisheye distortion.
///
/// Returns `None` when `z` is too close to zero.
pub fn project_equidistant(
    point: &[f64; 3],
    intrinsic: &CameraIntrinsic,
    distortion: &EquidistantDistortion,
) -> Option<(f64, f64)> {
    let x = point[0];
    let y = point[1];
    let z = point[2];

    if z.abs() < 1e-12 {
        return None;
    }

    let xn = x / z;
    let yn = y / z;
    let r = (xn * xn + yn * yn).sqrt();

    if r < 1e-12 {
        return Some((intrinsic.cx, intrinsic.cy));
    }

    let theta = r.atan();
    let theta2 = theta * theta;
    let theta4 = theta2 * theta2;
    let theta6 = theta4 * theta2;
    let theta8 = theta4 * theta4;
    let theta_d = theta
        * (1.0
            + distortion.k1 * theta2
            + distortion.k2 * theta4
            + distortion.k3 * theta6
            + distortion.k4 * theta8);

    let scale = theta_d / r;
    let xd = xn * scale;
    let yd = yn * scale;

    let u = intrinsic.fx * xd + intrinsic.cx;
    let v = intrinsic.fy * yd + intrinsic.cy;
    Some((u, v))
}

/// Unprojects a fisheye pixel to a unit 3D bearing vector in the camera frame.
pub fn unproject_equidistant(
    u: f64,
    v: f64,
    intrinsic: &CameraIntrinsic,
    distortion: &EquidistantDistortion,
) -> [f64; 3] {
    let xd = (u - intrinsic.cx) / intrinsic.fx;
    let yd = (v - intrinsic.cy) / intrinsic.fy;
    let rd = (xd * xd + yd * yd).sqrt();

    if rd < 1e-12 {
        return [0.0, 0.0, 1.0];
    }

    // Invert theta_d(theta) with Newton iterations.
    let mut theta = rd;
    for _ in 0..10 {
        let t2 = theta * theta;
        let t4 = t2 * t2;
        let t6 = t4 * t2;
        let t8 = t4 * t4;

        let theta_d = theta
            * (1.0
                + distortion.k1 * t2
                + distortion.k2 * t4
                + distortion.k3 * t6
                + distortion.k4 * t8);

        let dtheta_d = 1.0
            + 3.0 * distortion.k1 * t2
            + 5.0 * distortion.k2 * t4
            + 7.0 * distortion.k3 * t6
            + 9.0 * distortion.k4 * t8;

        if dtheta_d.abs() < 1e-12 {
            break;
        }

        let step = (theta_d - rd) / dtheta_d;
        theta -= step;
        if step.abs() < 1e-12 {
            break;
        }
    }

    let r = theta.tan();
    let scale = r / rd;
    let xn = xd * scale;
    let yn = yd * scale;

    let mut b = [xn, yn, 1.0];
    let norm = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
    if norm > 0.0 {
        b[0] /= norm;
        b[1] /= norm;
        b[2] /= norm;
    }
    b
}

#[cfg(test)]
mod tests {
    use super::{project_equidistant, unproject_equidistant, EquidistantDistortion};
    use crate::calibration::CameraIntrinsic;

    fn camera() -> CameraIntrinsic {
        CameraIntrinsic {
            fx: 461.6398879418857,
            fy: 459.7153043295965,
            cx: 732.9460954720965,
            cy: 720.5410566713475,
        }
    }

    fn distortion() -> EquidistantDistortion {
        EquidistantDistortion {
            k1: 0.03442918444998219,
            k2: -0.02155491263917851,
            k3: 0.0031292637056308044,
            k4: -0.0005356957576266091,
        }
    }

    #[test]
    fn optical_axis_projects_to_principal_point() {
        let k = camera();
        let d = distortion();
        let uv = project_equidistant(&[0.0, 0.0, 1.0], &k, &d).unwrap();
        assert!((uv.0 - k.cx).abs() < 1e-10);
        assert!((uv.1 - k.cy).abs() < 1e-10);
    }

    #[test]
    fn project_unproject_roundtrip_bearing() {
        let k = camera();
        let d = distortion();
        let points = [
            [0.10, 0.05, 1.00],
            [-0.20, 0.10, 1.20],
            [0.30, -0.20, 2.00],
            [-0.15, -0.25, 1.50],
        ];

        for p in points {
            let uv = project_equidistant(&p, &k, &d).unwrap();
            let b = unproject_equidistant(uv.0, uv.1, &k, &d);

            let n = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            let gt = [p[0] / n, p[1] / n, p[2] / n];

            let err =
                ((b[0] - gt[0]).powi(2) + (b[1] - gt[1]).powi(2) + (b[2] - gt[2]).powi(2)).sqrt();
            assert!(err < 1e-7, "roundtrip error too large: {err}");
        }
    }

    #[test]
    fn symmetry_holds_for_x_flip() {
        let k = camera();
        let d = distortion();

        let p1 = [0.2, 0.1, 1.0];
        let p2 = [-0.2, 0.1, 1.0];

        let (u1, v1) = project_equidistant(&p1, &k, &d).unwrap();
        let (u2, v2) = project_equidistant(&p2, &k, &d).unwrap();

        assert!((v1 - v2).abs() < 1e-8);
        assert!(((u1 - k.cx) + (u2 - k.cx)).abs() < 1e-8);
    }
}
