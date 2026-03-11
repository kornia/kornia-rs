use kornia_algebra::{Mat3F64, Vec2F64, Vec3F64};

/// Configuration for triangulation-backed pose validation.
#[derive(Clone, Debug)]
pub struct TriangulationConfig {
    /// Minimum parallax angle (degrees) for triangulated points.
    pub min_parallax_deg: f64,
    /// Maximum allowed gap between the closest points on the two rays.
    pub max_midpoint_gap: f64,
    /// Maximum reprojection error when validating a triangulated point.
    pub max_reprojection_error: f64,
    /// Minimum number of positive-depth triangulated points required.
    pub min_cheirality_count: usize,
}

impl Default for TriangulationConfig {
    fn default() -> Self {
        Self {
            min_parallax_deg: 1.0,
            max_midpoint_gap: 1.0,
            max_reprojection_error: 2.0,
            min_cheirality_count: 1,
        }
    }
}

pub(crate) struct TriangulateParams<'a> {
    pub k1: &'a Mat3F64,
    pub k2: &'a Mat3F64,
    pub k1_inv: &'a Mat3F64,
    pub k2_inv: &'a Mat3F64,
    pub config: &'a TriangulationConfig,
}

/// Triangulates a 3D point by midpoint between two rays with known relative pose.
///
/// This assumes a relative transform from camera 1 to camera 2:
/// `p_cam2 = r21 * p_cam1 + t21`.
///
/// `ray1_cam1` is a viewing ray in camera-1 coordinates and `ray2_cam2` is a
/// viewing ray in camera-2 coordinates. The returned point is expressed in the
/// camera-1 frame together with the distance (gap) between the closest points
/// on the two rays.
///
/// Returns `None` if the geometry is degenerate (parallel rays or invalid depth).
pub fn triangulate_midpoint_known_pose(
    ray1_cam1: &Vec3F64,
    ray2_cam2: &Vec3F64,
    r21: &Mat3F64,
    t21: &Vec3F64,
) -> Option<(Vec3F64, f64)> {
    let r21_t = r21.transpose();
    let d1 = ray1_cam1.normalize();
    let d2 = (r21_t * *ray2_cam2).normalize();
    let c2 = -(r21_t * *t21);

    let b = d1.dot(d2);
    let denom = 1.0 - b * b;
    if denom.abs() < 1e-8 {
        return None;
    }

    let w0 = -c2;
    let d = d1.dot(w0);
    let e = d2.dot(w0);
    let s1 = (b * e - d) / denom;
    let s2 = (e - b * d) / denom;
    if s1 <= 1e-8 || s2 <= 1e-8 {
        return None;
    }

    let p1 = d1 * s1;
    let p2 = c2 + d2 * s2;
    let gap = (p1 - p2).length();
    let p_cam1 = (p1 + p2) * 0.5;
    Some((p_cam1, gap))
}

pub(crate) fn triangulate_inliers(
    x1: &[Vec2F64],
    x2: &[Vec2F64],
    inliers: &[bool],
    r: &Mat3F64,
    t: &Vec3F64,
    params: &TriangulateParams<'_>,
) -> (usize, Vec<Vec3F64>, Vec<usize>) {
    let mut count = 0usize;
    let mut points = Vec::new();
    let mut indices = Vec::new();

    for i in 0..x1.len() {
        if !inliers[i] {
            continue;
        }
        let x1n = normalize_point(params.k1_inv, &x1[i]);
        let x2n = normalize_point(params.k2_inv, &x2[i]);
        let ray1 = Vec3F64::new(x1n.x, x1n.y, 1.0).normalize();
        let ray2 = Vec3F64::new(x2n.x, x2n.y, 1.0).normalize();
        let Some((_, midpoint_gap)) = triangulate_midpoint_known_pose(&ray1, &ray2, r, t) else {
            continue;
        };
        if midpoint_gap > params.config.max_midpoint_gap {
            continue;
        }

        if let Some(x) = triangulate_point_linear(&x1n, &x2n, r, t) {
            let z1 = x.z;
            let x2c = *r * x + *t;
            let z2 = x2c.z;
            let d2_world = r.transpose() * x2c;
            let reproj_th_sq =
                params.config.max_reprojection_error * params.config.max_reprojection_error;
            let err1 = reprojection_error_sq(params.k1, &x, &x1[i]);
            let err2 = reprojection_error_sq(params.k2, &x2c, &x2[i]);
            if z1 > 0.0
                && z2 > 0.0
                && parallax_ok(&x, &d2_world, params.config.min_parallax_deg)
                && err1 <= reproj_th_sq
                && err2 <= reproj_th_sq
            {
                points.push(x);
                indices.push(i);
                count += 1;
            }
        }
    }

    (count, points, indices)
}

pub(crate) fn parallax_ok(x1: &Vec3F64, x2: &Vec3F64, min_parallax_deg: f64) -> bool {
    let dot = x1.dot(*x2);
    let n1 = x1.length();
    let n2 = x2.length();
    if n1 <= 1e-12 || n2 <= 1e-12 {
        return false;
    }
    let cos_angle = (dot / (n1 * n2)).clamp(-1.0, 1.0);
    let angle = cos_angle.acos().to_degrees();
    angle >= min_parallax_deg
}

pub(crate) fn normalize_point(k_inv: &Mat3F64, x: &Vec2F64) -> Vec2F64 {
    let xh = Vec3F64::new(x.x, x.y, 1.0);
    let xn = *k_inv * xh;
    Vec2F64::new(xn.x / xn.z, xn.y / xn.z)
}

/// Triangulate a single point from two views using the DLT method.
///
/// P1 = [I | 0] (first camera at origin), P2 = [R | t].
/// Builds the 4x4 linear system `A * X = 0` and solves via SVD.
pub(crate) fn triangulate_point_linear(
    x1: &Vec2F64,
    x2: &Vec2F64,
    r: &Mat3F64,
    t: &Vec3F64,
) -> Option<Vec3F64> {
    let r_arr: [f64; 9] = (*r).into();

    let mut a = faer::Mat::<f64>::zeros(4, 4);
    a.write(0, 0, -1.0);
    a.write(0, 2, x1.x);
    a.write(1, 1, -1.0);
    a.write(1, 2, x1.y);

    let p2_2 = [r_arr[2], r_arr[5], r_arr[8], t.z];
    for j in 0..4 {
        let p2_0j = if j < 3 { r_arr[j * 3] } else { t.x };
        let p2_1j = if j < 3 { r_arr[j * 3 + 1] } else { t.y };
        a.write(2, j, x2.x * p2_2[j] - p2_0j);
        a.write(3, j, x2.y * p2_2[j] - p2_1j);
    }

    let svd = a.svd();
    let v = svd.v();
    let xh = v.col(3);
    let w = xh[3];
    if w.abs() < 1e-12 {
        return None;
    }
    Some(Vec3F64::new(xh[0] / w, xh[1] / w, xh[2] / w))
}

fn reprojection_error_sq(k: &Mat3F64, p_cam: &Vec3F64, x: &Vec2F64) -> f64 {
    let px = *k * *p_cam;
    if px.z.abs() < 1e-12 {
        return f64::INFINITY;
    }
    let dx = px.x / px.z - x.x;
    let dy = px.y / px.z - x.y;
    dx * dx + dy * dy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangulate_midpoint_known_pose_simple() {
        let r21 = Mat3F64::IDENTITY;
        let t21 = Vec3F64::new(-1.0, 0.0, 0.0);

        let p_true = Vec3F64::new(0.0, 0.0, 5.0);
        let ray1 = p_true.normalize();

        let p_cam2 = r21 * p_true + t21;
        let ray2 = p_cam2.normalize();

        let (p_est, gap) = triangulate_midpoint_known_pose(&ray1, &ray2, &r21, &t21).unwrap();
        assert!((p_est - p_true).length() < 1e-6);
        assert!(gap < 1e-6);
    }

    #[test]
    fn test_triangulate_midpoint_known_pose_parallel_rays() {
        let r21 = Mat3F64::IDENTITY;
        let t21 = Vec3F64::new(-1.0, 0.0, 0.0);

        let ray1 = Vec3F64::new(0.0, 0.0, 1.0);
        let ray2 = Vec3F64::new(0.0, 0.0, 1.0);
        assert!(triangulate_midpoint_known_pose(&ray1, &ray2, &r21, &t21).is_none());
    }

    #[test]
    fn test_parallax_ok_thresholds() {
        let x1 = Vec3F64::new(1.0, 0.0, 0.0);
        let x2 = Vec3F64::new(1.0, 0.0, 0.0);
        assert!(!parallax_ok(&x1, &x2, 1.0));

        let x3 = Vec3F64::new(0.0, 1.0, 0.0);
        assert!(parallax_ok(&x1, &x3, 30.0));
    }

    #[test]
    fn test_normalize_point_identity_and_scaled() {
        let k = Mat3F64::from_cols(
            Vec3F64::new(2.0, 0.0, 0.0),
            Vec3F64::new(0.0, 3.0, 0.0),
            Vec3F64::new(0.0, 0.0, 1.0),
        );
        let k_inv = k.inverse();
        let x = Vec2F64::new(4.0, 6.0);
        let xn = normalize_point(&k_inv, &x);
        assert!((xn.x - 2.0).abs() < 1e-12);
        assert!((xn.y - 2.0).abs() < 1e-12);
    }
}
