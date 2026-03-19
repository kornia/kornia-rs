use crate::pnp::{PnPError, PnPResult, PnPSolver};
use kornia_algebra::{Mat3AF32, Mat3F64, Vec2F32, Vec3AF32, Vec3F64, SO3F32};
use kornia_imgproc::calibration::distortion::PolynomialDistortion;

/// UP2P solver instance.
pub struct UP2P;

impl PnPSolver for UP2P {
    type Param = Vec3AF32;

    fn solve(
        points_world: &[Vec3AF32],
        points_image: &[Vec2F32],
        k: &Mat3AF32,
        _distortion: Option<&PolynomialDistortion>,
        gravity: &Self::Param,
    ) -> Result<PnPResult, PnPError> {
        let multi = solve_up2p_multi(points_world, points_image, k, _distortion, gravity)?;
        // Just return the first solution for now. RANSAC will use solve_multi directly.
        multi.into_iter().next().ok_or_else(|| {
            PnPError::SvdFailed("UP2P failed to find any valid solution".to_string())
        })
    }
}

pub(crate) fn solve_up2p_multi(
    points_world: &[Vec3AF32],
    points_image: &[Vec2F32],
    k: &Mat3AF32,
    _distortion: Option<&PolynomialDistortion>,
    gravity: &Vec3AF32,
) -> Result<Vec<PnPResult>, PnPError> {
    if points_world.len() < 2 || points_image.len() < 2 {
        return Err(PnPError::InsufficientCorrespondences {
            required: 2,
            actual: points_world.len().min(points_image.len()),
        });
    }

    // Prepare inputs as f64
    let pw = [
        [
            points_world[0].x as f64,
            points_world[0].y as f64,
            points_world[0].z as f64,
        ],
        [
            points_world[1].x as f64,
            points_world[1].y as f64,
            points_world[1].z as f64,
        ],
    ];

    // Convert pixel to normalized bearing vectors
    let fx = k.x_axis().x as f64;
    let fy = k.y_axis().y as f64;
    let cx = k.z_axis().x as f64;
    let cy = k.z_axis().y as f64;

    let mut bv = [[0.0; 3]; 2];
    for i in 0..2 {
        let u = points_image[i].x as f64;
        let v = points_image[i].y as f64;
        let x = (u - cx) / fx;
        let y = (v - cy) / fy;
        let norm = (x * x + y * y + 1.0).sqrt();
        bv[i] = [x / norm, y / norm, 1.0 / norm];
    }

    let g = [gravity.x as f64, gravity.y as f64, gravity.z as f64];

    let sols_f64 = solve_up2p(&pw, &bv, &g);

    let mut results = Vec::new();
    for (r, t) in sols_f64 {
        // Convert F64 back to AF32
        let r_f32 = Mat3AF32::from_cols(
            Vec3AF32::new(
                r.x_axis().x as f32,
                r.x_axis().y as f32,
                r.x_axis().z as f32,
            ),
            Vec3AF32::new(
                r.y_axis().x as f32,
                r.y_axis().y as f32,
                r.y_axis().z as f32,
            ),
            Vec3AF32::new(
                r.z_axis().x as f32,
                r.z_axis().y as f32,
                r.z_axis().z as f32,
            ),
        );
        let t_f32 = Vec3AF32::new(t.x as f32, t.y as f32, t.z as f32);

        let rvec = SO3F32::from_matrix(&r_f32).log();

        results.push(PnPResult {
            rotation: r_f32,
            translation: t_f32,
            rvec,
            reproj_rmse: None,
            num_iterations: None,
            converged: Some(true),
        });
    }

    Ok(results)
}

/// Solve absolute pose from 2 point correspondences with known gravity direction.
/// Gravity constrains roll and pitch, leaving yaw + translation (4 DOF).
/// Returns up to 2 solutions.
pub fn solve_up2p(
    points_world: &[[f64; 3]; 2],
    bearing_vectors: &[[f64; 3]; 2],
    gravity: &[f64; 3], // unit gravity vector in camera frame
) -> Vec<(Mat3F64, Vec3F64)> {
    let g_cam = Vec3F64::new(gravity[0], gravity[1], gravity[2]);
    let g_world = Vec3F64::new(0.0, 1.0, 0.0); // Assume upright world is Y-up

    let r_c = rotation_between(&g_cam, &g_world);
    let r_w = Mat3F64::IDENTITY; // world is already assumed to be Y-up. If not, from_two_vectors(g_world, [0,1,0])

    let mut x_upright = [Vec3F64::ZERO; 2];
    let mut x_upright_world = [Vec3F64::ZERO; 2];

    for i in 0..2 {
        let bv = Vec3F64::new(
            bearing_vectors[i][0],
            bearing_vectors[i][1],
            bearing_vectors[i][2],
        );
        let pw = Vec3F64::new(points_world[i][0], points_world[i][1], points_world[i][2]);
        x_upright[i] = r_c * bv;
        x_upright_world[i] = r_w * pw;
    }

    let a_matrix = [
        [
            -x_upright[0].z,
            0.0,
            x_upright[0].x,
            x_upright_world[0].x * x_upright[0].z - x_upright_world[0].z * x_upright[0].x,
        ],
        [
            0.0,
            -x_upright[0].z,
            x_upright[0].y,
            -x_upright_world[0].y * x_upright[0].z - x_upright_world[0].z * x_upright[0].y,
        ],
        [
            -x_upright[1].z,
            0.0,
            x_upright[1].x,
            x_upright_world[1].x * x_upright[1].z - x_upright_world[1].z * x_upright[1].x,
        ],
        [
            0.0,
            -x_upright[1].z,
            x_upright[1].y,
            -x_upright_world[1].y * x_upright[1].z - x_upright_world[1].z * x_upright[1].y,
        ],
    ];

    let b_vec = [
        [
            -2.0 * x_upright_world[0].x * x_upright[0].x
                - 2.0 * x_upright_world[0].z * x_upright[0].z,
            x_upright_world[0].z * x_upright[0].x - x_upright_world[0].x * x_upright[0].z,
        ],
        [
            -2.0 * x_upright_world[0].x * x_upright[0].y,
            x_upright_world[0].z * x_upright[0].y - x_upright_world[0].y * x_upright[0].z,
        ],
        [
            -2.0 * x_upright_world[1].x * x_upright[1].x
                - 2.0 * x_upright_world[1].z * x_upright[1].z,
            x_upright_world[1].z * x_upright[1].x - x_upright_world[1].x * x_upright[1].z,
        ],
        [
            -2.0 * x_upright_world[1].x * x_upright[1].y,
            x_upright_world[1].z * x_upright[1].y - x_upright_world[1].y * x_upright[1].z,
        ],
    ];

    // Invert A_matrix to solve A * y = b_vec (2 columns)
    let a_mat = nalgebra::Matrix4::new(
        a_matrix[0][0],
        a_matrix[0][1],
        a_matrix[0][2],
        a_matrix[0][3],
        a_matrix[1][0],
        a_matrix[1][1],
        a_matrix[1][2],
        a_matrix[1][3],
        a_matrix[2][0],
        a_matrix[2][1],
        a_matrix[2][2],
        a_matrix[2][3],
        a_matrix[3][0],
        a_matrix[3][1],
        a_matrix[3][2],
        a_matrix[3][3],
    );

    let b_mat = nalgebra::Matrix4x2::new(
        b_vec[0][0],
        b_vec[0][1],
        b_vec[1][0],
        b_vec[1][1],
        b_vec[2][0],
        b_vec[2][1],
        b_vec[3][0],
        b_vec[3][1],
    );

    // Solve for x
    let a_inv = match a_mat.try_inverse() {
        Some(inv) => inv,
        None => return vec![],
    };

    let y = a_inv * b_mat;

    let c2 = y[(3, 0)];
    let c3 = y[(3, 1)];

    let qq = solve_quadratic_real(1.0, c2, c3);

    let mut output = Vec::new();
    for q in qq {
        let q2 = q * q;
        let inv_norm = 1.0 / (1.0 + q2);
        let cq = (1.0 - q2) * inv_norm;
        let sq = 2.0 * q * inv_norm;

        let r = Mat3F64::from_cols(
            Vec3F64::new(cq, 0.0, -sq),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(sq, 0.0, cq),
        );

        let t0 = y[(0, 0)] * q + y[(0, 1)];
        let t1 = y[(1, 0)] * q + y[(1, 1)];
        let t2 = y[(2, 0)] * q + y[(2, 1)];

        let t = Vec3F64::new(t0, t1, t2) * (-inv_norm);

        // De-rotate
        let r_orig = r_c.transpose() * r * r_w;
        let t_orig = r_c.transpose() * t;

        output.push((r_orig, t_orig));
    }

    output
}

fn solve_quadratic_real(a: f64, b: f64, c: f64) -> Vec<f64> {
    let d = b * b - 4.0 * a * c;
    if d < 0.0 {
        vec![]
    } else if d == 0.0 {
        vec![-b / (2.0 * a)]
    } else {
        let root = d.sqrt();
        vec![(-b + root) / (2.0 * a), (-b - root) / (2.0 * a)]
    }
}

fn rotation_between(a: &Vec3F64, b: &Vec3F64) -> Mat3F64 {
    let cross = |v1: &Vec3F64, v2: &Vec3F64| {
        Vec3F64::new(
            v1.y * v2.z - v1.z * v2.y,
            v1.z * v2.x - v1.x * v2.z,
            v1.x * v2.y - v1.y * v2.x,
        )
    };

    let a_norm = *a;
    let b_norm = *b;
    let mut a_n = a_norm.normalize();
    let mut b_n = b_norm.normalize();

    // In case zero vector
    if a_n.x.is_nan() {
        a_n = Vec3F64::new(1.0, 0.0, 0.0);
    }
    if b_n.x.is_nan() {
        b_n = Vec3F64::new(1.0, 0.0, 0.0);
    }

    let c = a_n.dot(b_n);
    if c >= 1.0 - 1e-8 {
        return Mat3F64::IDENTITY;
    }

    if c <= -1.0 + 1e-8 {
        let mut ortho = cross(&Vec3F64::new(1.0, 0.0, 0.0), &a_n);
        if ortho.x * ortho.x + ortho.y * ortho.y + ortho.z * ortho.z < 1e-8 {
            ortho = cross(&Vec3F64::new(0.0, 1.0, 0.0), &a_n);
        }
        ortho = ortho.normalize();
        let vx = Mat3F64::from_cols(
            Vec3F64::new(0.0, ortho.z, -ortho.y),
            Vec3F64::new(-ortho.z, 0.0, ortho.x),
            Vec3F64::new(ortho.y, -ortho.x, 0.0),
        );
        return Mat3F64::IDENTITY + (vx * vx) * 2.0;
    }

    let v = cross(&a_n, &b_n);
    let vx = Mat3F64::from_cols(
        Vec3F64::new(0.0, v.z, -v.y),
        Vec3F64::new(-v.z, 0.0, v.x),
        Vec3F64::new(v.y, -v.x, 0.0),
    );
    let vx2 = vx * vx;
    Mat3F64::IDENTITY + vx + vx2 * (1.0 / (1.0 + c))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve_up2p() {
        let pw = [[1.0, 2.0, 5.0], [-1.0, 3.0, 6.0]];

        // Let's create a known ground truth pose (pure yaw for simplicity + some translation)
        let yaw = 0.5f64;
        let cy = yaw.cos();
        let sy = yaw.sin();
        let r_gt = Mat3F64::from_cols(
            Vec3F64::new(cy, 0.0, -sy),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(sy, 0.0, cy),
        );
        let t_gt = Vec3F64::new(0.1, -0.2, 0.3);

        let mut bv = [[0.0; 3]; 2];
        for i in 0..2 {
            let p = Vec3F64::new(pw[i][0], pw[i][1], pw[i][2]);
            let pc = r_gt * p + t_gt;
            let pcn = pc.normalize();
            bv[i] = [pcn.x, pcn.y, pcn.z];
        }

        // Camera's y-axis points down normally, but we aligned g to (0, 1, 0).
        // If the world is strictly Y-up, and there's no roll/pitch, gravity in cam is (0, 1, 0)
        let gravity = [0.0, 1.0, 0.0];

        let sols = solve_up2p(&pw, &bv, &gravity);
        assert!(!sols.is_empty());

        let mut found = false;
        for (r, t) in sols {
            if (r.x_axis().x - r_gt.x_axis().x).abs() < 1e-4 && (t.x - t_gt.x).abs() < 1e-4 {
                found = true;
                break;
            }
        }
        assert!(found, "Ground truth solution not found in up2p output");
    }

    /// Helper: given a ground-truth R, t and world points, synthesise bearing vectors.
    fn make_bearing_vectors(r: &Mat3F64, t: &Vec3F64, pw: &[[f64; 3]; 2]) -> [[f64; 3]; 2] {
        let mut bv = [[0.0; 3]; 2];
        for i in 0..2 {
            let p = Vec3F64::new(pw[i][0], pw[i][1], pw[i][2]);
            let pc = *r * p + *t;
            let pcn = pc.normalize();
            bv[i] = [pcn.x, pcn.y, pcn.z];
        }
        bv
    }

    /// Helper: check if any returned solution matches the ground truth.
    fn assert_solution_found(
        sols: &[(Mat3F64, Vec3F64)],
        r_gt: &Mat3F64,
        t_gt: &Vec3F64,
        tol: f64,
    ) {
        let found = sols.iter().any(|(r, t)| {
            let r_err = (r.x_axis().x - r_gt.x_axis().x).abs()
                + (r.x_axis().y - r_gt.x_axis().y).abs()
                + (r.x_axis().z - r_gt.x_axis().z).abs()
                + (r.z_axis().x - r_gt.z_axis().x).abs()
                + (r.z_axis().z - r_gt.z_axis().z).abs();
            let t_err = (t.x - t_gt.x).abs() + (t.y - t_gt.y).abs() + (t.z - t_gt.z).abs();
            r_err < tol && t_err < tol
        });
        assert!(
            found,
            "Ground truth solution not found among {} candidates",
            sols.len()
        );
    }

    #[test]
    fn test_up2p_zero_yaw() {
        // Identity rotation (yaw = 0), only translation
        let pw = [[2.0, 1.0, 8.0], [-3.0, 0.5, 6.0]];
        let r_gt = Mat3F64::IDENTITY;
        let t_gt = Vec3F64::new(0.5, -0.3, 1.0);
        let bv = make_bearing_vectors(&r_gt, &t_gt, &pw);
        let gravity = [0.0, 1.0, 0.0];

        let sols = solve_up2p(&pw, &bv, &gravity);
        assert!(!sols.is_empty());
        assert_solution_found(&sols, &r_gt, &t_gt, 1e-4);
    }

    #[test]
    fn test_up2p_large_yaw() {
        // ~90 degree yaw rotation
        let pw = [[1.0, 0.0, 10.0], [0.0, 2.0, 8.0]];
        let yaw = std::f64::consts::FRAC_PI_2; // 90°
        let cy = yaw.cos();
        let sy = yaw.sin();
        let r_gt = Mat3F64::from_cols(
            Vec3F64::new(cy, 0.0, -sy),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(sy, 0.0, cy),
        );
        let t_gt = Vec3F64::new(-1.0, 0.5, 2.0);
        let bv = make_bearing_vectors(&r_gt, &t_gt, &pw);
        let gravity = [0.0, 1.0, 0.0];

        let sols = solve_up2p(&pw, &bv, &gravity);
        assert!(!sols.is_empty());
        assert_solution_found(&sols, &r_gt, &t_gt, 1e-4);
    }

    #[test]
    fn test_up2p_tilted_gravity() {
        // Gravity is NOT along Y — simulating a tilted camera
        let pw = [[3.0, 1.0, 7.0], [-2.0, 4.0, 5.0]];
        let yaw = 0.3f64;
        let cy = yaw.cos();
        let sy = yaw.sin();

        // Ground truth rotation is pure yaw in the upright frame,
        // but gravity in camera frame is tilted
        let gravity_cam = [0.0, 0.7071, 0.7071]; // 45° tilt from Y toward Z

        // Build R_gt that is the composition: R_c^T * R_yaw * R_w
        // where R_c aligns gravity_cam to Y, and R_w = I (world is Y-up)
        let g_cam = Vec3F64::new(gravity_cam[0], gravity_cam[1], gravity_cam[2]);
        let g_world = Vec3F64::new(0.0, 1.0, 0.0);
        let r_c = rotation_between(&g_cam, &g_world);
        let r_yaw = Mat3F64::from_cols(
            Vec3F64::new(cy, 0.0, -sy),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(sy, 0.0, cy),
        );
        let r_gt = r_c.transpose() * r_yaw;
        let t_gt_upright = Vec3F64::new(0.2, -0.1, 0.5);
        let t_gt = r_c.transpose() * t_gt_upright;

        // Synthesize bearing vectors from ground truth
        let mut bv = [[0.0; 3]; 2];
        for i in 0..2 {
            let p = Vec3F64::new(pw[i][0], pw[i][1], pw[i][2]);
            let pc = r_gt * p + t_gt;
            let pcn = pc.normalize();
            bv[i] = [pcn.x, pcn.y, pcn.z];
        }

        let sols = solve_up2p(&pw, &bv, &gravity_cam);
        assert!(!sols.is_empty());
        assert_solution_found(&sols, &r_gt, &t_gt, 1e-3);
    }

    #[test]
    fn test_up2p_returns_at_most_two_solutions() {
        let pw = [[1.0, 2.0, 5.0], [-1.0, 3.0, 6.0]];
        let yaw = 1.0f64;
        let cy = yaw.cos();
        let sy = yaw.sin();
        let r_gt = Mat3F64::from_cols(
            Vec3F64::new(cy, 0.0, -sy),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(sy, 0.0, cy),
        );
        let t_gt = Vec3F64::new(0.1, -0.2, 0.3);
        let bv = make_bearing_vectors(&r_gt, &t_gt, &pw);
        let gravity = [0.0, 1.0, 0.0];

        let sols = solve_up2p(&pw, &bv, &gravity);
        assert!(
            sols.len() <= 2,
            "UP2P should return at most 2 solutions, got {}",
            sols.len()
        );
    }

    #[test]
    fn test_up2p_multi_insufficient_points() {
        // Only 1 point → should error
        let world = [Vec3AF32::new(1.0, 2.0, 3.0)];
        let image = [Vec2F32::new(400.0, 300.0)];
        let k = Mat3AF32::from_cols(
            Vec3AF32::new(800.0, 0.0, 0.0),
            Vec3AF32::new(0.0, 800.0, 0.0),
            Vec3AF32::new(640.0, 480.0, 1.0),
        );
        let gravity = Vec3AF32::new(0.0, 1.0, 0.0);

        let result = solve_up2p_multi(&world, &image, &k, None, &gravity);
        assert!(result.is_err());
    }
}
