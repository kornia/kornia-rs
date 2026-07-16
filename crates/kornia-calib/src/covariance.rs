//! Per-camera pose covariance and observability.
//!
//! In the rigid-board path every tag corner is a *fixed* 3D point, so a camera's pose is constrained
//! standalone by its own reprojections of those known points — no coupling through shared structure.
//! The `6×6` pose Hessian `M = Σ Jᵀ J` (J = ∂[normalized reprojection]/∂[camera-pose twist]) therefore
//! captures exactly how well the board pins that camera:
//! - `σ = σ̂²·M⁻¹` → per-camera rotation / translation standard deviation.
//! - the eigenvector of the smallest eigenvalue of the *non-dimensionalized* `M` names the weakest DOF;
//!   a near-zero eigenvalue is an **unobservable direction** (e.g. the in-plane tilt a single planar
//!   tag can't fix).
//!
//! Honest scope: this is conditional on fixed intrinsics + fixed board geometry and uses only the
//! fixed-point (board/gauge) observations. It detects under-constraint and low-parallax weakness; it
//! does NOT detect a confidently-wrong branch or intrinsic error.

use kornia_3d::ba::BaObservation;
use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec3F64;

use crate::types::CameraStats;

/// Cyclic Jacobi eigendecomposition of a symmetric 6×6 matrix. Returns `(eigenvalues, eigenvectors)`
/// where eigenvector `k` is column `k` of the returned matrix.
fn symmetric_eigen(mut a: [[f64; 6]; 6]) -> ([f64; 6], [[f64; 6]; 6]) {
    let n = 6;
    let mut v = [[0.0f64; 6]; 6];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _ in 0..100 {
        let mut off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off += a[p][q] * a[p][q];
            }
        }
        if off < 1e-30 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                // Rotate columns/rows p, q of a.
                for k in 0..n {
                    let akp = a[k][p];
                    let akq = a[k][q];
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p][k];
                    let aqk = a[q][k];
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                // Accumulate eigenvectors.
                for row in v.iter_mut() {
                    let vp = row[p];
                    let vq = row[q];
                    row[p] = c * vp - s * vq;
                    row[q] = s * vp + c * vq;
                }
            }
        }
    }
    let mut eig = [0.0f64; 6];
    for i in 0..n {
        eig[i] = a[i][i];
    }
    (eig, v)
}

/// Camera-pose twist Jacobian of the normalized reprojection at one fixed point.
/// Perturbation is left-multiplied in the camera frame: `p' = Rot(ω)·p + ν`, `δ = [ω(0..3), ν(3..6)]`.
/// Rows are `[∂u, ∂v]`, columns the 6 DOF. Returns `None` if the point is behind the camera.
fn reproj_jacobian(p_cam: Vec3F64) -> Option<[[f64; 6]; 2]> {
    if p_cam.z <= 1e-6 {
        return None;
    }
    let eps = 1e-6;
    let (u0, v0) = (p_cam.x / p_cam.z, p_cam.y / p_cam.z);
    let axes = [
        Vec3F64::new(1.0, 0.0, 0.0),
        Vec3F64::new(0.0, 1.0, 0.0),
        Vec3F64::new(0.0, 0.0, 1.0),
    ];
    let mut j = [[0.0f64; 6]; 2];
    for k in 0..6 {
        // dp = rotation (axis × p) for k<3, translation (unit axis) for k>=3.
        let dp = if k < 3 {
            axes[k].cross(p_cam)
        } else {
            axes[k - 3]
        };
        let p1 = Vec3F64::new(
            p_cam.x + eps * dp.x,
            p_cam.y + eps * dp.y,
            p_cam.z + eps * dp.z,
        );
        j[0][k] = (p1.x / p1.z - u0) / eps;
        j[1][k] = (p1.y / p1.z - v0) / eps;
    }
    Some(j)
}

/// Per-camera pose covariance + observability from each camera's fixed-point (board/gauge)
/// observations. One [`CameraStats`] per camera (registered or not).
pub(crate) fn camera_stats(
    cameras: &[PinholeCamera],
    poses: &[Pose3d],
    obs: &[BaObservation],
    points: &[Vec3F64],
    have: &[bool],
) -> Vec<CameraStats> {
    let n_cams = cameras.len();
    let mut out = Vec::with_capacity(n_cams);
    for c in 0..n_cams {
        // Accumulate the 6x6 pose Hessian over this camera's fixed-point observations.
        let mut m = [[0.0f64; 6]; 6];
        let mut sse = 0.0f64; // normalized residual sum of squares
        let mut se_px = 0.0f64; // pixel residual sum of squares
        let mut depth_sum = 0.0f64;
        let mut num = 0usize;
        for o in obs {
            if o.pose_idx != c || !o.fixed_point {
                continue;
            }
            let p_cam = poses[c].transform_point(&points[o.point_idx]);
            let Some(j) = reproj_jacobian(p_cam) else {
                continue;
            };
            for a in 0..6 {
                for b in 0..6 {
                    m[a][b] += j[0][a] * j[0][b] + j[1][a] * j[1][b];
                }
            }
            let (ru, rv) = (
                p_cam.x / p_cam.z - o.pixel[0] as f64,
                p_cam.y / p_cam.z - o.pixel[1] as f64,
            );
            sse += ru * ru + rv * rv;
            se_px += (ru * cameras[c].fx).powi(2) + (rv * cameras[c].fy).powi(2);
            depth_sum += p_cam.z.abs();
            num += 1;
        }

        if !have[c] || num * 2 <= 6 {
            out.push(CameraStats {
                camera: c,
                registered: have[c],
                num_obs: num,
                reproj_rmse_px: -1.0,
                rot_sigma_deg: f64::INFINITY,
                trans_sigma_m: f64::INFINITY,
                min_eigenvalue: 0.0,
                weakest_dof: [0.0; 6],
            });
            continue;
        }

        let dof = (2 * num - 6) as f64;
        let sigma_sq = (sse / dof).max(0.0);
        // Per-observation Euclidean RMS (divide by obs count, not scalar components) so this matches
        // the crate-wide `RigCalibration::reproj_rmse_px` convention.
        let reproj_rmse_px = (se_px / num as f64).sqrt();

        // Covariance Σ = σ̂²·M⁻¹ via eigendecomposition (skip near-null directions).
        let (eig, vec) = symmetric_eigen(m);
        let lam_max = eig.iter().cloned().fold(0.0f64, f64::max);
        let eps_lam = 1e-9 * lam_max.max(1e-30);
        let mut cov_diag = [0.0f64; 6];
        for k in 0..6 {
            if eig[k] <= eps_lam {
                continue; // unobservable direction — omit from the finite variance sum
            }
            let inv = sigma_sq / eig[k];
            for i in 0..6 {
                cov_diag[i] += inv * vec[i][k] * vec[i][k];
            }
        }
        let rot_sigma_deg = (cov_diag[0] + cov_diag[1] + cov_diag[2])
            .sqrt()
            .to_degrees();
        let trans_sigma_m = (cov_diag[3] + cov_diag[4] + cov_diag[5]).sqrt();

        // Observability: non-dimensionalize (scale translation rows/cols by the characteristic depth
        // L) so the eigenvalues are comparable, then take the weakest direction.
        let l = (depth_sum / num as f64).max(1e-6);
        let mut m_nd = m;
        for i in 0..6 {
            for j in 0..6 {
                let si = if i < 3 { 1.0 } else { l };
                let sj = if j < 3 { 1.0 } else { l };
                m_nd[i][j] *= si * sj;
            }
        }
        let (eig_nd, vec_nd) = symmetric_eigen(m_nd);
        let mut kmin = 0;
        for k in 1..6 {
            if eig_nd[k] < eig_nd[kmin] {
                kmin = k;
            }
        }
        let weakest_dof = [
            vec_nd[0][kmin],
            vec_nd[1][kmin],
            vec_nd[2][kmin],
            vec_nd[3][kmin],
            vec_nd[4][kmin],
            vec_nd[5][kmin],
        ];

        out.push(CameraStats {
            camera: c,
            registered: true,
            num_obs: num,
            reproj_rmse_px,
            rot_sigma_deg,
            trans_sigma_m,
            min_eigenvalue: eig_nd[kmin],
            weakest_dof,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eigen_of_diagonal_matrix() {
        let mut a = [[0.0f64; 6]; 6];
        let d = [5.0, 1.0, 9.0, 2.0, 7.0, 3.0];
        for i in 0..6 {
            a[i][i] = d[i];
        }
        let (eig, _) = symmetric_eigen(a);
        let mut got = eig.to_vec();
        got.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let mut want = d.to_vec();
        want.sort_by(|x, y| x.partial_cmp(y).unwrap());
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-9, "eig {g} vs {w}");
        }
    }

    #[test]
    fn eigen_recovers_known_symmetric() {
        // 2x2 block [[2,1],[1,2]] → eigenvalues 1 and 3.
        let mut a = [[0.0f64; 6]; 6];
        a[0][0] = 2.0;
        a[1][1] = 2.0;
        a[0][1] = 1.0;
        a[1][0] = 1.0;
        for i in 2..6 {
            a[i][i] = 4.0;
        }
        let (eig, _) = symmetric_eigen(a);
        let mut got = eig.to_vec();
        got.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert!((got[0] - 1.0).abs() < 1e-9);
        assert!((got[1] - 3.0).abs() < 1e-9);
    }
}
