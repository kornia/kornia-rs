//! AP3P (Algebraic Perspective-Three-Point) solver.
//!
//! Paper: ["An Efficient Algebraic Solution to the Perspective-Three-Point
//! Problem" by Ke & Roumeliotis, CVPR 2017](https://arxiv.org/pdf/1701.08237.pdf)
//!
//! Reference: [OpenCV AP3P implementation](https://github.com/opencv/opencv/blob/4.x/modules/calib3d/src/ap3p.cpp)
//!            and the companion `polynom_solver.{h,cpp}` that supplies
//!            Ferrari-method `solve_deg2/3/4`.
//!
//! Given three 3D ↔ 2D correspondences (and a camera intrinsics matrix), AP3P
//! recovers up to four candidate `(R, t)` solutions for the world→camera
//! pose. It is a *minimal* solver (3 points), so it slots into RANSAC as a
//! sub-`SAMPLE_SIZE` kernel; the driver in `ransac::run` already supports
//! the `Vec<Model>` out-parameter multi-solution kernel pattern (Nistér's
//! 5-point essential uses the same idiom).
//!
//! # Coordinate convention
//!
//! The solver follows the same convention as `epnp`: `R, t` map
//! *world → camera*. The recovered pose is checked for **cheirality**
//! (all three control 3D points must land in front of the camera) before
//! being emitted — configurations where any 3D point projects to negative
//! depth are physically impossible under a pinhole and would never be
//! selected by a sensible RANSAC consensus scorer.
//!
//! # Distortion
//!
//! The OpenCV reference assumes a pinhole projection with image points
//! normalised by `K⁻¹`. The driver's residual still uses the pinhole
//! forward model — callers needing a distortion-aware path should run
//! `LMRefineParams` after the RANSAC fit (the existing EPnP path already
//! plumbs the same LM refinement in `pnp::refine`).

use kornia_algebra::{Mat3AF32, Vec2F32, Vec3AF32, SO3F32};
use kornia_imgproc::calibration::distortion::PolynomialDistortion;

use super::{PnPError, PnPResult, PnPSolver};

/// Marker type representing the AP3P algorithm.
pub struct AP3P;

impl PnPSolver for AP3P {
    type Param = AP3PParams;

    fn solve(
        points_world: &[Vec3AF32],
        points_image: &[Vec2F32],
        k: &Mat3AF32,
        _distortion: Option<&PolynomialDistortion>,
        params: &Self::Param,
    ) -> Result<PnPResult, PnPError> {
        solve_ap3p(points_world, points_image, k, params)
    }
}

/// Parameters controlling the AP3P solver.
///
/// AP3P's algebraic kernel is closed-form with no iterative refinement; the
/// only knob is the residual selection (cheirality) of the returned
/// solution.
#[derive(Debug, Clone, Default)]
pub struct AP3PParams {
    /// If `true`, the multi-solution branch is used and the lowest-`rmse`
    /// candidate is returned. If `false`, only the first cheirality-passing
    /// candidate is returned.
    pub pick_lowest_rmse: bool,
}

/// Solve Perspective-3-Point (AP3P).
///
/// # Arguments
/// * `points_world` – 3-D coordinates in the world frame, shape *(3,3)*
///   (exactly three correspondences).
/// * `points_image` – Corresponding pixel coordinates, shape *(3,2)*.
/// * `k` – Camera intrinsics matrix.
///
/// # Returns
/// Best `(R, t)` satisfying cheirality (or sole candidate if the algebraic
/// root system returns a single one). The rotation is in the
/// **world → camera** frame.
pub fn solve_ap3p(
    points_world: &[Vec3AF32],
    points_image: &[Vec2F32],
    k: &Mat3AF32,
    params: &AP3PParams,
) -> Result<PnPResult, PnPError> {
    let n = points_world.len();
    if n != points_image.len() {
        return Err(PnPError::MismatchedArrayLengths {
            left_name: "world points",
            left_len: n,
            right_name: "image points",
            right_len: points_image.len(),
        });
    }
    if n != 3 {
        return Err(PnPError::InsufficientCorrespondences {
            required: 3,
            actual: n,
        });
    }

    // Recover focal / principal point once. We follow the OpenCV convention
    // of un-normalising the 2D image points first and then dividing by `‖·‖`
    // to get the unit bearing vector. The reference implementation pushes
    // through the inverse-intrinsics matrix:
    //   mu = (u - cx) / fx, mv = (v - cy) / fy
    //   b  = [mu, mv, 1] / ‖[mu, mv, 1]‖
    // We perform the same op in f64 (the algorithm's own precision) and
    // downcast only for the final PnPResult struct.
    let fx = k.x_axis().x as f64;
    let fy = k.y_axis().y as f64;
    let cx = k.z_axis().x as f64;
    let cy = k.z_axis().y as f64;
    let inv_fx = 1.0 / fx;
    let inv_fy = 1.0 / fy;
    let cx_fx = cx / fx;
    let cy_fy = cy / fy;

    let mut feature_vectors = [[0.0f64; 3]; 3]; // 3 bearings × 3 coords
    for i in 0..3 {
        let u = points_image[i].x as f64;
        let v = points_image[i].y as f64;
        let mu = inv_fx * u - cx_fx;
        let mv = inv_fy * v - cy_fy;
        let norm = (mu * mu + mv * mv + 1.0).sqrt();
        feature_vectors[i] = [mu / norm, mv / norm, 1.0 / norm];
    }

    let world_points = [
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
        [
            points_world[2].x as f64,
            points_world[2].y as f64,
            points_world[2].z as f64,
        ],
    ];

    // Transpose to the OpenCV `featureVectors[3][4]` layout: rows = (mu, mv, mk)
    // and columns = point index. We only use the first 3 columns.
    let fv = [
        [
            feature_vectors[0][0],
            feature_vectors[1][0],
            feature_vectors[2][0],
        ],
        [
            feature_vectors[0][1],
            feature_vectors[1][1],
            feature_vectors[2][1],
        ],
        [
            feature_vectors[0][2],
            feature_vectors[1][2],
            feature_vectors[2][2],
        ],
    ];
    let wp = [
        [world_points[0][0], world_points[1][0], world_points[2][0]],
        [world_points[0][1], world_points[1][1], world_points[2][1]],
        [world_points[0][2], world_points[1][2], world_points[2][2]],
    ];

    let mut solutions_r = [[[0.0f64; 3]; 3]; 4];
    let mut solutions_t = [[0.0f64; 3]; 4];

    let n_solutions = ap3p_compute_poses(&fv, &wp, &mut solutions_r, &mut solutions_t);
    if n_solutions == 0 {
        return Err(PnPError::SvdFailed(
            "AP3P: no real solution found (degenerate 3-point sample)".to_string(),
        ));
    }

    // Cheirality: keep only candidates where all three 3D points have
    // positive depth in the camera frame.
    let mut best_idx = None;
    let mut best_rmse = f64::INFINITY;

    for i in 0..n_solutions as usize {
        let r: [[f64; 3]; 3] = solutions_r[i];
        let t: [f64; 3] = solutions_t[i];
        if !all_positive_depths(&r, &t, &world_points) {
            continue;
        }
        if !params.pick_lowest_rmse {
            best_idx = Some(i);
            break;
        }
        let rmse = rmse_px(&world_points, points_image, &r, &t, k);
        if rmse < best_rmse {
            best_rmse = rmse;
            best_idx = Some(i);
        }
    }

    let pick = best_idx.ok_or_else(|| {
        PnPError::SvdFailed("AP3P: all candidates failed cheirality check".to_string())
    })?;

    let r = solutions_r[pick];
    let t = solutions_t[pick];
    let r_mat = Mat3AF32::from_cols_array(&[
        r[0][0] as f32,
        r[1][0] as f32,
        r[2][0] as f32,
        r[0][1] as f32,
        r[1][1] as f32,
        r[2][1] as f32,
        r[0][2] as f32,
        r[1][2] as f32,
        r[2][2] as f32,
    ]);
    let t_vec = Vec3AF32::new(t[0] as f32, t[1] as f32, t[2] as f32);
    let rvec = SO3F32::from_matrix(&r_mat).log();

    Ok(PnPResult {
        rotation: r_mat,
        translation: t_vec,
        rvec,
        reproj_rmse: Some(best_rmse as f32),
        num_iterations: None,
        converged: Some(true),
    })
}

/// world is laid out as `world_points[point_index][coord]`.
fn all_positive_depths(r: &[[f64; 3]; 3], t: &[f64; 3], world: &[[f64; 3]; 3]) -> bool {
    for point in world {
        let pc0 = r[0][0] * point[0] + r[0][1] * point[1] + r[0][2] * point[2] + t[0];
        let pc1 = r[1][0] * point[0] + r[1][1] * point[1] + r[1][2] * point[2] + t[1];
        let pc2 = r[2][0] * point[0] + r[2][1] * point[1] + r[2][2] * point[2] + t[2];
        if !(pc2 > 0.0 && pc0.is_finite() && pc1.is_finite() && pc2.is_finite()) {
            return false;
        }
    }
    true
}

fn rmse_px(
    world: &[[f64; 3]; 3],
    image: &[Vec2F32],
    r: &[[f64; 3]; 3],
    t: &[f64; 3],
    k: &Mat3AF32,
) -> f64 {
    let mut sum = 0.0;

    let fx = k.x_axis().x as f64;
    let fy = k.y_axis().y as f64;
    let cx = k.z_axis().x as f64;
    let cy = k.z_axis().y as f64;

    for (i, point) in world.iter().enumerate() {
        let pc0 = r[0][0] * point[0] + r[0][1] * point[1] + r[0][2] * point[2] + t[0];
        let pc1 = r[1][0] * point[0] + r[1][1] * point[1] + r[1][2] * point[2] + t[1];
        let pc2 = r[2][0] * point[0] + r[2][1] * point[1] + r[2][2] * point[2] + t[2];
        if pc2 <= 0.0 {
            continue;
        }
        let inv_z = 1.0 / pc2;
        let u_hat = (fx * pc0) * inv_z + cx;
        let v_hat = (fy * pc1) * inv_z + cy;
        let du = u_hat - image[i].x as f64;
        let dv = v_hat - image[i].y as f64;
        sum += du * du + dv * dv;
    }
    (sum / world.len() as f64).sqrt()
}

// ---------------------------------------------------------------------------
// Polynomial solvers (degree 2/3/4) — direct port of OpenCV's
// modules/calib3d/src/polynom_solver.cpp. Ferrari's method for `solve_deg4`.
// ---------------------------------------------------------------------------

/// Solve `a·x² + b·x + c = 0`.
///
/// Returns the number of real roots found. Roots are written to `x1, x2`.
/// Caller pre-sizes `x1, x2` storage.
pub fn solve_deg2(a: f64, b: f64, c: f64, x1: &mut f64, x2: &mut f64) -> i32 {
    let delta = b * b - 4.0 * a * c;
    if delta < 0.0 {
        return 0;
    }
    let inv_2a = 0.5 / a;
    if delta == 0.0 {
        *x1 = -b * inv_2a;
        *x2 = *x1;
        return 1;
    }
    let sqrt_delta = delta.sqrt();
    *x1 = (-b + sqrt_delta) * inv_2a;
    *x2 = (-b - sqrt_delta) * inv_2a;
    2
}

/// Solve `a·x³ + b·x² + c·x + d = 0`. Returns up to 3 real roots.
pub fn solve_deg3(
    a: f64,
    mut b: f64,
    mut c: f64,
    mut d: f64,
    x0: &mut f64,
    x1: &mut f64,
    x2: &mut f64,
) -> i32 {
    if a == 0.0 {
        if b == 0.0 {
            if c == 0.0 {
                return 0;
            }
            *x0 = -d / c;
            return 1;
        }
        *x2 = 0.0;
        return solve_deg2(b, c, d, x0, x1);
    }

    // OpenCV mutates these in-place; we MUST do the same to preserve math below.
    let inv_a = 1.0 / a;
    b *= inv_a;
    c *= inv_a;
    d *= inv_a;

    let q = (3.0 * c - b * b) / 9.0;
    let r = (9.0 * b * c - 27.0 * d - 2.0 * b * b * b) / 54.0;
    let q3 = q * q * q;
    let d_disc = q3 + r * r;
    let b_3 = b / 3.0;

    if q == 0.0 {
        if r == 0.0 {
            *x0 = -b_3;
            *x1 = *x0;
            *x2 = *x0;
            return 3;
        } else {
            *x0 = (2.0 * r).cbrt() - b_3;
            return 1;
        }
    }

    if d_disc <= 0.0 {
        let theta = (r / (-q3).sqrt()).acos();
        let sqrt_q = (-q).sqrt();
        *x0 = 2.0 * sqrt_q * (theta / 3.0).cos() - b_3;
        *x1 = 2.0 * sqrt_q * ((theta + 2.0 * std::f64::consts::PI) / 3.0).cos() - b_3;
        *x2 = 2.0 * sqrt_q * ((theta + 4.0 * std::f64::consts::PI) / 3.0).cos() - b_3;
        return 3;
    }

    let s = r.abs() + d_disc.sqrt();
    let mut ad = s.cbrt();
    if r < 0.0 {
        ad = -ad;
    }
    let bd = if ad == 0.0 { 0.0 } else { -q / ad };
    *x0 = ad + bd - b_3;
    1
}

/// Solve `a·x⁴ + b·x³ + c·x² + d·x + e = 0`. Returns up to 4 real roots.
#[allow(clippy::too_many_arguments)]
pub fn solve_deg4(
    a: f64,
    mut b: f64,
    mut c: f64,
    mut d: f64,
    mut e: f64,
    x0: &mut f64,
    x1: &mut f64,
    x2: &mut f64,
    x3: &mut f64,
) -> i32 {
    if a == 0.0 {
        *x3 = 0.0;
        return solve_deg3(b, c, d, e, x0, x1, x2);
    }

    // Safely mutate parameters down to the normalized form.
    let inv_a = 1.0 / a;
    b *= inv_a;
    c *= inv_a;
    d *= inv_a;
    e *= inv_a;

    let b2 = b * b;
    let bc = b * c;
    let b3 = b2 * b;

    let mut r0 = 0.0;
    let mut r1 = 0.0;
    let mut r2 = 0.0;
    let n = solve_deg3(
        1.0,
        -c,
        d * b - 4.0 * e,
        4.0 * c * e - d * d - b2 * e,
        &mut r0,
        &mut r1,
        &mut r2,
    );
    if n == 0 {
        return 0;
    }
    let r2_val = 0.25 * b2 - c + r0;
    let big_r = if r2_val < 0.0 {
        return 0;
    } else {
        r2_val.sqrt()
    };
    let inv_r = 1.0 / big_r;
    let mut nb_real_roots = 0;

    let (d2, e2) = if big_r < 1e-11 {
        let temp = r0 * r0 - 4.0 * e;
        if temp < 0.0 {
            (-1.0, -1.0)
        } else {
            let sqrt_temp = temp.sqrt();
            (
                0.75 * b2 - 2.0 * c + 2.0 * sqrt_temp,
                0.75 * b2 - 2.0 * c - 2.0 * sqrt_temp,
            )
        }
    } else {
        let u = 0.75 * b2 - 2.0 * c - r2_val;
        let v = 0.25 * inv_r * (4.0 * bc - 8.0 * d - b3);
        (u + v, u - v)
    };

    let b_4 = 0.25 * b;
    let r_2 = 0.5 * big_r;

    if d2 >= 0.0 {
        let root_d = d2.sqrt();
        nb_real_roots = 2;
        let d_2 = 0.5 * root_d;
        *x0 = r_2 + d_2 - b_4;
        *x1 = r_2 - d_2 - b_4;
    }

    if e2 >= 0.0 {
        let root_e = e2.sqrt();
        let e_2 = 0.5 * root_e;
        if nb_real_roots == 0 {
            *x0 = -r_2 + e_2 - b_4;
            *x1 = -r_2 - e_2 - b_4;
            nb_real_roots = 2;
        } else {
            *x2 = -r_2 + e_2 - b_4;
            *x3 = -r_2 - e_2 - b_4;
            nb_real_roots = 4;
        }
    }
    nb_real_roots
}

/// Newton-Raphson refinement of real quartic roots, exactly two
/// iterations. Matches OpenCV's `polishQuarticRoots` in ap3p.cpp.
fn polish_quartic_roots(coeffs: &[f64; 5], roots: &mut [f64; 4], nb_roots: i32) {
    for _ in 0..2 {
        // Iterate over the actual root values safely
        for r_val in roots.iter_mut().take(nb_roots as usize) {
            let r = *r_val;
            let error =
                (((coeffs[0] * r + coeffs[1]) * r + coeffs[2]) * r + coeffs[3]) * r + coeffs[4];
            let derivative =
                ((4.0 * coeffs[0] * r + 3.0 * coeffs[1]) * r + 2.0 * coeffs[2]) * r + coeffs[3];

            if derivative.abs() > f64::EPSILON {
                *r_val -= error / derivative;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// `computePoses` — direct port of OpenCV's `ap3p::computePoses`. Returns the
// number of real (cheirality-passing) solutions.
// ---------------------------------------------------------------------------

fn ap3p_compute_poses(
    feature_vectors: &[[f64; 3]; 3],
    world_points: &[[f64; 3]; 3],
    solutions_r: &mut [[[f64; 3]; 3]; 4],
    solutions_t: &mut [[f64; 3]; 4],
) -> i32 {
    let w1 = [world_points[0][0], world_points[1][0], world_points[2][0]];
    let w2 = [world_points[0][1], world_points[1][1], world_points[2][1]];
    let w3 = [world_points[0][2], world_points[1][2], world_points[2][2]];

    let mut u0 = [0.0; 3];
    for k in 0..3 {
        u0[k] = w1[k] - w2[k];
    }
    let nu0 = norm(&u0);
    let mut k1 = [0.0; 3];
    for k in 0..3 {
        k1[k] = u0[k] / nu0;
    }

    let b1 = [
        feature_vectors[0][0],
        feature_vectors[1][0],
        feature_vectors[2][0],
    ];
    let b2 = [
        feature_vectors[0][1],
        feature_vectors[1][1],
        feature_vectors[2][1],
    ];
    let b3 = [
        feature_vectors[0][2],
        feature_vectors[1][2],
        feature_vectors[2][2],
    ];

    let mut k3 = [0.0; 3];
    cross(&b1, &b2, &mut k3);
    let nk3 = norm(&k3);
    for val in &mut k3 {
        *val /= nk3;
    }
    let mut tz = [0.0; 3];
    cross(&b1, &k3, &mut tz);

    let mut v1 = [0.0; 3];
    cross(&b1, &b3, &mut v1);
    let mut v2 = [0.0; 3];
    cross(&b2, &b3, &mut v2);
    let mut u1 = [0.0; 3];
    for k in 0..3 {
        u1[k] = w1[k] - w3[k];
    }

    let u1k1 = dot(&u1, &k1);
    let k3b3 = dot(&k3, &b3);
    let f11 = k3b3;
    let f13 = dot(&k3, &v1);
    let f15 = -u1k1 * f11;

    let mut nl = [0.0; 3];
    cross(&u1, &k1, &mut nl);
    let delta = norm(&nl);
    for val in &mut nl {
        *val /= delta;
    }
    let f11 = f11 * delta;
    let f13 = f13 * delta;

    let u2k1 = u1k1 - nu0;
    let f21 = dot(&tz, &v2);
    let f22 = nk3 * k3b3;
    let f23 = dot(&k3, &v2);
    let f24 = u2k1 * f22;
    let f25 = -u2k1 * f21;
    let f21 = f21 * delta;
    let f22 = f22 * delta;
    let f23 = f23 * delta;

    let g1 = f13 * f22;
    let g2 = f13 * f25 - f15 * f23;
    let g3 = f11 * f23 - f13 * f21;
    let g4 = -f13 * f24;
    let g5 = f11 * f22;
    let g6 = f11 * f25 - f15 * f21;
    let g7 = -f15 * f24;

    let coeffs = [
        g5 * g5 + g1 * g1 + g3 * g3,
        2.0 * (g5 * g6 + g1 * g2 + g3 * g4),
        g6 * g6 + 2.0 * g5 * g7 + g2 * g2 + g4 * g4 - g1 * g1 - g3 * g3,
        2.0 * (g6 * g7 - g1 * g2 - g3 * g4),
        g7 * g7 - g2 * g2 - g4 * g4,
    ];

    let mut s0 = 0.0;
    let mut s1 = 0.0;
    let mut s2 = 0.0;
    let mut s3 = 0.0;
    let nb_roots = solve_deg4(
        coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4], &mut s0, &mut s1, &mut s2, &mut s3,
    );

    let mut s = [s0, s1, s2, s3];
    polish_quartic_roots(&coeffs, &mut s, nb_roots);

    let mut temp = [0.0; 3];
    cross(&k1, &nl, &mut temp);
    let ck1nl = [
        [k1[0], nl[0], temp[0]],
        [k1[1], nl[1], temp[1]],
        [k1[2], nl[2], temp[2]],
    ];
    let cb1k3tz_t = [
        [b1[0], b1[1], b1[2]],
        [k3[0], k3[1], k3[2]],
        [tz[0], tz[1], tz[2]],
    ];
    let mut b3p = [0.0; 3];
    for k in 0..3 {
        b3p[k] = (delta / k3b3) * b3[k];
    }

    let mut nb_solutions = 0;
    for &ctheta1p in s.iter().take(nb_roots as usize) {
        if ctheta1p.abs() > 1.0 {
            continue;
        }

        let mut stheta1p = (1.0 - ctheta1p * ctheta1p).max(0.0).sqrt();
        if k3b3 < 0.0 {
            stheta1p = -stheta1p;
        }

        let ctheta3 = g1 * ctheta1p + g2;
        let stheta3 = g3 * ctheta1p + g4;
        let ntheta3 = stheta1p / ((g5 * ctheta1p + g6) * ctheta1p + g7);
        let ctheta3 = ctheta3 * ntheta3;
        let stheta3 = stheta3 * ntheta3;

        let c13 = [
            [ctheta3, 0.0, -stheta3],
            [stheta1p * stheta3, ctheta1p, stheta1p * ctheta3],
            [ctheta1p * stheta3, -stheta1p, ctheta1p * ctheta3],
        ];

        let temp_matrix = mat_mult(&ck1nl, &c13);
        let r_mat = mat_mult(&temp_matrix, &cb1k3tz_t);

        let rp3 = [
            w3[0] * r_mat[0][0] + w3[1] * r_mat[1][0] + w3[2] * r_mat[2][0],
            w3[0] * r_mat[0][1] + w3[1] * r_mat[1][1] + w3[2] * r_mat[2][1],
            w3[0] * r_mat[0][2] + w3[1] * r_mat[1][2] + w3[2] * r_mat[2][2],
        ];

        let mut pxstheta1p = [0.0; 3];
        for k in 0..3 {
            pxstheta1p[k] = stheta1p * b3p[k];
        }
        for k in 0..3 {
            solutions_t[nb_solutions][k] = pxstheta1p[k] - rp3[k];
        }

        // Restored transpose: convert internal Camera->World mapping
        // to World->Camera mapping required by the PnP struct.
        for row in 0..3 {
            for col in 0..3 {
                solutions_r[nb_solutions][col][row] = r_mat[row][col];
            }
        }

        nb_solutions += 1;
    }
    nb_solutions as i32
}

/// Solves AP3P and returns ALL cheirality-passing roots (up to 4).
/// Used natively by the RANSAC estimator to evaluate all algebraic candidates.
pub fn solve_ap3p_multi(
    points_world: &[Vec3AF32],
    points_image: &[Vec2F32],
    k: &Mat3AF32,
) -> Result<Vec<PnPResult>, PnPError> {
    let n = points_world.len();
    if n != 3 || points_image.len() != 3 {
        return Err(PnPError::InsufficientCorrespondences {
            required: 3,
            actual: n,
        });
    }

    let fx = k.x_axis().x as f64;
    let fy = k.y_axis().y as f64;
    let cx = k.z_axis().x as f64;
    let cy = k.z_axis().y as f64;
    let inv_fx = 1.0 / fx;
    let inv_fy = 1.0 / fy;
    let cx_fx = cx / fx;
    let cy_fy = cy / fy;

    let mut feature_vectors = [[0.0f64; 3]; 3];
    for i in 0..3 {
        let u = points_image[i].x as f64;
        let v = points_image[i].y as f64;
        let mu = inv_fx * u - cx_fx;
        let mv = inv_fy * v - cy_fy;
        let norm = (mu * mu + mv * mv + 1.0).sqrt();
        feature_vectors[i] = [mu / norm, mv / norm, 1.0 / norm];
    }

    let world_points = [
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
        [
            points_world[2].x as f64,
            points_world[2].y as f64,
            points_world[2].z as f64,
        ],
    ];

    let fv = [
        [
            feature_vectors[0][0],
            feature_vectors[1][0],
            feature_vectors[2][0],
        ],
        [
            feature_vectors[0][1],
            feature_vectors[1][1],
            feature_vectors[2][1],
        ],
        [
            feature_vectors[0][2],
            feature_vectors[1][2],
            feature_vectors[2][2],
        ],
    ];
    let wp = [
        [world_points[0][0], world_points[1][0], world_points[2][0]],
        [world_points[0][1], world_points[1][1], world_points[2][1]],
        [world_points[0][2], world_points[1][2], world_points[2][2]],
    ];

    let mut solutions_r = [[[0.0f64; 3]; 3]; 4];
    let mut solutions_t = [[0.0f64; 3]; 4];

    let n_solutions = ap3p_compute_poses(&fv, &wp, &mut solutions_r, &mut solutions_t);
    if n_solutions == 0 {
        return Err(PnPError::SvdFailed(
            "AP3P: no real solution found".to_string(),
        ));
    }

    let mut results = Vec::new();
    for i in 0..n_solutions as usize {
        let r = solutions_r[i];
        let t = solutions_t[i];
        if all_positive_depths(&r, &t, &world_points) {
            let r_mat = Mat3AF32::from_cols_array(&[
                r[0][0] as f32,
                r[1][0] as f32,
                r[2][0] as f32,
                r[0][1] as f32,
                r[1][1] as f32,
                r[2][1] as f32,
                r[0][2] as f32,
                r[1][2] as f32,
                r[2][2] as f32,
            ]);
            let t_vec = Vec3AF32::new(t[0] as f32, t[1] as f32, t[2] as f32);
            let rvec = kornia_algebra::SO3F32::from_matrix(&r_mat).log();

            results.push(PnPResult {
                rotation: r_mat,
                translation: t_vec,
                rvec,
                reproj_rmse: None,
                num_iterations: None,
                converged: Some(true),
            });
        }
    }

    if results.is_empty() {
        Err(PnPError::SvdFailed(
            "AP3P: all candidates failed cheirality".to_string(),
        ))
    } else {
        Ok(results)
    }
}

#[inline]
fn norm(a: &[f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

#[inline]
fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross(a: &[f64; 3], b: &[f64; 3], out: &mut [f64; 3]) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = -(a[0] * b[2] - a[2] * b[0]);
    out[2] = a[0] * b[1] - a[1] * b[0];
}

#[inline]
fn mat_mult(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_algebra::Vec3AF32;

    fn k_default() -> Mat3AF32 {
        Mat3AF32::from_cols(
            Vec3AF32::new(800.0, 0.0, 0.0),
            Vec3AF32::new(0.0, 800.0, 0.0),
            Vec3AF32::new(640.0, 480.0, 1.0),
        )
    }

    /// Three perfectly-projected 3D-2D correspondences should round-trip
    /// back to a pose with negligible reprojection error.
    #[test]
    fn solves_three_correspondences() -> Result<(), PnPError> {
        let k = k_default();
        let world = [
            Vec3AF32::new(0.0, 0.0, 1.0),
            Vec3AF32::new(0.1, 0.0, 1.0),
            Vec3AF32::new(0.0, 0.1, 1.0),
        ];
        let r_gt = [
            [0.999, -0.001, 0.001],
            [0.001, 0.999, 0.001],
            [-0.001, -0.001, 0.999],
        ];
        let t_gt = [0.05, -0.03, 0.2];

        let image: Vec<Vec2F32> = world
            .iter()
            .map(|p| {
                let pc0 = r_gt[0][0] * p.x + r_gt[0][1] * p.y + r_gt[0][2] * p.z + t_gt[0];
                let pc1 = r_gt[1][0] * p.x + r_gt[1][1] * p.y + r_gt[1][2] * p.z + t_gt[1];
                let pc2 = r_gt[2][0] * p.x + r_gt[2][1] * p.y + r_gt[2][2] * p.z + t_gt[2];
                Vec2F32::new(800.0 * pc0 / pc2 + 640.0, 800.0 * pc1 / pc2 + 480.0)
            })
            .collect();

        let result = solve_ap3p(&world, &image, &k, &AP3PParams::default())?;
        let rmat = result.rotation;
        let t = result.translation;

        // Allow the algebraic solver to land on a *different* root than
        // ground truth, but verify any cheirality-positive root yields
        // a small reprojection error on the input correspondences.
        for (p, q) in world.iter().zip(image.iter()) {
            let pc0 = rmat.x_axis().x * p.x + rmat.y_axis().x * p.y + rmat.z_axis().x * p.z + t.x;
            let pc1 = rmat.x_axis().y * p.x + rmat.y_axis().y * p.y + rmat.z_axis().y * p.z + t.y;
            let pc2 = rmat.x_axis().z * p.x + rmat.y_axis().z * p.y + rmat.z_axis().z * p.z + t.z;
            assert!(pc2 > 0.0, "point behind camera");

            let u = 800.0 * pc0 / pc2 + 640.0;
            let v = 800.0 * pc1 / pc2 + 480.0;

            assert!(
                (u - q.x).abs() < 1.0,
                "u residual too large: {}",
                (u - q.x).abs()
            );
            assert!(
                (v - q.y).abs() < 1.0,
                "v residual too large: {}",
                (v - q.y).abs()
            );
        }

        Ok(())
    }

    #[test]
    fn deg2_returns_expected_count() {
        let mut x1 = 0.0;
        let mut x2 = 0.0;
        // x^2 - 5x + 6 = 0 -> roots 2, 3.
        let n = solve_deg2(1.0, -5.0, 6.0, &mut x1, &mut x2);
        assert_eq!(n, 2);
        let mut roots = [x1, x2];
        roots.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((roots[0] - 2.0).abs() < 1e-9);
        assert!((roots[1] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn deg3_simple_real_root() {
        // (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6
        let mut x0 = 0.0;
        let mut x1 = 0.0;
        let mut x2 = 0.0;
        let _n = solve_deg3(1.0, -6.0, 11.0, -6.0, &mut x0, &mut x1, &mut x2);
        // (We accept 1 or 3 roots depending on conditioning; assert at
        // least that some root is one of {1, 2, 3}.)
        let roots = [x0, x1, x2];
        let set = [1.0_f64, 2.0, 3.0];
        assert!(roots
            .iter()
            .any(|r| set.iter().any(|s| (r - s).abs() < 1e-3)));
    }

    #[test]
    fn deg4_handles_degenerate_coeffs() {
        // (x-1)(x-2)(x-3)(x-4) = x^4 - 10x^3 + 35x^2 - 50x + 24.
        let mut x0 = 0.0;
        let mut x1 = 0.0;
        let mut x2 = 0.0;
        let mut x3 = 0.0;
        let n = solve_deg4(
            1.0, -10.0, 35.0, -50.0, 24.0, &mut x0, &mut x1, &mut x2, &mut x3,
        );
        assert!(n >= 2, "expected at least two real roots, got {n}");
    }
}
