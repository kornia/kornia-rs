//! `kornia_rs.k3d.bundle_adjust` — joint pose + point optimization.
//!
//! Thin wrapper over `kornia_3d::ba::bundle_adjust`. Uses LM with analytical
//! Jacobians; intrinsics are pinhole (no distortion). Pose 0 is held fixed
//! by default to remove the gauge freedom.

use kornia_3d::ba::{bundle_adjust as rs_bundle_adjust, BaObservation, BaParams};
use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_3d::ransac::RobustKernelKind;
use kornia_algebra::{Mat3F64, Vec3F64};
use numpy::{PyArray, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;

/// Build a `Pose3d` from a (3, 3) row-major rotation + (3,) translation slice.
fn pose_from_slice(r: &[f64], t: &[f64]) -> Pose3d {
    // numpy row-major → glam column-major Mat3F64
    let rotation = Mat3F64::from_cols(
        Vec3F64::new(r[0], r[3], r[6]),
        Vec3F64::new(r[1], r[4], r[7]),
        Vec3F64::new(r[2], r[5], r[8]),
    );
    Pose3d::new(rotation, Vec3F64::new(t[0], t[1], t[2]))
}

/// Joint LM optimization over poses and 3D points.
///
/// Args:
///     rotations: `(P, 3, 3)` float64 — world→camera rotation per pose.
///     translations: `(P, 3)` float64 — world→camera translation per pose.
///     points: `(N, 3)` float64 — initial 3D point positions (world frame).
///     observations: `(M, 4)` float64 — each row `[pose_idx, point_idx, u, v]`
///         where `pose_idx`/`point_idx` are integer-valued. Undistorted pixels.
///     k: `(3, 3)` float64 pinhole intrinsics. Distortion is ignored.
///     fixed_pose_indices: optional list of pose indices to hold fixed
///         (default: `[0]` to remove gauge freedom).
///     max_iterations: LM iteration cap (default 10).
///     robust: `"identity" | "huber" | "cauchy"` (default `"identity"`).
///     robust_scale: linear scale for the robust kernel in pixels
///         (default 1.0; ignored for identity).
///
/// Returns:
///     `(rotations_opt, translations_opt, points_opt, iterations, converged)`.
#[pyfunction(name = "bundle_adjust")]
#[pyo3(signature = (
    rotations, translations, points, observations, k,
    fixed_pose_indices=None, fix_all_points=false,
    max_iterations=10,
    robust="identity", robust_scale=1.0,
))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn bundle_adjust_py<'py>(
    py: Python<'py>,
    rotations: Bound<'py, PyArray<f64, numpy::Ix3>>,
    translations: Bound<'py, PyArray2<f64>>,
    points: Bound<'py, PyArray2<f64>>,
    observations: Bound<'py, PyArray2<f64>>,
    k: Bound<'py, PyArray2<f64>>,
    fixed_pose_indices: Option<Vec<usize>>,
    fix_all_points: bool,
    max_iterations: usize,
    robust: &str,
    robust_scale: f32,
) -> PyResult<(
    Bound<'py, PyArray<f64, numpy::Ix3>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    usize,
    bool,
)> {
    let r_shape = rotations.shape();
    let t_shape = translations.shape();
    let p_shape = points.shape();
    let o_shape = observations.shape();
    let k_shape = k.shape();

    if r_shape.len() != 3 || r_shape[1] != 3 || r_shape[2] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "rotations must be (P, 3, 3) float64",
        ));
    }
    if t_shape.len() != 2 || t_shape[1] != 3 || t_shape[0] != r_shape[0] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "translations must be (P, 3) and match rotations[0]",
        ));
    }
    if p_shape.len() != 2 || p_shape[1] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "points must be (N, 3) float64",
        ));
    }
    if o_shape.len() != 2 || o_shape[1] != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "observations must be (M, 4) float64: [pose_idx, point_idx, u, v]",
        ));
    }
    if k_shape.len() != 2 || k_shape[0] != 3 || k_shape[1] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "k must be (3, 3) float64",
        ));
    }
    let n_poses = r_shape[0];
    let n_points = p_shape[0];
    let n_obs = o_shape[0];

    // Unpack poses
    let r_data = unsafe { std::slice::from_raw_parts(rotations.data(), n_poses * 9) };
    let t_data = unsafe { std::slice::from_raw_parts(translations.data(), n_poses * 3) };
    let mut poses: Vec<Pose3d> = Vec::with_capacity(n_poses);
    for i in 0..n_poses {
        poses.push(pose_from_slice(&r_data[i * 9..i * 9 + 9], &t_data[i * 3..i * 3 + 3]));
    }

    // Unpack points
    let p_data = unsafe { std::slice::from_raw_parts(points.data(), n_points * 3) };
    let points_vec: Vec<Vec3F64> = (0..n_points)
        .map(|i| Vec3F64::new(p_data[i * 3], p_data[i * 3 + 1], p_data[i * 3 + 2]))
        .collect();

    // Unpack observations
    let fixed: std::collections::HashSet<usize> =
        fixed_pose_indices.unwrap_or_else(|| vec![0]).into_iter().collect();
    let o_data = unsafe { std::slice::from_raw_parts(observations.data(), n_obs * 4) };
    let mut obs_vec: Vec<BaObservation> = Vec::with_capacity(n_obs);
    for i in 0..n_obs {
        let pose_idx = o_data[i * 4] as usize;
        let point_idx = o_data[i * 4 + 1] as usize;
        if pose_idx >= n_poses || point_idx >= n_points {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "observation {i} has out-of-range pose_idx={pose_idx} or point_idx={point_idx}"
            )));
        }
        obs_vec.push(BaObservation {
            pose_idx,
            point_idx,
            pixel: [o_data[i * 4 + 2] as f32, o_data[i * 4 + 3] as f32],
            fixed_pose: fixed.contains(&pose_idx),
            fixed_point: fix_all_points,
        });
    }

    // Camera
    let k_data = unsafe { std::slice::from_raw_parts(k.data(), 9) };
    let camera = PinholeCamera {
        fx: k_data[0],
        fy: k_data[4],
        cx: k_data[2],
        cy: k_data[5],
        k1: 0.0,
        k2: 0.0,
        p1: 0.0,
        p2: 0.0,
    };

    let robust_kind = match robust.to_lowercase().as_str() {
        "identity" => RobustKernelKind::Identity,
        "huber"    => RobustKernelKind::Huber,
        "cauchy"   => RobustKernelKind::Cauchy,
        "tukey"    => RobustKernelKind::Tukey,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown robust kernel: {other:?}"
            )))
        }
    };

    let params = BaParams {
        max_iterations,
        cost_tolerance: 1e-5,
        gradient_tolerance: 1e-5,
        initial_lambda: 1e-3,
        robust: robust_kind,
        robust_scale_sq: robust_scale * robust_scale,
    };

    let result = rs_bundle_adjust(&poses, &points_vec, &obs_vec, &camera, &params)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;

    // Pack output
    let r_out = unsafe { PyArray::<f64, _>::new(py, [n_poses, 3, 3], false) };
    let t_out = unsafe { PyArray2::<f64>::new(py, [n_poses, 3], false) };
    let r_out_data = unsafe { std::slice::from_raw_parts_mut(r_out.data(), n_poses * 9) };
    let t_out_data = unsafe { std::slice::from_raw_parts_mut(t_out.data(), n_poses * 3) };
    for (i, p) in result.poses.iter().enumerate() {
        let cols = p.rotation.to_cols_array();   // column-major
        // To row-major
        for r in 0..3 {
            for c in 0..3 {
                r_out_data[i * 9 + r * 3 + c] = cols[c * 3 + r];
            }
        }
        t_out_data[i * 3]     = p.translation.x;
        t_out_data[i * 3 + 1] = p.translation.y;
        t_out_data[i * 3 + 2] = p.translation.z;
    }

    let p_out = unsafe { PyArray2::<f64>::new(py, [n_points, 3], false) };
    let p_out_data = unsafe { std::slice::from_raw_parts_mut(p_out.data(), n_points * 3) };
    for (i, pt) in result.points.iter().enumerate() {
        p_out_data[i * 3]     = pt.x;
        p_out_data[i * 3 + 1] = pt.y;
        p_out_data[i * 3 + 2] = pt.z;
    }

    Ok((r_out, t_out, p_out, result.iterations, result.converged))
}
