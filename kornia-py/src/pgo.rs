//! `kornia_rs.k3d.pose_graph_optimize` — SE(3) pose-graph LM solver.
//!
//! Each edge contributes a 6-d residual `weight · log(T_ab_meas⁻¹ · T_b · T_a⁻¹)`
//! in `se(3)` tangent. Anchor poses are held fixed via `fixed_pose_indices`.

use kornia_3d::pgo::{pose_graph_optimize as rs_pgo, PgoEdge, PgoParams};
use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, SE3F32, SO3F32, Vec3AF32, Vec3F64};
use numpy::{PyArray, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;

/// Build a [`Pose3d`] from row-major (3, 3) rotation + (3,) translation slices.
fn pose_from_slice(r: &[f64], t: &[f64]) -> Pose3d {
    let rotation = Mat3F64::from_cols(
        Vec3F64::new(r[0], r[3], r[6]),
        Vec3F64::new(r[1], r[4], r[7]),
        Vec3F64::new(r[2], r[5], r[8]),
    );
    Pose3d::new(rotation, Vec3F64::new(t[0], t[1], t[2]))
}

/// Build an [`SE3F32`] (camera-to-camera relative transform) from (3, 3) row-major
/// rotation + (3,) translation slices.
fn se3_from_slice(r: &[f64], t: &[f64]) -> SE3F32 {
    use kornia_algebra::Mat3AF32;
    let r_mat = Mat3AF32::from_cols(
        Vec3AF32::new(r[0] as f32, r[3] as f32, r[6] as f32),
        Vec3AF32::new(r[1] as f32, r[4] as f32, r[7] as f32),
        Vec3AF32::new(r[2] as f32, r[5] as f32, r[8] as f32),
    );
    let so3 = SO3F32::from_matrix(&r_mat);
    SE3F32::new(
        so3,
        Vec3AF32::new(t[0] as f32, t[1] as f32, t[2] as f32),
    )
}

/// Solve a pose graph.
///
/// Args:
///     rotations: `(P, 3, 3)` float64 — world→camera rotation per pose.
///     translations: `(P, 3)` float64 — world→camera translation per pose.
///     edges: `(E, 15)` float64 — per row
///         `[pose_a, pose_b, R(9 row-major), t(3), weight]`
///         where `R` and `t` form `T_ab_meas` = transform from cam_a to cam_b.
///     fixed_pose_indices: list[int] — poses to hold constant (e.g. `[0]`
///         to anchor the gauge).
///     max_iterations: LM iteration cap (default 30).
///
/// Returns:
///     `(rotations_opt, translations_opt, iterations, converged)`.
#[pyfunction(name = "pose_graph_optimize")]
#[pyo3(signature = (rotations, translations, edges, fixed_pose_indices, max_iterations=30))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn pose_graph_optimize_py<'py>(
    py: Python<'py>,
    rotations: Bound<'py, PyArray<f64, numpy::Ix3>>,
    translations: Bound<'py, PyArray2<f64>>,
    edges: Bound<'py, PyArray2<f64>>,
    fixed_pose_indices: Vec<usize>,
    max_iterations: usize,
) -> PyResult<(
    Bound<'py, PyArray<f64, numpy::Ix3>>,
    Bound<'py, PyArray2<f64>>,
    usize,
    bool,
)> {
    let r_shape = rotations.shape();
    let t_shape = translations.shape();
    let e_shape = edges.shape();

    if r_shape.len() != 3 || r_shape[1] != 3 || r_shape[2] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "rotations must be (P, 3, 3) float64",
        ));
    }
    if t_shape.len() != 2 || t_shape[1] != 3 || t_shape[0] != r_shape[0] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "translations must be (P, 3) matching rotations[0]",
        ));
    }
    if e_shape.len() != 2 || e_shape[1] != 15 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "edges must be (E, 15) float64: [pose_a, pose_b, R(9), t(3), weight]",
        ));
    }

    let n_poses = r_shape[0];
    let n_edges = e_shape[0];

    // Build poses.
    let r_data = unsafe { std::slice::from_raw_parts(rotations.data(), n_poses * 9) };
    let t_data = unsafe { std::slice::from_raw_parts(translations.data(), n_poses * 3) };
    let mut poses: Vec<Pose3d> = Vec::with_capacity(n_poses);
    for i in 0..n_poses {
        poses.push(pose_from_slice(
            &r_data[i * 9..i * 9 + 9],
            &t_data[i * 3..i * 3 + 3],
        ));
    }

    // Edge row layout (15 cols): [pose_a, pose_b, R[0..9] row-major, t[0..3], weight]
    let e_data = unsafe { std::slice::from_raw_parts(edges.data(), n_edges * 15) };
    let mut edge_vec: Vec<PgoEdge> = Vec::with_capacity(n_edges);
    for i in 0..n_edges {
        let off = i * 15;
        let pose_a = e_data[off] as usize;
        let pose_b = e_data[off + 1] as usize;
        if pose_a >= n_poses || pose_b >= n_poses {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "edge {i} has out-of-range pose index (a={pose_a}, b={pose_b}, n={n_poses})"
            )));
        }
        let r = &e_data[off + 2..off + 11];
        let t = &e_data[off + 11..off + 14];
        let weight = e_data[off + 14] as f32;
        edge_vec.push(PgoEdge {
            pose_a,
            pose_b,
            t_ab_meas: se3_from_slice(r, t),
            weight,
        });
    }

    let params = PgoParams {
        max_iterations,
        ..Default::default()
    };
    let result = rs_pgo(&poses, &edge_vec, &fixed_pose_indices, &params).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("{e}"))
    })?;

    // Pack output (R, t).
    let r_out = unsafe { PyArray::<f64, _>::new(py, [n_poses, 3, 3], false) };
    let t_out = unsafe { PyArray2::<f64>::new(py, [n_poses, 3], false) };
    let r_out_data = unsafe { std::slice::from_raw_parts_mut(r_out.data(), n_poses * 9) };
    let t_out_data = unsafe { std::slice::from_raw_parts_mut(t_out.data(), n_poses * 3) };
    for (i, p) in result.poses.iter().enumerate() {
        let cols = p.rotation.to_cols_array();
        for r in 0..3 {
            for c in 0..3 {
                r_out_data[i * 9 + r * 3 + c] = cols[c * 3 + r];
            }
        }
        t_out_data[i * 3] = p.translation.x;
        t_out_data[i * 3 + 1] = p.translation.y;
        t_out_data[i * 3 + 2] = p.translation.z;
    }

    Ok((r_out, t_out, result.iterations, result.converged))
}
