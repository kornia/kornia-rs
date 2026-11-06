//! NumPy interface to the float64 affine Schur solver.
use crate::pyutils::{c_slice, value_err};
use kornia_3d::{
    ba_affine::{bundle_adjust_affine, AffineBaParams, AffineObservation},
    camera::AffineCamera,
};
use kornia_algebra::Vec3F64;
use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::{prelude::*, types::PyDict};

/// Simultaneous float64 affine BA with Cauchy loss and fixed-point gauge.
///
/// Arrays must be aligned and C-contiguous. Cameras are (C,2,4), points (N,3),
/// pixels (O,2), camera_indices and point_indices (O,) int64. Fixed point indices
/// must include four noncoplanar anchors. No inputs are modified. The returned
/// dictionary contains cameras, points, initial_cost, cost, iterations,
/// accepted_steps and converged. Costs use 0.5*robust_scale**2*sum(log1p(r²/scale²)).
#[pyfunction(name = "bundle_adjust_affine")]
#[pyo3(signature=(cameras,points,camera_indices,point_indices,pixels,fixed_point_indices,*,max_iterations=100,robust_scale=2.0,cost_tolerance=1e-7,initial_damping=1e-3))]
#[allow(clippy::too_many_arguments)]
pub fn bundle_adjust_affine_py<'py>(
    py: Python<'py>,
    cameras: Bound<'py, PyArray3<f64>>,
    points: Bound<'py, PyArray2<f64>>,
    camera_indices: Bound<'py, PyArray1<i64>>,
    point_indices: Bound<'py, PyArray1<i64>>,
    pixels: Bound<'py, PyArray2<f64>>,
    fixed_point_indices: Vec<usize>,
    max_iterations: usize,
    robust_scale: f64,
    cost_tolerance: f64,
    initial_damping: f64,
) -> PyResult<Bound<'py, PyDict>> {
    if cameras.shape()[1..] != [2, 4] || points.shape()[1] != 3 || pixels.shape()[1] != 2 {
        return Err(value_err(
            "expected cameras (C,2,4), points (N,3), pixels (O,2)",
        ));
    }
    let no = pixels.shape()[0];
    if camera_indices.len() != no || point_indices.len() != no {
        return Err(value_err("observation array lengths must match"));
    }
    let cameras_vec: Vec<_> = c_slice(&cameras, "cameras")?
        .chunks_exact(8)
        .map(|c| AffineCamera {
            matrix: [[c[0], c[1], c[2], c[3]], [c[4], c[5], c[6], c[7]]],
        })
        .collect();
    let points_vec: Vec<_> = c_slice(&points, "points")?
        .chunks_exact(3)
        .map(|x| Vec3F64::new(x[0], x[1], x[2]))
        .collect();
    let mut fixed = vec![false; points_vec.len()];
    for p in fixed_point_indices {
        let Some(flag) = fixed.get_mut(p) else {
            return Err(value_err("fixed point index out of bounds"));
        };
        *flag = true;
    }
    let ci = c_slice(&camera_indices, "camera_indices")?;
    let pi = c_slice(&point_indices, "point_indices")?;
    let xy = c_slice(&pixels, "pixels")?;
    let observations: Vec<_> = (0..no)
        .map(|i| {
            Ok(AffineObservation {
                camera_index: usize::try_from(ci[i]).map_err(value_err)?,
                point_index: usize::try_from(pi[i]).map_err(value_err)?,
                pixel: [xy[2 * i], xy[2 * i + 1]],
            })
        })
        .collect::<PyResult<_>>()?;
    let params = AffineBaParams {
        max_iterations,
        robust_scale,
        cost_tolerance,
        initial_damping,
        ..Default::default()
    };
    // Only owned Rust buffers cross the detached section, including on free-threaded Python.
    let result = py
        .detach(move || {
            bundle_adjust_affine(&cameras_vec, &points_vec, &observations, &fixed, &params)
        })
        .map_err(value_err)?;
    let nc = result.cameras.len();
    let np = result.points.len();
    let camera_output: Vec<_> = result
        .cameras
        .iter()
        .flat_map(|c| c.matrix.into_iter().flatten())
        .collect();
    let point_output: Vec<_> = result.points.iter().flat_map(|p| p.to_array()).collect();
    let out = PyDict::new(py);
    out.set_item(
        "cameras",
        PyArray1::from_vec(py, camera_output).reshape([nc, 2, 4])?,
    )?;
    out.set_item(
        "points",
        PyArray1::from_vec(py, point_output).reshape([np, 3])?,
    )?;
    out.set_item("initial_cost", result.initial_cost)?;
    out.set_item("cost", result.cost)?;
    out.set_item("iterations", result.iterations)?;
    out.set_item("accepted_steps", result.accepted_steps)?;
    out.set_item("converged", result.converged)?;
    Ok(out)
}
