//! NumPy interface to the float64 affine Schur solver.
use crate::pyutils::{c_slice, rows_to_numpy, value_err};
use kornia_3d::{
    ba_affine::{bundle_adjust_affine, AffineBaParams, AffineObservation},
    camera::AffineCamera,
};
use kornia_algebra::{Vec2F64, Vec3F64};
use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;

/// Result of an affine bundle adjustment. Field access mirrors the Rust `AffineBaResult`.
#[pyclass(name = "AffineBaResult", module = "kornia_rs.k3d", frozen)]
pub struct PyAffineBaResult {
    /// `(C, 2, 4)` float64 optimized world-to-pixel matrices.
    #[pyo3(get)]
    pub cameras: Py<PyArray3<f64>>,
    /// `(N, 3)` float64 optimized world points; fixed points keep their input values.
    #[pyo3(get)]
    pub points: Py<PyArray2<f64>>,
    /// Cost at the input state.
    #[pyo3(get)]
    pub initial_cost: f64,
    /// Cost at the returned state.
    #[pyo3(get)]
    pub cost: f64,
    /// Linearizations performed.
    #[pyo3(get)]
    pub iterations: usize,
    /// Steps that lowered the cost.
    #[pyo3(get)]
    pub accepted_steps: usize,
    /// False when the iteration or damping-trial budget ran out before a stopping test passed.
    #[pyo3(get)]
    pub converged: bool,
}

#[pymethods]
impl PyAffineBaResult {
    fn __repr__(&self) -> String {
        format!(
            "AffineBaResult(cost={}, initial_cost={}, iterations={}, accepted_steps={}, converged={})",
            self.cost,
            self.initial_cost,
            self.iterations,
            self.accepted_steps,
            if self.converged { "True" } else { "False" }
        )
    }
}

/// Simultaneous float64 affine BA with Cauchy loss and fixed-point gauge.
///
/// Arrays must be aligned and C-contiguous. No inputs are modified. Costs use
/// 0.5*robust_scale**2*sum(log1p(r²/robust_scale²)), summed per 2D observation.
///
/// Args:
///     cameras: `(C, 2, 4)` float64 world-to-pixel matrices.
///     points: `(N, 3)` float64 initial world points.
///     camera_indices: `(O,)` int64 camera index per observation.
///     point_indices: `(O,)` int64 point index per observation.
///     pixels: `(O, 2)` float64 measured pixels.
///     fixed_point_indices: Sequence of indices of points to hold exactly. Must
///         include four noncoplanar anchors, each seen in at least two views
///         when any point is free.
///     max_iterations: Maximum linearizations (default 100).
///     robust_scale: Positive Cauchy scale in pixels (default 2.0).
///     cost_tolerance: Positive relative cost decrease at which to stop, once
///         damping is at most 1 (default 1e-7).
///     initial_damping: Positive initial LM diagonal multiplier (default 1e-3).
///     max_trials: Maximum damping trials per linearization (default 30).
///
/// Returns:
///     An `AffineBaResult` with new `cameras` and `points` arrays, plus `initial_cost`,
///     `cost`, `iterations`, `accepted_steps` and `converged`. Exhausting
///     `max_iterations`, or the damping trials after some progress, returns the last
///     accepted state with `converged=False`.
///
/// Raises:
///     ValueError: Invalid shapes, memory layout, indices, nonfinite values or
///         solver parameters, a degenerate gauge, or damping trials exhausted
///         before any accepted step.
///     TypeError: An array has the wrong dtype or dimensions.
///     OverflowError: A negative `fixed_point_indices` entry or integer option.
#[pyfunction(name = "bundle_adjust_affine")]
#[pyo3(signature=(cameras,points,camera_indices,point_indices,pixels,fixed_point_indices,*,max_iterations=100,robust_scale=2.0,cost_tolerance=1e-7,initial_damping=1e-3,max_trials=30))]
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
    max_trials: usize,
) -> PyResult<PyAffineBaResult> {
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
                pixel: Vec2F64::new(xy[2 * i], xy[2 * i + 1]),
            })
        })
        .collect::<PyResult<_>>()?;
    let params = AffineBaParams {
        max_iterations,
        robust_scale,
        cost_tolerance,
        initial_damping,
        max_trials,
    };
    // Only owned Rust buffers cross the detached section, including on free-threaded Python.
    let result = py
        .detach(move || {
            bundle_adjust_affine(&cameras_vec, &points_vec, &observations, &fixed, &params)
        })
        .map_err(value_err)?;
    let nc = result.cameras.len();
    let camera_output: Vec<_> = result
        .cameras
        .iter()
        .flat_map(|c| c.matrix.into_iter().flatten())
        .collect();
    let point_output: Vec<_> = result.points.iter().flat_map(|p| p.to_array()).collect();
    Ok(PyAffineBaResult {
        cameras: PyArray1::from_vec(py, camera_output)
            .reshape([nc, 2, 4])?
            .unbind(),
        points: rows_to_numpy(py, point_output, 3)?.unbind(),
        initial_cost: result.initial_cost,
        cost: result.cost,
        iterations: result.iterations,
        accepted_steps: result.accepted_steps,
        converged: result.converged,
    })
}
