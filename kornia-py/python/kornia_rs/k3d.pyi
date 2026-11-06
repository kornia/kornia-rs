"""Permissive type stubs for the ``kornia_rs.k3d`` submodule (``Any`` for now)."""

from typing import Any

def __getattr__(name: str) -> Any: ...

from typing import TypedDict, Sequence
import numpy as np
import numpy.typing as npt

class AffineBaResult(TypedDict):
    cameras: npt.NDArray[np.float64]
    points: npt.NDArray[np.float64]
    initial_cost: float
    cost: float
    iterations: int
    accepted_steps: int
    converged: bool

def bundle_adjust_affine(
    cameras: npt.NDArray[np.float64],
    points: npt.NDArray[np.float64],
    camera_indices: npt.NDArray[np.int64],
    point_indices: npt.NDArray[np.int64],
    pixels: npt.NDArray[np.float64],
    fixed_point_indices: Sequence[int],
    *,
    max_iterations: int = 100,
    robust_scale: float = 2.0,
    cost_tolerance: float = 1e-7,
    initial_damping: float = 1e-3,
) -> AffineBaResult: ...
