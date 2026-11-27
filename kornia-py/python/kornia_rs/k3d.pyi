"""Affine BA types and a permissive fallback for other ``kornia_rs.k3d`` APIs."""

from typing import Any, Sequence

import numpy as np
import numpy.typing as npt

def __getattr__(name: str) -> Any: ...

class AffineBaResult:
    @property
    def cameras(self) -> npt.NDArray[np.float64]: ...
    @property
    def points(self) -> npt.NDArray[np.float64]: ...
    @property
    def initial_cost(self) -> float: ...
    @property
    def cost(self) -> float: ...
    @property
    def iterations(self) -> int: ...
    @property
    def accepted_steps(self) -> int: ...
    @property
    def converged(self) -> bool: ...

def bundle_adjust_affine(
    cameras: npt.NDArray[np.float64],
    points: npt.NDArray[np.float64],
    camera_indices: npt.NDArray[np.int64],
    point_indices: npt.NDArray[np.int64],
    pixels: npt.NDArray[np.float64],
    fixed_point_indices: Sequence[int] | npt.NDArray[np.integer[Any]],
    *,
    max_iterations: int = 100,
    robust_scale: float = 2.0,
    cost_tolerance: float = 1e-7,
    initial_damping: float = 1e-3,
    max_trials: int = 30,
) -> AffineBaResult: ...
