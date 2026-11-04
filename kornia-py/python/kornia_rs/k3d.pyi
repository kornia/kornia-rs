"""Type stubs for the ``kornia_rs.k3d`` submodule.

Only selected symbols are currently typed; unknown symbols resolve to ``Any``.
"""

from typing import Any

import numpy as np


class PinholeCamera:
    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        k1: float = ...,
        k2: float = ...,
        p1: float = ...,
        p2: float = ...,
    ) -> None: ...

    def project(self, points_cam: np.ndarray) -> np.ndarray: ...


def __getattr__(name: str) -> Any: ...
