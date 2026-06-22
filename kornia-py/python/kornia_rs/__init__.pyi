"""Top-level type stubs for the ``kornia_rs`` package.

``image`` and ``pipeline`` are precisely typed; the remaining submodules are
permissive (``Any``) for now and can be fleshed out incrementally.
"""

from typing import Any

from . import apriltag as apriltag
from . import augmentations as augmentations
from . import cpu as cpu
from . import depth as depth
from . import features as features
from . import image as image
from . import imgproc as imgproc
from . import io as io
from . import k3d as k3d
from . import pipeline as pipeline
from . import segmentation as segmentation

__version__: str

def __getattr__(name: str) -> Any: ...
