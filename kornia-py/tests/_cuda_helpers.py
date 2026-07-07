"""Shared CUDA test helpers — the two-step device-Image constructions that
mirror the removed ``Image.cuda.*`` namespace, in one place so the three device
test modules can't drift."""

from kornia_rs.cuda import Stream
from kornia_rs.image import Image


def dev(a, stream=None):
    """Host numpy/array -> device Image (was ``Image.cuda.from_numpy``)."""
    return Image.from_numpy(a).to_cuda(stream)


def dzeros(*args, stream=None, **kw):
    """Zero-init device Image (was ``Image.cuda.zeros``)."""
    return Image.zeros(*args, **kw, stream=stream if stream is not None else Stream.default())
