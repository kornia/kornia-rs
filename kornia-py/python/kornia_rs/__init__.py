# Best-effort NVRTC preload for kornia_rs.cuda on systems without the CUDA
# toolkit: the nvidia-cuda-nvrtc-cu12 pip wheel ships libnvrtc, but it is not
# on the default loader path — dlopen it before the native module needs it.
def _preload_nvrtc() -> None:
    try:
        import ctypes
        import glob
        import os

        import nvidia.cuda_nvrtc as _nvrtc  # type: ignore[import-not-found]

        libdir = os.path.join(os.path.dirname(_nvrtc.__file__), "lib")
        for lib in sorted(glob.glob(os.path.join(libdir, "libnvrtc.so*"))):
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            break
    except Exception:
        pass  # toolkit installs (e.g. Jetson) already resolve libnvrtc


_preload_nvrtc()

from .kornia_rs import *

__doc__ = kornia_rs.__doc__
if hasattr(kornia_rs, "__all__"):
    __all__ = kornia_rs.__all__
