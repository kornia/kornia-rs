# Best-effort NVRTC preload for kornia_rs.cuda on systems without the CUDA
# toolkit: the nvidia-cuda-nvrtc-cu12 pip wheel ships libnvrtc, but it is not
# on the default loader path — dlopen it before the native module needs it.
def _preload_nvrtc() -> None:
    try:
        import ctypes
        import glob
        import os

        import nvidia.cuda_nvrtc as _nvrtc  # type: ignore[import-not-found]

        # ``nvidia.cuda_nvrtc`` is a PEP 420 namespace package, so ``__file__``
        # is ``None`` and ``os.path.dirname(None)`` would raise — locate the
        # bundled ``lib/`` via ``__path__`` instead (always present on a
        # package, namespace or regular).
        for root in getattr(_nvrtc, "__path__", []):
            for lib in sorted(glob.glob(os.path.join(root, "lib", "libnvrtc.so*"))):
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                return
    except Exception:
        pass  # toolkit installs (e.g. Jetson) already resolve libnvrtc


_preload_nvrtc()

from .kornia_rs import *

__doc__ = kornia_rs.__doc__
if hasattr(kornia_rs, "__all__"):
    __all__ = kornia_rs.__all__
