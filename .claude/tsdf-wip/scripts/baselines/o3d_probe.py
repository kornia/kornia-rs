"""Probe an Open3D install: version, CUDA availability, numpy, legacy ScalableTSDFVolume sanity (1 line)."""
import sys, json, numpy as np, open3d as o3d, open3d.core as o3c
info = dict(python=sys.version.split()[0], executable=sys.executable, open3d=o3d.__version__, open3d_file=o3d.__file__,
            numpy=np.__version__, cuda_available=o3c.cuda.is_available(), cuda_device_count=o3c.cuda.device_count(),
            build_cuda=getattr(o3d, "_build_config", {}).get("BUILD_CUDA_MODULE"),
            build_config={k: v for k, v in getattr(o3d, "_build_config", {}).items() if k.startswith(("BUILD", "GIT", "CUDA", "WITH"))})
print(json.dumps(info))
