# Integrates N Replica room0 frames into VDBFusion and extracts a mesh.
# Poses: traj.txt is c2w with OpenGL axes (NICE-SLAM convention) -> flip y,z columns to get OpenCV camera axes.
import sys, numpy as np, cv2, time
from vdbfusion import VDBVolume
R = "/mnt/data/datasets/replica/Replica/room0"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
W, H, fx, fy, cx, cy, S = 1200, 680, 600.0, 600.0, 599.5, 339.5, 6553.5
poses = np.loadtxt(f"{R}/traj.txt").reshape(-1, 4, 4)
u, v = np.meshgrid(np.arange(W), np.arange(H))
vol = VDBVolume(voxel_size=0.05, sdf_trunc=0.2, space_carving=False)
for i in range(N):
    d = cv2.imread(f"{R}/results/depth{i:06d}.png", cv2.IMREAD_UNCHANGED).astype(np.float64) / S
    m = (d > 0) & (d < 10)
    z = d[m]
    pts = np.stack([(u[m] - cx) * z / fx, (v[m] - cy) * z / fy, z], 1)
    T = poses[i].copy(); T[:3, 1] *= -1; T[:3, 2] *= -1
    vol.integrate(np.ascontiguousarray(pts, dtype=np.float64), np.ascontiguousarray(T, dtype=np.float64))
V, F = vol.extract_triangle_mesh(fill_holes=True, min_weight=0.0)
print("vertices", V.shape, "triangles", F.shape)
print("bbox min", V.min(0), "max", V.max(0))
