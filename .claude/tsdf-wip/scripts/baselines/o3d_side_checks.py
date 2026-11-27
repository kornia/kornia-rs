"""Untimed side checks: (1) VBG Row R weight_threshold=3 gives 0 vertices for the first frames;
(2) legacy ScalableTSDFVolume on the first 5 frames (expected broken = 0 vertices on aarch64 wheel); (3) mesh colour range."""
import json, sys, numpy as np, open3d as o3d, open3d.core as o3c
root = "/mnt/data/datasets/replica/Replica/room0"; dev = o3c.Device(sys.argv[1] if len(sys.argv) > 1 else "CPU:0")
K = o3c.Tensor([[600.0, 0, 599.5], [0, 600.0, 339.5], [0, 0, 1]], o3c.float64)
traj = np.loadtxt(f"{root}/traj.txt").reshape(-1, 4, 4)
out = dict(open3d=o3d.__version__, file=o3d.__file__, device=str(dev), vbg_rowR_thr3=[], vbg_rowM_thr1=[])
for key, tm, dmax, thr in (("vbg_rowR_thr3", 8.0, 3.0, 3.0), ("vbg_rowM_thr1", 4.0, 7.0, 1.0)):
    vbg = o3d.t.geometry.VoxelBlockGrid(("tsdf", "weight", "color"), (o3c.float32,) * 3, ((1), (1), (3)), 0.05, 16, 3000, dev)
    for i in range(6):
        d = o3d.t.io.read_image(f"{root}/results/depth{i:06d}.png").to(dev); c = o3d.t.io.read_image(f"{root}/results/frame{i:06d}.jpg").to(dev)
        ext = o3c.Tensor(np.linalg.inv(traj[i]), o3c.float64)
        fb = vbg.compute_unique_block_coordinates(d, K, ext, 6553.5, dmax, tm); vbg.integrate(fb, d, c, K, K, ext, 6553.5, dmax, tm)
        m = vbg.extract_triangle_mesh(thr)
        nv = int(m.vertex.positions.shape[0]) if "positions" in m.vertex else 0
        col = m.vertex.colors.cpu().numpy() if "colors" in m.vertex else None
        out[key].append(dict(frames=i + 1, vertices=nv, color_min=None if col is None or nv == 0 else float(col.min()),
                             color_max=None if col is None or nv == 0 else float(col.max())))
vol = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=0.02, sdf_trunc=0.08,
                                                   color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
intr = o3d.camera.PinholeCameraIntrinsic(1200, 680, 600.0, 600.0, 599.5, 339.5)
for i in range(5):
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.io.read_image(f"{root}/results/frame{i:06d}.jpg"),
        o3d.io.read_image(f"{root}/results/depth{i:06d}.png"), depth_scale=6553.5, depth_trunc=7.0, convert_rgb_to_intensity=False)
    vol.integrate(rgbd, intr, np.linalg.inv(traj[i]))
lm = vol.extract_triangle_mesh(); pc = vol.extract_point_cloud()
out["legacy_scalable_5frames_v0.02"] = dict(mesh_vertices=len(lm.vertices), mesh_triangles=len(lm.triangles), point_cloud_points=len(pc.points))
print(json.dumps(out))
