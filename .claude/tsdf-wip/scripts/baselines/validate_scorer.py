#!/usr/bin/env python
"""Validation of score_mesh.py: (1) camera-convention check, (2) self-tests, (3) cull sanity, (4) degenerate inputs.
Run: /mnt/data/venvs/tsdf-bench/bin/python -I validate_scorer.py > ../scorer/validation.log
Not timed; not a benchmark."""
import importlib.util, json, os, subprocess, sys, tempfile
import numpy as np
import open3d as o3d

HERE = os.path.dirname(os.path.abspath(__file__))
PY = "/mnt/data/venvs/tsdf-bench/bin/python"
R = "/mnt/data/datasets/replica/Replica"
SC = os.path.join(HERE, "score_mesh.py")
OUT = os.path.join(os.path.dirname(HERE), "scorer")
spec = importlib.util.spec_from_file_location("score_mesh", SC)
sm = importlib.util.module_from_spec(spec); spec.loader.exec_module(sm)


def run(mesh, gt, *extra):
    r = subprocess.run([PY, "-I", SC, mesh, "--gt", gt, *extra], capture_output=True, text=True)
    line = [l for l in r.stdout.splitlines() if l.startswith("{")][-1]
    d = json.loads(line); d["_rc"] = r.returncode
    return d


def brief(d):
    ks = ["accuracy_cm", "completeness_cm", "completion_ratio_pct", "precision_pct", "accuracy_surf_cm",
          "completeness_surf_cm", "gt_points_kept", "gt_kept_frac", "pred_triangles", "empty", "error", "_rc"]
    return {k: d[k] for k in ks if k in d}


print("== 1. camera-convention check (GT samples projected with each convention vs sensor depth) ==")
I = sm.DEF_INTR
for scene in ["room0", "office0"]:
    gt = f"{R}/{scene}_mesh.ply"
    _, v, t, _ = sm.load_mesh(gt)
    pts, _ = sm.sample_surface(v, t, 200000, 1)
    traj = sm.load_traj(f"{R}/{scene}/traj.txt")
    for conv in ["opengl", "opencv"]:
        tcv = sm.c2w_opencv(traj, conv)
        res, nproj = [], 0
        for fi in [0, 250, 500, 750, 1000, 1250, 1500, 1999]:
            Rm, tt = tcv[fi][:3, :3], tcv[fi][:3, 3]
            pc = (pts - tt) @ Rm; z = pc[:, 2]
            ok = (z > 0.1) & (z < 7.0)
            u = np.rint(I["fx"] * pc[ok, 0] / z[ok] + I["cx"]).astype(int); vv = np.rint(I["fy"] * pc[ok, 1] / z[ok] + I["cy"]).astype(int)
            inb = (u >= 0) & (u < 1200) & (vv >= 0) & (vv < 680)
            d = sm.read_depth(f"{R}/{scene}/results/depth{fi:06d}.png").astype(float) / I["depth_scale"]
            dp = d[vv[inb], u[inb]]; zz = z[ok][inb]
            # only points that the sensor sees at that pixel (depth valid and point not behind surface by >5 cm)
            m = (dp > 0) & (zz <= dp + 0.05)
            nproj += int(inb.sum()); res.append((zz[m] - dp[m]))
        r = np.concatenate(res) if res else np.zeros(0)
        ar = np.abs(r)
        print(f"{scene} {conv:7s}: in-image pts={nproj} usable={len(r)} median|z-depth|={np.median(ar)*1000 if len(r) else float('nan'):.1f} mm "
              f"frac<2cm={(ar<0.02).mean()*100 if len(r) else 0:.1f}% frac<5cm={(ar<0.05).mean()*100 if len(r) else 0:.1f}%")

print("\n== 2. self-tests (N=200k, scene room0 and office0) ==")
for scene in ["room0", "office0"]:
    gt = f"{R}/{scene}_mesh.ply"
    gm, v, t, _ = sm.load_mesh(gt)
    print(f"-- {scene}")
    print("GT vs GT, SAME seed      :", json.dumps(brief(run(gt, gt, "--seed-pred", "1"))))
    print("GT vs GT, DIFFERENT seed :", json.dumps(brief(run(gt, gt, "--seed-pred", "0"))), "<- sampling floor")
    td = tempfile.mkdtemp(dir=OUT)
    # (a) rigid 1 cm translation along (1,1,1)/sqrt3
    sh = o3d.geometry.TriangleMesh(gm); sh.translate(np.array([1, 1, 1]) / np.sqrt(3) * 0.01)
    p = os.path.join(td, "shift1cm.ply"); o3d.io.write_triangle_mesh(p, sh)
    print("GT rigid-shifted 1 cm     :", json.dumps(brief(run(p, gt))))
    # (b) 1 cm along vertex normals
    nm = o3d.geometry.TriangleMesh(gm); nm.compute_vertex_normals()
    nm.vertices = o3d.utility.Vector3dVector(np.asarray(nm.vertices) + 0.01 * np.asarray(nm.vertex_normals))
    p = os.path.join(td, "normal1cm.ply"); o3d.io.write_triangle_mesh(p, nm)
    print("GT offset 1 cm along normal:", json.dumps(brief(run(p, gt))))
    for f in os.listdir(td): os.remove(os.path.join(td, f))
    os.rmdir(td)

print("\n== 3. frustum-cull sanity ==")
for scene in ["room0", "office0"]:
    gt = f"{R}/{scene}_mesh.ply"; tj = f"{R}/{scene}/traj.txt"
    for fr in ["0:100:1", "0:2000:1"]:
        d = run(gt, gt, "--seed-pred", "0", "--cull-to-frustum", "--traj", tj, "--frames", fr)
        print(f"{scene} frames={fr:9s} gt_kept_frac={d.get('gt_kept_frac')} kept={d.get('gt_points_kept')} cache={d.get('gt_mask_cache')} "
              f"wall_s={d.get('wall_s')} completeness_cm(self,diff-seed)={d.get('completeness_cm')} err={d.get('error')}")

print("\n== 4. degenerate inputs ==")
td = tempfile.mkdtemp(dir=OUT); gt = f"{R}/room0_mesh.ply"
open(os.path.join(td, "empty.ply"), "w").write("ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nproperty float y\nproperty float z\nelement face 0\nproperty list uchar int vertex_indices\nend_header\n")
open(os.path.join(td, "cloud.ply"), "w").write("ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n1 1 1\n")
open(os.path.join(td, "tri_ascii.ply"), "w").write("ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n")
for n in ["empty.ply", "cloud.ply", "tri_ascii.ply"]:
    print(n, json.dumps(brief(run(os.path.join(td, n), gt))))
print("missing file", json.dumps(brief(run("/nonexistent.ply", gt))))
for f in os.listdir(td): os.remove(os.path.join(td, f))
os.rmdir(td)
