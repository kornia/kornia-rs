"""VDBFusion 0.1.6 (CPU, single-threaded std::for_each) Replica bench. Run with /usr/bin/python3.10 -I (sys.path set here).
Poses: traj.txt c2w used AS-IS (OpenCV axes). VDBFusion integrate(points, extrinsic) takes WORLD-frame points and uses only
extrinsic[:3,3] as the ray origin, so camera points are transformed to world here (inside the timer).
One integration (constant weight 1, uncapped) serves Row M (mesh min_weight 1) and Row R (min_weight 5, the RGB-D ICL/TUM
example configs); integration params are identical for both rows."""
import sys, os, json, time, ctypes, argparse, struct
sys.path.insert(0, "/mnt/data/venvs/vdbfusion")
ctypes.CDLL("libc.so.6").prctl(15, b"vdb_bench", 0, 0, 0)  # PR_SET_NAME for mem_sampler
import numpy as np, cv2
import vdbfusion
from vdbfusion import VDBVolume

ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True); ap.add_argument("--voxel", type=float, required=True)
ap.add_argument("--nframes", type=int, default=2000); ap.add_argument("--out", required=True)
ap.add_argument("--min-weights", default="M:1.0,R:5.0"); ap.add_argument("--flip-yz", action="store_true")
ap.add_argument("--mesh-reps", type=int, default=2); ap.add_argument("--warmup", type=int, default=5)
ap.add_argument("--no-vdb", action="store_true")
a = ap.parse_args()
R = f"/mnt/data/datasets/replica/Replica/{a.scene}"
W, H, fx, fy, cx, cy, S = 1200, 680, 600.0, 600.0, 599.5, 339.5, 6553.5
DMIN, DMAX = 0.1, 7.0
TRUNC = 4 * a.voxel
poses = np.loadtxt(f"{R}/traj.txt").reshape(-1, 4, 4)
u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
xr, yr = (u - cx) / fx, (v - cy) / fy  # per-pixel ray factors, precomputed once (outside timer)

def ply_write(path, V, F):
    with open(path, "wb") as f:
        f.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {len(V)}\nproperty float x\nproperty float y\n"
                f"property float z\nelement face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n".encode())
        f.write(np.ascontiguousarray(V, dtype="<f4").tobytes())
        fa = np.empty(len(F), dtype=[("n", "u1"), ("i", "<i4", 3)]); fa["n"] = 3; fa["i"] = F
        f.write(fa.tobytes())

def stats(x):
    x = np.asarray(x, dtype=np.float64)
    return dict(n=int(x.size), mean=float(x.mean()), median=float(np.median(x)), p95=float(np.percentile(x, 95)),
                min=float(x.min()), max=float(x.max()), std=float(x.std()))

t_c = time.perf_counter(); vol = VDBVolume(voxel_size=a.voxel, sdf_trunc=TRUNC, space_carving=False)
construct_ms = (time.perf_counter() - t_c) * 1e3
dec, unp, integ, e2e, npts = [], [], [], [], []
t_loop = time.perf_counter()
for i in range(a.nframes):
    td = time.perf_counter()
    d16 = cv2.imread(f"{R}/results/depth{i:06d}.png", cv2.IMREAD_UNCHANGED)
    T = poses[i].copy()
    if a.flip_yz: T[:3, 1] *= -1; T[:3, 2] *= -1
    T = np.ascontiguousarray(T)
    t0 = time.perf_counter()
    d = d16 * (1.0 / S)
    m = (d >= DMIN) & (d < DMAX)
    z = d[m]; X = xr[m] * z; Y = yr[m] * z
    Rm, t = T[:3, :3], T[:3, 3]
    pts = np.empty((z.size, 3), dtype=np.float64)
    pts[:, 0] = Rm[0, 0] * X + Rm[0, 1] * Y + Rm[0, 2] * z + t[0]
    pts[:, 1] = Rm[1, 0] * X + Rm[1, 1] * Y + Rm[1, 2] * z + t[1]
    pts[:, 2] = Rm[2, 0] * X + Rm[2, 1] * Y + Rm[2, 2] * z + t[2]
    t1 = time.perf_counter()
    vol.integrate(pts, T)
    t2 = time.perf_counter()
    dec.append((t0 - td) * 1e3); unp.append((t1 - t0) * 1e3); integ.append((t2 - t1) * 1e3); e2e.append((t2 - t0) * 1e3)
    npts.append(int(z.size))
    if i % 50 == 0:
        print(f"frame {i} e2e {e2e[-1]:.1f} ms integ {integ[-1]:.1f} pts {z.size} elapsed {time.perf_counter()-t_loop:.0f}s", file=sys.stderr, flush=True)
loop_s = time.perf_counter() - t_loop
w = a.warmup
res = dict(library="vdbfusion", version=getattr(vdbfusion, "__version__", "0.1.6"), scene=a.scene, voxel_m=a.voxel,
           sdf_trunc_m=TRUNC, space_carving=False, depth_range_m=[DMIN, DMAX], depth_range_kind="z-depth (camera z), [min, max)",
           flip_yz=a.flip_yz, frames_integrated=a.nframes, warmup_frames_discarded=w, construct_ms=construct_ms,
           integrate_loop_wall_s=loop_s,
           integrate_e2e_ms=stats(e2e[w:]), integrate_only_ms=stats(integ[w:]), unproject_ms=stats(unp[w:]),
           decode_ms_excluded=stats(dec[w:]), points_per_frame=stats(npts[w:]), omp_threads="n/a (std::for_each sequential)")
np.savez_compressed(f"{a.out}/per_frame.npz", e2e=e2e, integ=integ, unp=unp, dec=dec, npts=npts)
if not a.no_vdb:
    vdbf = f"{a.out}/grids.vdb"; vol.extract_vdb_grids(vdbf); res["vdb_file_bytes"] = os.path.getsize(vdbf)
res["meshes"] = {}
for spec in a.min_weights.split(","):
    row, mw = spec.split(":"); mw = float(mw)
    ts = []
    for r in range(1 + a.mesh_reps):
        tm = time.perf_counter(); V, F = vol.extract_triangle_mesh(fill_holes=True, min_weight=mw); ts.append((time.perf_counter() - tm) * 1e3)
    V = np.asarray(V); F = np.asarray(F)
    ply = f"{a.out}/mesh_{row}.ply"
    if len(F): ply_write(ply, V, F)
    res["meshes"][row] = dict(min_weight=mw, fill_holes=True, mesh_full_cold_ms=ts[0], mesh_full_warm_ms=ts[1:],
                              mesh_full_warm_median_ms=float(np.median(ts[1:])) if len(ts) > 1 else None,
                              vertices=int(len(V)), triangles=int(len(F)), ply=ply if len(F) else None)
    print(f"mesh {row} min_weight {mw}: {len(V)} V {len(F)} F cold {ts[0]:.0f} ms", file=sys.stderr, flush=True)
json.dump(res, open(f"{a.out}/bench.json", "w"), indent=1)
print(json.dumps(dict(e2e_median=res["integrate_e2e_ms"]["median"], loop_s=loop_s, meshes={k: v["triangles"] for k, v in res["meshes"].items()})))
