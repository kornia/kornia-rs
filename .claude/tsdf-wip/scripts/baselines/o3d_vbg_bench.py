"""Open3D 0.20 tensor VoxelBlockGrid baseline runner (one process = one (scene, voxel, row, device) config).
Extended from the WF-0 rough script o3d_vbg_timing.py. Frames are decoded one at a time OUTSIDE the timer (never preloaded).
Per-frame timer (e2e) = sync -> depth.to(dev) + color.to(dev) [H2D on CUDA; no-op on CPU] ->
compute_unique_block_coordinates -> integrate (incl. hashmap Activate/Find = block allocation) -> sync.
Excludes: PNG/JPG decode (o3d.t.io.read_image, timed separately as decode_ms), pose parse + inv(c2w), python bookkeeping.
Camera: traj.txt c2w used AS-IS (OpenCV axes), extrinsic = inv(c2w)  (binding correction 1).
Modes: --mode size  : block_count small, hashmap grows; records required capacity = max_i(size_before_i + n_frustum_i)
       --mode time  : fixed block_count; mesh extraction (cold + warm reps), ray_cast reps, counts, PLY written.
"""
import argparse, ctypes, json, os, sys, time
import numpy as np
libc = ctypes.CDLL("libc.so.6"); libc.prctl(15, b"o3d_bench", 0, 0, 0)  # PR_SET_NAME so mem_sampler finds us
import open3d as o3d, open3d.core as o3c

ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True); ap.add_argument("--voxel", type=float, required=True)
ap.add_argument("--row", choices=["M", "R"], required=True); ap.add_argument("--device", default="CUDA:0")
ap.add_argument("--block-count", type=int, required=True); ap.add_argument("--nframes", type=int, default=2000)
ap.add_argument("--mode", choices=["size", "time"], default="time"); ap.add_argument("--out", required=True)
ap.add_argument("--nvtx", action="store_true"); ap.add_argument("--warm", type=int, default=5)
ap.add_argument("--mesh-reps", type=int, default=3); ap.add_argument("--rc-warm", type=int, default=5)
ap.add_argument("--rc-reps", type=int, default=50)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
ROW = {"M": dict(trunc_mult=4.0, depth_max=7.0, weight_threshold=1.0),
       "R": dict(trunc_mult=8.0, depth_max=3.0, weight_threshold=3.0)}[a.row]
DEPTH_SCALE = 6553.5
root = f"/mnt/data/datasets/replica/Replica/{a.scene}"
dev = o3c.Device(a.device); is_cuda = a.device.startswith("CUDA")
sync = o3c.cuda.synchronize if is_cuda else (lambda: None)

nv = None
if a.nvtx:
    nv = ctypes.CDLL("/usr/local/cuda-12.6/targets/aarch64-linux/lib/libnvToolsExt.so.1")
def push(s):
    if nv: nv.nvtxRangePushA(s.encode())
def pop():
    if nv: nv.nvtxRangePop()

rt = None
if is_cuda:
    rt = ctypes.CDLL("/usr/local/cuda-12.6/targets/aarch64-linux/lib/libcudart.so.12")
def memgetinfo():
    if rt is None: return None
    f, t = ctypes.c_size_t(), ctypes.c_size_t(); rt.cudaMemGetInfo(ctypes.byref(f), ctypes.byref(t)); return f.value
def rss_kb():
    with open("/proc/self/status") as f:
        for l in f:
            if l.startswith("VmRSS:"): return int(l.split()[1])

K = o3c.Tensor([[600.0, 0, 599.5], [0, 600.0, 339.5], [0, 0, 1]], o3c.float64)  # CPU float64, as VBG expects
traj = np.loadtxt(f"{root}/traj.txt").reshape(-1, 4, 4)
assert traj.shape[0] >= a.nframes

# warm the CUDA context before the memory baseline (context creation is not part of the volume)
if is_cuda:
    o3c.Tensor.zeros((1,), o3c.float32, dev); sync()
free0 = memgetinfo(); rss0 = rss_kb()
push("construct"); t = time.perf_counter()
vbg = o3d.t.geometry.VoxelBlockGrid(("tsdf", "weight", "color"), (o3c.float32, o3c.float32, o3c.float32),
                                    ((1), (1), (3)), a.voxel, 16, a.block_count, dev)
sync(); construct_ms = (time.perf_counter() - t) * 1e3; pop()
free_after_construct = memgetinfo()
cap0 = vbg.hashmap().capacity()

rows = []; cap_events = []; required = 0; free_min = free_after_construct
prev_size = 0; fb = None; ext = None
for i in range(a.nframes):
    t = time.perf_counter()
    d = o3d.t.io.read_image(f"{root}/results/depth{i:06d}.png")
    c = o3d.t.io.read_image(f"{root}/results/frame{i:06d}.jpg")
    decode_ms = (time.perf_counter() - t) * 1e3
    ext = o3c.Tensor(np.linalg.inv(traj[i]), o3c.float64)
    sync(); push("frame"); t0 = time.perf_counter()
    dd, cc = d.to(dev), c.to(dev)
    fb = vbg.compute_unique_block_coordinates(dd, K, ext, DEPTH_SCALE, ROW["depth_max"], ROW["trunc_mult"])
    vbg.integrate(fb, dd, cc, K, K, ext, DEPTH_SCALE, ROW["depth_max"], ROW["trunc_mult"])
    sync(); t1 = time.perf_counter(); pop()
    nfr = int(fb.shape[0]); cap = vbg.hashmap().capacity()
    required = max(required, prev_size + nfr)
    size = vbg.hashmap().size() if (a.mode == "size" or i % 50 == 0 or i == a.nframes - 1) else -1
    if size >= 0: prev_size = size
    elif a.mode == "time": prev_size = prev_size  # 'required' only exact in size mode
    if cap != cap0 and (not cap_events or cap_events[-1][1] != cap): cap_events.append((i, cap))
    fm = memgetinfo() if (i % 50 == 0 or i == a.nframes - 1) else None
    if fm is not None: free_min = min(free_min, fm)
    rows.append((i, (t1 - t0) * 1e3, decode_ms, nfr, size, cap, -1 if fm is None else fm, rss_kb() if i % 50 == 0 else -1))

with open(f"{a.out}/frames.csv", "w") as f:
    f.write("frame,e2e_ms,decode_ms_excluded,n_frustum_blocks,hash_size,hash_capacity,cuda_free_b,vmrss_kb\n")
    for r in rows: f.write(",".join(str(x) for x in r) + "\n")

def stats(x):
    x = np.asarray(x, float)
    return dict(n=int(x.size), mean=float(x.mean()), median=float(np.median(x)), p95=float(np.percentile(x, 95)),
                min=float(x.min()), max=float(x.max()), std=float(x.std()))
e2e = [r[1] for r in rows[a.warm:]]
res = dict(library="open3d", open3d_version=o3d.__version__, open3d_file=o3d.__file__, python=sys.executable,
           scene=a.scene, voxel_m=a.voxel, row=a.row, device=a.device, mode=a.mode, block_resolution=16,
           block_count=a.block_count, hash_capacity_start=int(cap0), hash_capacity_end=int(vbg.hashmap().capacity()),
           capacity_growth_events=cap_events, required_capacity_max_size_plus_frustum=int(required) if a.mode == "size" else None,
           params=dict(trunc_voxel_multiplier=ROW["trunc_mult"], sdf_trunc_m=ROW["trunc_mult"] * a.voxel,
                       depth_max=ROW["depth_max"], depth_scale=DEPTH_SCALE, mesh_weight_threshold=ROW["weight_threshold"],
                       attrs="tsdf/weight/color float32 (1,1,3)"),
           frames_integrated=a.nframes, warmup_frames_discarded=a.warm,
           integrate_e2e_ms=stats(e2e), decode_ms_excluded=stats([r[2] for r in rows[a.warm:]]),
           n_frustum_blocks=stats([r[3] for r in rows]), construct_ms=construct_ms,
           block_count_active=int(vbg.hashmap().size()))
if a.mode == "time":
    sync()
    # full mesh extraction: cold (first call) + warm reps; timer = sync, extract_triangle_mesh, sync (device-resident result; D2H separate)
    mesh_ms = []
    for k in range(1 + a.mesh_reps):
        mesh = None; sync(); push("mesh"); t = time.perf_counter()
        mesh = vbg.extract_triangle_mesh(ROW["weight_threshold"])
        sync(); mesh_ms.append((time.perf_counter() - t) * 1e3); pop()
        fm = memgetinfo(); free_min = min(free_min, fm) if fm is not None else free_min
    t = time.perf_counter(); mcpu = mesh.cpu(); d2h_ms = (time.perf_counter() - t) * 1e3
    nv_ = int(mcpu.vertex.positions.shape[0]) if "positions" in mcpu.vertex else 0
    nt_ = int(mcpu.triangle.indices.shape[0]) if "indices" in mcpu.triangle else 0
    res.update(mesh_full_cold_ms=mesh_ms[0], mesh_full_warm_ms=stats(mesh_ms[1:]) if a.mesh_reps else None,
               mesh_d2h_ms=d2h_ms if is_cuda else None, mesh_vertices=nv_, mesh_triangles=nt_, mesh_incremental_ms="n/a")
    del mesh
    ply = f"{a.out}/mesh.ply"
    if nv_ > 0:
        lm = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mcpu.vertex.positions.numpy().astype(np.float64)),
                                       o3d.utility.Vector3iVector(mcpu.triangle.indices.numpy().astype(np.int32)))
        if "colors" in mcpu.vertex:
            lm.vertex_colors = o3d.utility.Vector3dVector(np.clip(mcpu.vertex.colors.numpy().astype(np.float64), 0, 1))
        o3d.io.write_triangle_mesh(ply, lm, write_ascii=False, compressed=False, write_vertex_normals=False)
        res["mesh_ply"] = ply
    del mcpu
    # ray_cast: one 1200x680 view at the last frame's pose, frustum blocks of the last frame (recon-system call)
    rc_ms = []; rc_attrs = ["depth", "normal", "color"]
    for k in range(a.rc_warm + a.rc_reps):
        sync(); push("ray_cast"); t = time.perf_counter()
        out = vbg.ray_cast(fb, K, ext, 1200, 680, rc_attrs, DEPTH_SCALE, 0.1, ROW["depth_max"], 1.0, ROW["trunc_mult"], 8)
        sync(); rc_ms.append((time.perf_counter() - t) * 1e3); pop()
    dep = out["depth"].cpu().numpy()
    res.update(ray_cast=dict(view=f"frame {a.nframes-1} pose, 1200x680", attrs=rc_attrs, weight_threshold=1.0,
                             depth_min=0.1, depth_max=ROW["depth_max"], range_map_down_factor=8,
                             block_coords="frustum blocks of the last frame (compute_unique_block_coordinates)",
                             warmup_calls_discarded=a.rc_warm, ms=stats(rc_ms[a.rc_warm:]),
                             valid_px_frac=float((dep > 0).mean())))
    del out
    # observed voxel count (weight > 0) over active blocks, chunked; after all timing
    idx = vbg.hashmap().active_buf_indices().to(o3c.int64)
    w = vbg.attribute("weight"); obs = 0
    for s in range(0, idx.shape[0], 1000):
        obs += int((w[idx[s:s + 1000]] > 0).to(o3c.int64).sum().item())
    res.update(voxels_allocated=int(idx.shape[0]) * 4096, voxels_observed_weight_gt0=obs)
res.update(cuda_memgetinfo=None if rt is None else dict(
    free_before_construct_mb=free0 / 2**20, construct_delta_mb=(free0 - free_after_construct) / 2**20,
    peak_delta_mb=(free0 - free_min) / 2**20, note="system-wide on Orin unified memory; sampled every 50 frames + after each mesh call, outside timers"),
    vmrss_kb_before_construct=rss0, vmrss_kb_end=rss_kb())
json.dump(res, open(f"{a.out}/summary.json", "w"), indent=1)
print(json.dumps({k: res[k] for k in ("scene", "voxel_m", "row", "device", "block_count_active", "hash_capacity_end")}),
      "e2e median", round(res["integrate_e2e_ms"]["median"], 2), "mesh tris", res.get("mesh_triangles"))
