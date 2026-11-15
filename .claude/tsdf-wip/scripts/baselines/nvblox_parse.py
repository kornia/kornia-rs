"""Parse one nvblox run dir (from nvblox_run.sh) into summary.json. CPU only, run between GPU runs.
Per-frame stats come from NVTX ranges (nsys sqlite) when present; built-in table always parsed."""
import json, os, re, sqlite3, statistics, sys
from datetime import datetime
import numpy as np
R = sys.argv[1]; WARM = int(sys.argv[2]) if len(sys.argv) > 2 else 5
S = dict(run_dir=R, warm_frames_discarded=WARM)

def table(path):
    out = {}
    for line in open(path):
        m = re.match(r"^(\S+)\s+(\d+)\s+(\S+)\s+\((\S+) \+- (\S+)\)\s+\[(\S+),(\S+)\]", line)
        if m:
            t = m.group(1); n = int(m.group(2)); v = [float(m.group(i)) for i in range(3, 8)]
            out[t] = dict(n=n, total_s=v[0], mean_ms=v[1]*1e3, std_ms=v[2]*1e3, min_ms=v[3]*1e3, max_ms=v[4]*1e3,
                          mean_excl_max_ms=((v[0]-v[4])/(n-1)*1e3 if n > 1 else None))
    return out
S["builtin"] = table(os.path.join(R, "timing.txt")) if os.path.exists(os.path.join(R, "timing.txt")) else {}

def st(a):
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0: return None
    return dict(n=int(a.size), mean=float(a.mean()), median=float(np.median(a)), p95=float(np.percentile(a, 95)),
                min=float(a.min()), max=float(a.max()), std=float(a.std(ddof=1)) if a.size > 1 else 0.0)

prof = os.path.join(R, "prof.sqlite")
if os.path.exists(prof):
    c = sqlite3.connect(prof)
    rows = c.execute("select e.start, e.end, coalesce(e.text, s.value) from NVTX_EVENTS e left join StringIds s on e.textId = s.id "
                     "where e.eventType = 60 and e.end is not null order by e.start").fetchall()
    per = {}
    for a, b, t in rows: per.setdefault(t, []).append((b - a) / 1e6)
    tags = {"depth": "fuser/integrate_depth", "color": "fuser/integrate_color", "dl": "file_loading/depth_image",
            "dstbi": "file_loading/depth_image/stbi", "cl": "file_loading/color_image", "cstbi": "file_loading/color_image/stbi",
            "mesh": "fuser/mesh", "tpf": "fuser/time_per_frame", "flat": "mapper/update_flat_mesh", "load": "fuser/file_loading",
            "tsdf": "tsdf/integrate", "colint": "color/integrate"}
    d = {k: np.array(per.get(v, [])) for k, v in tags.items()}
    nfr = len(d["tpf"])
    S["nvtx_counts"] = {v: len(per.get(v, [])) for v in tags.values()}
    def w(x): return x[WARM:nfr] if len(x) >= nfr else x[WARM:]
    integ = d["depth"][:nfr] + d["color"][:nfr]
    conv_h2d = (d["dl"][:nfr] - d["dstbi"][:nfr]) + (d["cl"][:nfr] - d["cstbi"][:nfr])
    pf = dict(integrate_depth_ms=st(w(d["depth"])), integrate_color_ms=st(w(d["color"])),
              integrate_ms=st(w(integ)), convert_h2d_ms=st(w(conv_h2d)), integrate_e2e_ms=st(w(integ + conv_h2d)),
              tsdf_integrate_ms=st(w(d["tsdf"])), color_integrate_ms=st(w(d["colint"])),
              time_per_frame_ms=st(w(d["tpf"])), file_loading_ms=st(w(d["load"])),
              mesh_incremental_ms=st(w(d["mesh"])) if len(d["mesh"]) else None,
              flat_mesh_full_rebuild_ms=st(w(d["flat"])) if len(d["flat"]) else None)
    S["per_frame_nvtx"] = pf
    tr = c.execute("select min(e.start), max(e.end) from NVTX_EVENTS e left join StringIds s on e.textId=s.id "
                   "where coalesce(e.text, s.value) = 'fuser/file_loading'").fetchone()
    S["nvtx_loop_span_s"] = (tr[1] - tr[0]) / 1e9 if tr[0] else None
    # CUDA trace (if captured)
    tabs = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}
    if "CUPTI_ACTIVITY_KIND_KERNEL" in tabs:
        k = c.execute("select count(*), sum(end-start) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()
        top = c.execute("select s.value, count(*), sum(k.end-k.start) from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id "
                        "group by s.value order by 3 desc limit 15").fetchall()
        S["cuda_kernels"] = dict(launches=k[0], total_gpu_ms=k[1]/1e6, top=[dict(name=a, n=b, total_ms=t/1e6) for a, b, t in top])
        if "CUPTI_ACTIVITY_KIND_MEMCPY" in tabs:
            mm = c.execute("select copyKind, count(*), sum(bytes), sum(end-start) from CUPTI_ACTIVITY_KIND_MEMCPY group by copyKind").fetchall()
            kinds = {r[0]: r[1] for r in c.execute("select id, name from ENUM_CUDA_MEMCPY_OPER")}
            S["cuda_memcpy"] = [dict(kind=kinds.get(a, a), n=b, bytes=int(by), total_ms=t/1e6) for a, b, by, t in mm]
        if "CUPTI_ACTIVITY_KIND_MEMSET" in tabs:
            ms = c.execute("select count(*), sum(end-start) from CUPTI_ACTIVITY_KIND_MEMSET").fetchone()
            S["cuda_memset"] = dict(n=ms[0], total_ms=(ms[1] or 0)/1e6)
        if "CUDA_GPU_MEMORY_USAGE_EVENTS" in tabs:
            ev = c.execute("select start, bytes, memKind, memoryOperationType from CUDA_GPU_MEMORY_USAGE_EVENTS order by start").fetchall()
            mk = {r[0]: r[2] for r in c.execute("select id, name, label from ENUM_CUDA_MEM_KIND")}
            live = {}; peak = {}; tot = 0; peak_tot = 0
            loop_end = c.execute("select max(e.end) from NVTX_EVENTS e left join StringIds s on e.textId=s.id where coalesce(e.text, s.value) = 'fuser/time_per_frame'").fetchone()[0] or 0
            peak_loop = 0
            for t_, b, k, op in ev:
                dv = b if op == 0 else -b; live[k] = live.get(k, 0) + dv; peak[k] = max(peak.get(k, 0), live[k])
                tot += dv; peak_tot = max(peak_tot, tot)
                if t_ <= loop_end: peak_loop = max(peak_loop, tot)
            pool = c.execute("select max(localMemoryPoolSize), max(localMemoryPoolUtilizedSize) from CUDA_GPU_MEMORY_POOL_EVENTS").fetchone() if "CUDA_GPU_MEMORY_POOL_EVENTS" in tabs else (None, None)
            S["cuda_device_memory"] = dict(method="nsys --cuda-memory-usage: running sum of CUDA allocation - deallocation events (all kinds) + max cudaMallocAsync pool size",
                peak_live_all_kinds_mb=peak_tot / 2**20, peak_live_loop_mb=peak_loop / 2**20, peak_live_by_kind_mb={mk.get(k, k): v / 2**20 for k, v in peak.items()},
                final_live_mb=tot / 2**20, n_events=len(ev),
                pool_max_size_mb=(pool[0] or 0) / 2**20, pool_max_utilized_mb=(pool[1] or 0) / 2**20)
        if "CUPTI_ACTIVITY_KIND_RUNTIME" in tabs:
            api = c.execute("select s.value, count(*), sum(r.end-r.start) from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id "
                            "group by s.value order by 3 desc limit 12").fetchall()
            S["cuda_api_top"] = [dict(name=a, n=b, total_ms=t/1e6) for a, b, t in api]

# glog anchors: loop wall + outer full-mesh wall
def ts(line):
    m = re.match(r"^I(\d{8}) (\d\d:\d\d:\d\d\.\d+)", line)
    return datetime.strptime(m.group(1) + " " + m.group(2), "%Y%m%d %H:%M:%S.%f").timestamp() if m else None
t_int = t_end = t_gen = t_out = None; oom = []
for line in open(os.path.join(R, "run.log"), errors="replace"):
    if "Integrating" in line and "frames..." in line: t_int = ts(line)
    elif "Ran out of data at frame" in line: t_end = ts(line); S["last_frame_logged"] = int(line.rsplit(":", 1)[1])
    elif "Generating the mesh." in line: t_gen = ts(line)
    elif "Outputting mesh ply file" in line: t_out = ts(line)
    if "out of memory" in line or "CUDA error" in line: oom.append(line.strip()[-200:])
S["cuda_errors"] = oom[:5]
S["loop_wall_s"] = (t_end - t_int) if t_int and t_end else None
S["full_mesh_outer_wall_ms"] = (t_out - t_gen) * 1e3 if t_gen and t_out else None
S["loop_t0_t1"] = [t_int, t_end]
# memory sidecar
mem = np.genfromtxt(os.path.join(R, "mem.csv"), delimiter=",", names=True)
base = float(np.median(mem["sys_used_kb"][mem["vmrss_kb"] < 0][:10]))
alive = mem["vmrss_kb"] > 0
inloop = alive & (mem["t"] >= (t_int or 0)) & (mem["t"] <= (t_end or 1e18))
S["memory"] = dict(method="sidecar 50 ms: system-wide MemTotal-MemAvailable delta vs pre-run median (Orin unified: includes cudaMalloc/nvmap pages, which do NOT appear in VmRSS) + process VmRSS/VmHWM + getrusage maxrss",
                   baseline_sys_used_mb=base/1024, peak_sys_delta_mb=float(mem["sys_used_kb"][alive].max() - base)/1024,
                   peak_sys_delta_inloop_mb=float(mem["sys_used_kb"][inloop].max() - base)/1024 if inloop.any() else None,
                   peak_vmhwm_mb=float(mem["vmhwm_kb"][alive].max())/1024, samples=int(alive.sum()))
tv = json.load(open(os.path.join(R, "timev.json")))
S["memory"]["getrusage_maxrss_mb"] = tv["max_rss_kb"] / 1024; S["process_wall_s"] = tv["wall_s"]; S["rc"] = tv["rc"]
S["memory"]["device_minus_host_est_mb"] = S["memory"]["peak_sys_delta_mb"] - S["memory"]["peak_vmhwm_mb"]
# mesh counts
mp = os.path.join(R, "mesh.ply")
if os.path.exists(mp):
    with open(mp, "rb") as f:
        hdr = b""
        while b"end_header" not in hdr: hdr += f.readline()
    S["mesh_vertices"] = int(re.search(rb"element vertex (\d+)", hdr).group(1)); S["mesh_triangles"] = int(re.search(rb"element face (\d+)", hdr).group(1))
# blocks from serialized map
mpx = os.path.join(R, "map.nvblx")
if os.path.exists(mpx):
    c = sqlite3.connect(mpx)
    nb = c.execute("select count(*) from tsdf_layer_data").fetchone()[0]
    obs = 0; sz = set()
    for (blob,) in c.execute("select data from tsdf_layer_data"):
        a = np.frombuffer(blob, dtype=np.float32); sz.add(a.size)
        obs += int((a[1::2] > 0).sum()) if a.size == 1024 else 0
    S["tsdf_blocks"] = nb; S["tsdf_voxels_allocated"] = nb * 512
    S["tsdf_voxels_observed_weight_gt0"] = obs if sz == {1024} else None; S["tsdf_blob_floats"] = sorted(sz)
    S["color_blocks"] = c.execute("select count(*) from color_layer_data").fetchone()[0]
S["clock_start"] = open(os.path.join(R, "clock_start.txt")).read(); S["clock_end"] = open(os.path.join(R, "clock_end.txt")).read()
S["command"] = open(os.path.join(R, "command.txt")).read().strip()
S["waited"] = open(os.path.join(R, "waited.txt")).read() if os.path.exists(os.path.join(R, "waited.txt")) else ""
json.dump(S, open(os.path.join(R, "summary.json"), "w"), indent=1)
print(json.dumps({k: S[k] for k in ("rc", "loop_wall_s", "full_mesh_outer_wall_ms") if k in S}), json.dumps(S["memory"])[:400])
if "per_frame_nvtx" in S: print({k: (round(v["mean"], 3), round(v["median"], 3), round(v["p95"], 3)) for k, v in S["per_frame_nvtx"].items() if v})
