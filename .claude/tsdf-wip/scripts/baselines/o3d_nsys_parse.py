"""Parse an Open3D-runner nsys sqlite (-t cuda,nvtx --cuda-memory-usage=true). CPU only, untimed. Prints one JSON:
kernel launches total and per NVTX phase (frame / mesh / ray_cast / construct), top kernels, memcpy by kind,
device memory = running sum of CUDA_GPU_MEMORY_USAGE_EVENTS allocations - frees (all memory kinds): live at loop start,
peak during integrate loop, peak overall; plus cudaMallocAsync pool sizes if recorded."""
import json, sqlite3, sys, collections
import numpy as np
c = sqlite3.connect(sys.argv[1])
tabs = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}
rng = c.execute("select e.start, e.end, coalesce(e.text, s.value) from NVTX_EVENTS e left join StringIds s on e.textId = s.id "
                "where e.end is not null order by e.start").fetchall()
ph = collections.defaultdict(list)
OWN = ("construct", "frame", "mesh", "ray_cast")  # runner's own NVTX ranges; CUB's nested ranges are ignored
for s, e, n in rng:
    if n in OWN: ph[n].append((s, e))
cub = collections.Counter(n for _, _, n in rng if n not in OWN)
out = dict(nvtx_ranges={k: len(v) for k, v in ph.items()}, other_nvtx_ranges=dict(cub))
k = c.execute("select k.start, k.end, k.correlationId, s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName = s.id order by k.start").fetchall()
# attribute kernels to the NVTX phase of their launching runtime call (correlationId -> CUPTI_ACTIVITY_KIND_RUNTIME start)
rt = dict(c.execute("select correlationId, start from CUPTI_ACTIVITY_KIND_RUNTIME").fetchall())
def phase_of(t):
    for n, v in ph.items():
        st = np.array([x[0] for x in v]); i = np.searchsorted(st, t) - 1
        if i >= 0 and v[i][0] <= t <= v[i][1]: return n
    return "outside_nvtx"
cnt = collections.Counter(); dur = collections.Counter(); bykern = collections.Counter(); bykdur = collections.Counter()
for s, e, cid, name in k:
    p = phase_of(rt.get(cid, s)); cnt[p] += 1; dur[p] += e - s; bykern[(p, name)] += 1; bykdur[(p, name)] += e - s
out["kernel_launches_total"] = len(k)
out["kernel_launches_by_phase"] = dict(cnt); out["kernel_gpu_ms_by_phase"] = {p: v / 1e6 for p, v in dur.items()}
nf = len(ph.get("frame", [])) or 1
out["kernels_per_frame"] = cnt.get("frame", 0) / nf
out["top_kernels"] = [dict(phase=p, kernel=n, launches=bykern[(p, n)], gpu_ms=bykdur[(p, n)] / 1e6)
                      for (p, n), _ in bykdur.most_common(15)]
if "frame" in ph:
    fr = ph["frame"]; out["frame_nvtx_ms_under_nsys"] = dict(median=float(np.median([(e - s) / 1e6 for s, e in fr[5:]])), n=len(fr) - 5)
if "CUPTI_ACTIVITY_KIND_MEMCPY" in tabs:
    kinds = {r[0]: r[1] for r in c.execute("select id, name from ENUM_CUDA_MEMCPY_OPER")}
    out["memcpy"] = [dict(kind=kinds.get(a, a), count=b, bytes=int(cc or 0), ms=(d or 0) / 1e6) for a, b, cc, d in
                     c.execute("select copyKind, count(*), sum(bytes), sum(end-start) from CUPTI_ACTIVITY_KIND_MEMCPY group by copyKind")]
if "CUDA_GPU_MEMORY_USAGE_EVENTS" in tabs:
    ev = c.execute("select start, bytes, memoryOperationType from CUDA_GPU_MEMORY_USAGE_EVENTS order by start").fetchall()
    t = np.array([x[0] for x in ev]); live = np.cumsum([b if op == 0 else -b for _, b, op in ev])
    def live_at(ts):
        i = np.searchsorted(t, ts, side="right") - 1; return float(live[i]) / 2**20 if i >= 0 else 0.0
    f0 = ph["frame"][0][0] if "frame" in ph else None; f1 = ph["frame"][-1][1] if "frame" in ph else None
    d = dict(peak_live_mb=float(live.max()) / 2**20, final_live_mb=float(live[-1]) / 2**20, n_events=len(ev),
             largest_alloc_mb=max((b for _, b, op in ev if op == 0), default=0) / 2**20)
    if f0:
        d["live_after_construct_before_first_frame_mb"] = live_at(f0)
        m = (t >= f0) & (t <= f1); d["peak_live_during_integrate_loop_mb"] = float(live[m].max()) / 2**20 if m.any() else live_at(f1)
        d["live_end_of_integrate_loop_mb"] = live_at(f1)
    if "mesh" in ph:
        m = (t >= ph["mesh"][0][0]) & (t <= ph["mesh"][-1][1]); d["peak_live_during_mesh_mb"] = float(live[m].max()) / 2**20 if m.any() else None
    if "ray_cast" in ph:
        m = (t >= ph["ray_cast"][0][0]) & (t <= ph["ray_cast"][-1][1]); d["peak_live_during_ray_cast_mb"] = float(live[m].max()) / 2**20 if m.any() else live_at(ph["ray_cast"][-1][1])
        m = t <= ph["ray_cast"][-1][1]; d["peak_live_through_last_ray_cast_mb"] = float(live[m].max()) / 2**20
        d["note"] = "peak_live_through_last_ray_cast_mb = library workload peak (construct+integrate+4 mesh calls+55 ray_cast calls); peak_live_mb also includes the runner's post-hoc voxel counting"
    if "CUDA_GPU_MEMORY_POOL_EVENTS" in tabs:
        p = c.execute("select max(localMemoryPoolSize), max(localMemoryPoolUtilizedSize) from CUDA_GPU_MEMORY_POOL_EVENTS").fetchone()
        d["mempool_max_size_mb"] = (p[0] or 0) / 2**20; d["mempool_max_utilized_mb"] = (p[1] or 0) / 2**20
    out["device_memory"] = d
print(json.dumps(out))
