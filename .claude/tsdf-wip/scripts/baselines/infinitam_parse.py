"""Parse one InfiniTAM run dir (infinitam_run.sh output). CPU only, untimed. Prints one JSON object."""
import json, re, sys, os
import numpy as np
d = sys.argv[1]
out = dict(run_dir=d)
txt = open(os.path.join(d, "stdout.txt")).read()
ft = [float(m.group(2)) for m in re.finditer(r"^frame (\d+): time ([\d.]+), avg", txt, re.M)]
fi = [int(m.group(1)) for m in re.finditer(r"^frame (\d+): time", txt, re.M)]
assert fi == list(range(len(fi))), "frame index gap"
def st(a):
    a = np.asarray(a, float)
    return dict(n=int(a.size), mean=float(a.mean()), median=float(np.median(a)), p95=float(np.percentile(a, 95)),
                min=float(a.min()), max=float(a.max()), std=float(a.std())) if a.size else None
out["frames_integrated"] = len(ft)
out["warmup_frames_discarded"] = 5
out["integrate_e2e_ms"] = st(ft[5:])
out["integrate_e2e_ms_frames0_4"] = ft[:5]
out["integrate_loop_sum_s"] = float(np.sum(ft)) / 1e3
m = re.search(r"^BENCH settings (.*)$", txt, re.M); out["settings_echo"] = m.group(1) if m else None
m = re.search(r"^BENCH construct_ms ([\d.]+)", txt, re.M); out["construct_ms"] = float(m.group(1)) if m else None
mem = {k: float(v) for k, v in re.findall(r"^BENCH mem (\S+) free_mb ([\d.]+)", txt, re.M)}
out["memgetinfo_free_mb"] = mem
if "ctx" in mem:
    out["memgetinfo_delta_mb"] = {k: mem["ctx"] - v for k, v in mem.items() if k != "ctx"}
m = re.search(r"^BENCH blocks allocated (\d+) capacity (\d+)", txt, re.M)
if m: out["blocks_allocated"] = int(m.group(1)); out["block_capacity_preallocated"] = int(m.group(2))
m = re.search(r"^BENCH voxels_observed_w_gt0 (\d+)", txt, re.M); out["voxels_observed_w_gt0"] = int(m.group(1)) if m else None
if "blocks_allocated" in out: out["voxels_allocated"] = out["blocks_allocated"] * 512
m = re.search(r"^BENCH mesh_alloc_ms ([\d.]+) max_triangles (\d+) buffer_mb ([\d.]+)", txt, re.M)
if m: out["mesh_alloc_ms"] = float(m.group(1)); out["mesh_max_triangles"] = int(m.group(2)); out["mesh_buffer_mb"] = float(m.group(3))
mf = re.findall(r"^BENCH mesh_full (cold|warm) (\d+) ms ([\d.]+) triangles (\d+)", txt, re.M)
cold = [float(x[2]) for x in mf if x[0] == "cold"]; warm = [float(x[2]) for x in mf if x[0] == "warm"]
out["mesh_full_cold_ms"] = cold[0] if cold else None
out["mesh_full_warm_ms"] = st(warm) if warm else None
tris = sorted({int(x[3]) for x in mf})
out["mesh_triangles_per_call"] = [int(x[3]) for x in mf]
out["mesh_triangles"] = int(mf[-1][3]) if mf else None
if mf and "mesh_max_triangles" in out:
    out["mesh_buffer_clamped"] = any(int(x[3]) >= out["mesh_max_triangles"] - 1 for x in mf)
m = re.search(r"^BENCH mesh_d2h_ms ([\d.]+) bytes (\d+)", txt, re.M)
if m: out["mesh_d2h_ms"] = float(m.group(1)); out["mesh_d2h_bytes"] = int(m.group(2))
out["mesh_ply"] = os.path.join(d, "mesh.ply") if os.path.exists(os.path.join(d, "mesh.ply")) else None
tv = os.path.join(d, "timev.json")
if os.path.exists(tv):
    t = json.load(open(tv)); out["peak_host_rss_kb"] = t["max_rss_kb"]; out["wall_s"] = t["wall_s"]; out["rc"] = t["rc"]
    out["user_s"] = t["user_s"]; out["sys_s"] = t["sys_s"]
mc = os.path.join(d, "mem.csv")
if os.path.exists(mc):
    a = np.genfromtxt(mc, delimiter=",", names=True)
    base = a["sys_used_kb"][a["vmrss_kb"] < 0][:10]
    out["mem_sampler"] = dict(n=int(a.size), vmhwm_peak_kb=int(a["vmhwm_kb"].max()), vmrss_peak_kb=int(a["vmrss_kb"].max()),
                              sys_used_baseline_kb=float(base.mean()) if base.size else None,
                              sys_used_peak_kb=int(a["sys_used_kb"].max()),
                              sys_used_delta_peak_kb=float(a["sys_used_kb"].max() - base.mean()) if base.size else None)
for k in ("clock_start", "clock_end"):
    p = os.path.join(d, k + ".txt")
    if os.path.exists(p):
        out[k] = dict(l.rstrip("\n").split(": ", 1) for l in open(p) if ": " in l)
w = os.path.join(d, "waited.txt"); out["waited_for_other_jobs"] = open(w).read() if os.path.exists(w) else ""
out["command"] = open(os.path.join(d, "command.txt")).read().strip()
out["exit"] = open(os.path.join(d, "exit.txt")).read().strip()
print(json.dumps(out))
