"""nsys overhead check: same config (office0 5 cm Row M incremental) plain vs nvtx-only vs cuda-trace, built-in timer means excl. the cold max."""
import json
R = "/mnt/data/kr-tsdf-spikes/results/baselines/nvblox/runs/office0_v0.05_M_inc_"
out = {}
for v in ("plain", "nvtx", "cuda"):
    s = json.load(open(f"{R}{v}_n2000/summary.json"))["builtin"]
    out[v] = {t: round(s[t]["mean_excl_max_ms"], 4) for t in ("fuser/integrate_depth", "fuser/integrate_color", "fuser/mesh", "fuser/time_per_frame", "mapper/update_flat_mesh")}
for v in ("nvtx", "cuda"):
    out[f"{v}_vs_plain_pct"] = {t: round(100 * (out[v][t] / out["plain"][t] - 1), 1) for t in out["plain"]}
print(json.dumps(out, indent=1))
