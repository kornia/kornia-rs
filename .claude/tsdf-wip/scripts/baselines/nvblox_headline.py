"""Compact per-row headline from nvblox/results.json (printed as JSON)."""
import json
rows = json.load(open("/mnt/data/kr-tsdf-spikes/results/baselines/nvblox/results.json"))
r2 = lambda x: round(x, 2) if isinstance(x, (int, float)) else x
st = lambda d: {k: r2(d[k]) for k in ("mean", "median", "p95", "min", "max")} if isinstance(d, dict) else d
out = []
for r in rows:
    fm = r["full_mesh_ms"]; sf = r["score_full_mesh"]; si = r["score_incremental_mesh"]
    sc = lambda s: {k: s[k] for k in ("accuracy_cm", "completeness_cm", "completion_ratio_pct", "accuracy_surf_cm", "completeness_surf_cm")} if isinstance(s, dict) else s
    out.append(dict(id=f'{r["scene"]}_{r["voxel_m"]}_{r["row"]}', status=r["status"], integrate_e2e_ms=st(r.get("integrate_e2e_ms")),
        integrate_no_h2d_ms_mean=r2(r["integrate_no_h2d_ms"]["mean"]) if r.get("integrate_no_h2d_ms") else None,
        mesh_incremental_ms=st(r["mesh_incremental_ms"]) if isinstance(r["mesh_incremental_ms"], dict) else r["mesh_incremental_ms"],
        full_mesh_ms=(r2(fm["builtin_mesh_gpu_integrate"]["mean_ms"]) if isinstance(fm, dict) else fm),
        tsdf_blocks=r.get("tsdf_blocks"), triangles=r.get("mesh_triangles"), peak_host_rss_mb=r2(r.get("peak_host_rss_mb")),
        peak_device_mb_loop=r2(r.get("peak_device_mb_loop")), peak_device_mb_total=r2(r.get("peak_device_mb_total")), device_peak_scope=r.get("device_peak_scope"),
        score_full=sc(sf), score_inc=sc(si), sources=r["sources"]))
print(json.dumps(dict(results_json="/mnt/data/kr-tsdf-spikes/results/baselines/nvblox/results.json", meta="/mnt/data/kr-tsdf-spikes/results/baselines/nvblox/meta.json", rows=out)))
