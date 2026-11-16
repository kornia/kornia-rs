import json
for r in json.load(open("/mnt/data/kr-tsdf-spikes/results/baselines/vdbfusion/results.json")):
    s = r["score"]; e = r["integrate_e2e_ms"]; m = r["mem_sampler"]
    print(r["scene"], r["voxel_m"], r["row"], "e2e mean/med/p95/min/max %.1f/%.1f/%.1f/%.1f/%.1f" % (e["mean"], e["median"], e["p95"], e["min"], e["max"]),
          "integ_only med %.1f unproj med %.1f" % (r["integrate_only_ms"]["median"], r["unproject_ms"]["median"]),
          "mesh cold %.0f warm %s" % (r["mesh_full_cold_ms"], [round(x) for x in r["mesh_full_warm_ms"]]),
          "rss_kb %d hwm %d sysdelta_kb %d tree_B %s" % (r["peak_host_rss_kb"], m["vmhwm_peak_kb"], m["sys_used_delta_peak_kb"], r["tree_mem_bytes_tsdf_plus_weights"]),
          "surf acc/comp/cr %.3f/%.3f/%.2f prec %.2f F %.2f kept %.3f" % (s["accuracy_surf_cm"], s["completeness_surf_cm"], s["completion_ratio_surf_pct"], s["precision_pct"], s["f_score_pct"], s["gt_kept_frac"]))
