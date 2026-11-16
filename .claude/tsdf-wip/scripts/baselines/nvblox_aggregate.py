"""Build nvblox/results.json from per-run summary.json + score.json (CPU only).
Per config, each axis is taken from the first run variant that has it; missing axes get a not_run reason
from nvblox/not_run_notes.json (key "scene_vVOX_ROW" -> {axis: reason})."""
import json, os
B = "/mnt/data/kr-tsdf-spikes/results/baselines/nvblox"; RUNS = f"{B}/runs"
NOTES = json.load(open(f"{B}/not_run_notes.json")) if os.path.exists(f"{B}/not_run_notes.json") else {}
def load(name, f="summary.json"):
    p = f"{RUNS}/{name}/{f}"
    try: return json.load(open(p)), p
    except Exception: return None, p
def first(sc, v, r, variants, pred):
    for var in variants:
        s, p = load(f"{sc}_v{v}_{r}_{var}_n2000")
        if s and pred(s): return s, os.path.dirname(p), var
    return None, None, None
ROWM = ("--projective_integrator_weighting_mode=0 --projective_integrator_max_weight=64 --projective_integrator_truncation_distance_vox=4 "
        "--projective_integrator_max_integration_distance_m=7 --mesh_integrator_min_weight=1")
COVER = ("integrate_e2e = fuser/integrate_depth + fuser/integrate_color (MultiMapper::integrateDepth / integrateColor: view calc, CPU-side block "
         "allocation, block-pointer transfer, kernels, cudaStreamSynchronize) + convert_h2d = (file_loading/depth_image - file_loading/depth_image/stbi) "
         "+ (file_loading/color_image - file_loading/color_image/stbi), i.e. nvblox's own CPU u16->float depth scaling + both synchronous image copies "
         "to device memory. EXCLUDES: PNG/JPG decode (stbi), file IO, pose parsing, the fuser's per-frame updateFlatColorMesh (visualizer path, always on, "
         "reported separately), incremental mesh (reported separately). Per-frame values = NVTX ranges of nvblox's own TimerNvtx (chrono + NVTX, "
         "every timed call ends in a stream sync), frames 0-4 discarded, stats over frames 5..N-1.")
rows = []
for sc in ("room0", "office0"):
    for v in ("0.02", "0.05"):
        for r in ("M", "R"):
            key = f"{sc}_v{v}_{r}"; notes = NOTES.get(key, {})
            row = dict(library="nvblox", version="v0.0.10 (git c457c3fc, clean tree, CMAKE_BUILD_TYPE=Release, sm_87, /mnt/data/nvblox-build/executables/fuse_replica)",
                       scene=sc, voxel_m=float(v), row=r, block_size_vox=8,
                       config=("Row M: " + ROWM if r == "M" else "Row R: nvblox defaults (trunc 4 vox, max_weight 5, weighting_mode 2 inverse_square, max dist 7 m, mesh min_weight 1e-4, mesh every frame)")
                              + "; both rows: --esdf_frame_subsampling=-1 (ESDF off), --visualization=false, colour integration every frame; static TSDF mapping (fuser default)",
                       timer_coverage=COVER, not_run={}, sources={})
            a, pa, va = first(sc, v, r, ["inc_nvtx", "full_nvtx", "nomesh_nvtx"], lambda s: "per_frame_nvtx" in s and s["nvtx_counts"].get("fuser/time_per_frame", 0) == 2000)
            if a:
                pf = a["per_frame_nvtx"]; nfr = a["nvtx_counts"]["fuser/time_per_frame"]
                row["sources"]["timing"] = pa
                row.update(frames_integrated=nfr, command_timing=a["command"], clock_start=a["clock_start"], clock_end=a["clock_end"], waited_for_other_jobs=a["waited"],
                           timing_run_mode=va, integrate_e2e_ms=pf["integrate_e2e_ms"], integrate_no_h2d_ms=pf["integrate_ms"], integrate_depth_ms=pf["integrate_depth_ms"],
                           integrate_color_ms=pf["integrate_color_ms"], convert_h2d_ms=pf["convert_h2d_ms"], tsdf_integrate_ms=pf["tsdf_integrate_ms"],
                           color_integrate_ms=pf["color_integrate_ms"], fuser_time_per_frame_ms=pf["time_per_frame_ms"],
                           flat_mesh_full_rebuild_per_frame_ms=pf["flat_mesh_full_rebuild_ms"], file_loading_excluded_ms=pf["file_loading_ms"],
                           builtin_timers_timing_run={k: a["builtin"].get(k) for k in ("fuser/integrate_depth", "fuser/integrate_color", "fuser/mesh", "fuser/time_per_frame")},
                           outer_wall_loop_s=a["loop_wall_s"], outer_wall_per_frame_ms=(a["loop_wall_s"] / nfr * 1e3 if a["loop_wall_s"] else None),
                           outer_wall_note="glog 'Integrating N frames' -> 'Ran out of data' wall clock / N: whole fuser loop incl. decode, file IO, integrate, incremental mesh (inc mode), always-on flat mesh; under nsys nvtx-only")
                if va == "inc_nvtx": row["mesh_incremental_ms"] = pf["mesh_incremental_ms"]
            else: row["not_run"]["integrate"] = notes.get("integrate", "no timing run found")
            if "mesh_incremental_ms" not in row:
                row["mesh_incremental_ms"] = "not run"; row["not_run"]["mesh_incremental"] = notes.get("mesh_incremental", "no incremental-mesh run")
                pi, ppi = load(f"{key}_inc_nvtx_n2000")
                if pi and "per_frame_nvtx" in pi:
                    row["mesh_incremental_partial"] = dict(frames_completed=pi["nvtx_counts"]["fuser/time_per_frame"], run=os.path.dirname(ppi), cuda_errors=pi["cuda_errors"],
                        mesh_incremental_ms=pi["per_frame_nvtx"]["mesh_incremental_ms"], integrate_e2e_ms=pi["per_frame_nvtx"]["integrate_e2e_ms"],
                        note="incremental-mesh run died with cudaMallocAsync OOM before frame 2000; stats over its completed frames 5..k only, NOT comparable to 2000-frame rows")
            f, pf_, vf = first(sc, v, r, ["full_plain", "full_nvtx"], lambda s: s.get("rc") == 0 and "mesh/gpu/integrate" in s["builtin"])
            if f:
                row["sources"]["full_mesh"] = pf_
                row["full_mesh_ms"] = dict(builtin_mesh_gpu_integrate=f["builtin"]["mesh/gpu/integrate"], outer_wall_glog_ms=f["full_mesh_outer_wall_ms"], n=1, run_mode=vf,
                                           note="single cold call after the loop: Mapper::updateColorMesh over ALL blocks (incl. mesh-block + marching-cubes scratch allocation); the fuser cannot repeat it without a rebuild")
                row.update(mesh_triangles=f.get("mesh_triangles"), mesh_vertices=f.get("mesh_vertices"))
            else: row["full_mesh_ms"] = "not run"; row["not_run"]["full_mesh"] = notes.get("full_mesh", "no successful full-mesh run")
            m, pm, vm = first(sc, v, r, ["full_plain", "nomesh_plain"], lambda s: s.get("rc") == 0 and "tsdf_blocks" in s)
            if m:
                row["sources"]["counts_host_memory"] = pm
                row.update(tsdf_blocks=m["tsdf_blocks"], tsdf_voxels_allocated=m["tsdf_voxels_allocated"], tsdf_voxels_observed=m.get("tsdf_voxels_observed_weight_gt0"),
                           color_blocks=m.get("color_blocks"), peak_host_rss_mb=m["memory"]["peak_vmhwm_mb"], getrusage_maxrss_mb=m["memory"]["getrusage_maxrss_mb"],
                           sys_used_delta_mb_secondary=m["memory"]["peak_sys_delta_mb"], command_counts=m["command"],
                           builtin_timers_plain_run={k: m["builtin"].get(k) for k in ("fuser/integrate_depth", "fuser/integrate_color", "fuser/time_per_frame")})
            else: row["not_run"]["counts_host_memory"] = notes.get("counts_host_memory", "no plain run with map")
            c, pc, vc = first(sc, v, r, ["inc_cuda", "full_cuda", "nomesh_cuda"], lambda s: "cuda_device_memory" in s and s["nvtx_counts"].get("fuser/time_per_frame", 0) == 2000)
            if c:
                cm = c["cuda_device_memory"]; ck = c.get("cuda_kernels", {}); nfc = c["nvtx_counts"]["fuser/time_per_frame"]
                h2d = [x for x in c.get("cuda_memcpy", []) if x["kind"].endswith("HTOD")]
                row["sources"]["cuda_trace"] = pc
                scope = {"inc_cuda": "loop incl. per-frame incremental mesh", "full_cuda": "loop without per-frame mesh / total incl. post-loop full mesh", "nomesh_cuda": "loop, no colour mesh at all"}[vc]
                row.update(peak_device_mb_loop=cm["peak_live_loop_mb"], peak_device_mb_total=cm["peak_live_all_kinds_mb"], device_peak_scope=scope, device_memory=cm, cuda_trace_mode=vc, command_cuda=c["command"], cuda_run_rc=c.get("rc"),
                           gpu_proof=dict(kernel_launches=ck.get("launches"), kernel_gpu_ms_total=ck.get("total_gpu_ms"), top_kernels=ck.get("top", [])[:10],
                                          memcpy=c.get("cuda_memcpy"), memset=c.get("cuda_memset"), frames=nfc,
                                          h2d_gpu_ms_per_frame=(h2d[0]["total_ms"] / nfc if h2d else None), h2d_bytes_per_frame=(h2d[0]["bytes"] / nfc if h2d else None)),
                           cuda_trace_integrate_e2e_ms=c["per_frame_nvtx"]["integrate_e2e_ms"])
            else: row["not_run"]["device_memory"] = notes.get("device_memory", "no cuda-trace run")
            for tag, variants in (("score_full_mesh", ["full_plain", "full_nvtx"]), ("score_incremental_mesh", ["inc_nvtx"])):
                for var in variants:
                    s, p = load(f"{key}_{var}_n2000", "score.json")
                    if s and "error" not in s and not s.get("empty"):
                        row[tag] = {k: s.get(k) for k in ("accuracy_cm", "completeness_cm", "completion_ratio_pct", "precision_pct", "f_score_pct",
                                   "accuracy_surf_cm", "completeness_surf_cm", "completion_ratio_surf_pct", "precision_surf_pct", "accuracy_median_cm",
                                   "completeness_median_cm", "pred_area_m2", "gt_kept_frac", "pred_triangles")}
                        row[tag]["file"] = p; break
                else: row[tag] = "not run"; row["not_run"][tag] = notes.get(tag, "no mesh to score")
            row["status"] = "ok" if not row["not_run"] else "partial"
            rows.append(row)
json.dump(rows, open(f"{B}/results.json", "w"), indent=1)
print(len(rows), "rows;", {r["scene"] + "_" + str(r["voxel_m"]) + r["row"]: (r["status"], list(r["not_run"])) for r in rows})
