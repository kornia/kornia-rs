"""Build vdbfusion/results.json (one row per scene x voxel x row) from the run dirs. Untimed."""
import json, os, csv, glob
B = "/mnt/data/kr-tsdf-spikes/results/baselines/vdbfusion"
def rd(p, default=None):
    try: return open(p).read()
    except OSError: return default
def clock(p):
    t = rd(p, ""); keep = ("date", "nvpmodel", "gpu_freq", "cpu_freq", "loadavg", "heavy_procs", "thermal")
    return {l.split(":", 1)[0]: l.split(":", 1)[1].strip() for l in t.splitlines() if l.split(":", 1)[0] in keep}
def mem(p):
    rows = list(csv.DictReader(open(p)))
    base = [int(r["sys_used_kb"]) for r in rows if r["vmrss_kb"] == "-1"][:10]
    run = [r for r in rows if r["vmrss_kb"] != "-1"]
    b0 = sum(base) / len(base)
    return dict(samples=len(run), period_s=0.05, vmhwm_peak_kb=max(int(r["vmhwm_kb"]) for r in run),
                vmrss_peak_kb=max(int(r["vmrss_kb"]) for r in run), sys_used_baseline_kb=round(b0),
                sys_used_peak_kb=max(int(r["sys_used_kb"]) for r in run),
                sys_used_delta_peak_kb=round(max(int(r["sys_used_kb"]) for r in run) - b0))
ROWCFG = {"M": "Row M (matched): voxel {v} m, sdf_trunc 4 voxels = {t:.3f} m, space_carving False, z-depth range [0.1, 7.0) m, constant weight +1 per observation UNCAPPED (VDBFusion has no max_weight; frozen cap 64 cannot be expressed), NO colour (VDBFusion has no colour channel), mesh extract_triangle_mesh(fill_holes=True, min_weight=1.0)",
          "R": "Row R (recommended): same integration as Row M (upstream default constant weight 1, space_carving False, preprocess = range filter; sdf_trunc fixed at 4 voxels by the bench, upstream RGB-D configs use 3 (ICL) / 7 (TUM) voxels), NO colour, mesh extract_triangle_mesh(fill_holes=True, min_weight=5.0) as in upstream examples/python/config/icl.yaml and tum.yaml"}
TIMER = ("integrate_e2e = time.perf_counter() around: uint16 depth -> metres, z-range mask, unprojection with precomputed per-pixel ray factors, "
         "camera->world transform (float64 Nx3; VDBFusion needs WORLD-frame points, it uses only extrinsic[:3,3] as ray origin) + VDBVolume.integrate "
         "(pybind copy numpy -> std::vector<Eigen::Vector3d>, then sequential per-point DDA ray update over [d - trunc, d + trunc] with on-demand "
         "OpenVDB node allocation). CPU-only, no H2D. integrate_only = the VDBVolume.integrate call alone (incl. the pybind copy). EXCLUDES: PNG decode "
         "(cv2.imread, reported as decode_ms_excluded), pose parse (np.loadtxt once), per-pixel ray-factor grid (computed once), volume construction (construct_ms). "
         "Frames 0-4 discarded as warm-up, stats over frames 5..1999.")
rows = []
for d in sorted(glob.glob(f"{B}/runs/*_n2000")):
    bj = json.load(open(f"{d}/bench.json")); tv = json.load(open(f"{d}/timev.json")); m = mem(f"{d}/mem.csv")
    meta = json.loads(rd(f"{d}/vdb_meta.json", "{}") or "{}")
    vc = [r["value"] for r in meta.get("records", []) if r["key"] == "file_voxel_count"]
    mb = [r["value"] for r in meta.get("records", []) if r["key"] == "file_mem_bytes"]
    for row in ("M", "R"):
        mj = bj["meshes"][row]; sc = json.loads(rd(f"{d}/score_{row}.json", "null") or "null")
        rows.append(dict(
            library="vdbfusion", version="0.1.6", commit="vdbfusion 1b22ae8 (/mnt/data/src/vdbfusion) + OpenVDB fork nachovizzo/openvdb nacho/vdbfusion c110123 (9.1.1-based, static core)",
            build="/mnt/data/kr-tsdf-spikes/vdbfusion_build.sh (Release, -j1), pybind module in /mnt/data/venvs/vdbfusion, /usr/bin/python3.10, numpy+cv2 system",
            scene=bj["scene"], voxel_m=bj["voxel_m"], row=row, device="CPU, 1 thread (upstream Integrate uses sequential std::for_each; TBB only inside OpenVDB internals)",
            colour="off (not supported by VDBFusion)",
            config=ROWCFG[row].format(v=bj["voxel_m"], t=bj["sdf_trunc_m"]),
            params=dict(sdf_trunc_m=bj["sdf_trunc_m"], trunc_voxels=4, space_carving=False, depth_range_m=bj["depth_range_m"], depth_range_kind=bj["depth_range_kind"],
                        integration_weight="constant 1, uncapped", mesh_min_weight=mj["min_weight"], fill_holes=True, poses="traj.txt c2w as-is (OpenCV axes), no flip"),
            shared_integration=f"ONE integration run serves rows M and R (identical integration params; rows differ only in mesh min_weight) -> integrate timings and memory are the same measurement for both rows ({d})",
            command=rd(f"{d}/command.txt", "").strip(), run_dir=d,
            clock_start=clock(f"{d}/clock_start.txt"), clock_end=clock(f"{d}/clock_end.txt"),
            timer_coverage=TIMER,
            frames_integrated=bj["frames_integrated"], warmup_frames_discarded=bj["warmup_frames_discarded"],
            integrate_e2e_ms=bj["integrate_e2e_ms"], integrate_only_ms=bj["integrate_only_ms"], unproject_ms=bj["unproject_ms"],
            decode_ms_excluded=bj["decode_ms_excluded"], points_per_frame=bj["points_per_frame"], integrate_loop_wall_s=bj["integrate_loop_wall_s"],
            construct_ms=bj["construct_ms"],
            mesh_threshold_semantics="VDBFusion MarchingCubes.cpp: a cube is dropped if ANY of its 8 corners has weight < min_weight (strict <), so kept corners have weight >= min_weight (>=1 for M, >=5 for R); weights are integer counts (constant +1 per observation)", mesh_timer_coverage="perf_counter around vdbfusion extract_triangle_mesh(fill_holes=True, min_weight) (C++ marching cubes over all active voxels + pybind conversion to numpy); PLY write excluded. cold = first call for this row (after the other row's calls for row R), warm = 2 further calls.",
            mesh_full_cold_ms=mj["mesh_full_cold_ms"], mesh_full_warm_ms=mj["mesh_full_warm_ms"], mesh_full_warm_median_ms=mj["mesh_full_warm_median_ms"],
            mesh_incremental_ms="n/a (VDBFusion has no incremental / dirty-block meshing)",
            peak_host_rss_kb=tv["max_rss_kb"], peak_host_rss_method="timev.py getrusage(RUSAGE_CHILDREN).ru_maxrss (GNU /usr/bin/time not installed); whole process incl. mesh extraction + .vdb write + numpy buffers",
            mem_sampler=m, device_memory="n/a (CPU-only library). Unified-memory proxy: system MemTotal-MemAvailable sampled every 50 ms by scripts/mem_sampler.py, peak minus the mean of the pre-run baseline samples = mem_sampler.sys_used_delta_peak_kb",
            gpu_path_proof="n/a (CPU library)",
            active_voxel_count=vc[0] if vc else None, active_voxel_count_method="OpenVDB file_voxel_count metadata of the TSDF grid in grids.vdb (written by extract_vdb_grids after integration), parsed by scripts/vdb_meta.py",
            tree_mem_bytes_tsdf_plus_weights=sum(mb) if mb else None,
            block_count="n/a (OpenVDB 8^3 leaf-node count is not exposed by the vdbfusion pybind module without pyopenvdb; not estimated)",
            vertices=mj["vertices"], triangles=mj["triangles"], mesh_ply=mj["ply"],
            score=sc, score_cmd=rd(f"{d}/score_{row}_cmd.txt", "").strip(), score_log=f"{d}/score_{row}.json",
            wall_s=tv["wall_s"], rc=tv["rc"], cpu_over_wall=round((tv["user_s"] + tv["sys_s"]) / tv["wall_s"], 3), cpu_over_wall_note="(user+sys)/wall of the whole process from timev.json; ~1.0 or below = single-threaded", logs=dict(run_log=f"{d}/run.log", bench=f"{d}/bench.json", per_frame=f"{d}/per_frame.npz", timev=f"{d}/timev.json", mem=f"{d}/mem.csv", waited=f"{d}/waited.txt" if os.path.exists(f"{d}/waited.txt") else None)))
json.dump(rows, open(f"{B}/results.json", "w"), indent=1)
for r in rows:
    s = r["score"] or {}
    print(r["scene"], r["voxel_m"], r["row"], "e2e med %.1f p95 %.1f" % (r["integrate_e2e_ms"]["median"], r["integrate_e2e_ms"]["p95"]),
          "mesh %.0f" % r["mesh_full_warm_median_ms"], "rss MB %.0f" % (r["peak_host_rss_kb"] / 1024), "vox", r["active_voxel_count"], "tri", r["triangles"],
          "acc", s.get("accuracy_cm"), "comp", s.get("completeness_cm"), "cr", s.get("completion_ratio_pct"))
