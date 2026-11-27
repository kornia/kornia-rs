"""Build open3d/results.json from the run dirs (CPU only, untimed). One row per (scene, voxel, row, device)."""
import csv, glob, json, os
B = "/mnt/data/kr-tsdf-spikes/results/baselines/open3d"; RUNS = f"{B}/runs"
def J(p):
    try: return json.load(open(p))
    except Exception: return None
def T(p):
    try: return open(p).read()
    except Exception: return None
def sampler(p):
    try:
        rows = list(csv.DictReader(open(p)))
    except Exception: return None
    base = [int(r["sys_used_kb"]) for r in rows[:8]]
    live = [r for r in rows if int(r["vmrss_kb"]) > 0]
    if not live: return None
    return dict(vmhwm_kb=max(int(r["vmhwm_kb"]) for r in live),
                sys_used_delta_peak_mb=(max(int(r["sys_used_kb"]) for r in live) - min(base)) / 1024, n_samples=len(live))
TIMER = ("integrate_e2e = time.perf_counter() between two device syncs (o3d.core.cuda.synchronize(); no-op on CPU) around: "
         "depth.to(dev) + color.to(dev) [the H2D copy of the decoded uint16 depth and uint8 RGB images on CUDA, pageable "
         "host memory; a no-op on CPU:0] + vbg.compute_unique_block_coordinates (frustum block keys, unique) + vbg.integrate "
         "(hashmap Activate = block allocation into the preallocated pool, Find, TSDF+colour kernel). One sync at the end only. "
         "EXCLUDES: PNG/JPG decode (o3d.t.io.read_image, reported as decode_ms_excluded), pose parse + inv(c2w), VBG construction "
         "(pool preallocation of block_count blocks, reported as construct_ms), python per-frame bookkeeping after the second sync. "
         "Frames 0-4 discarded (warm-up incl. CUDA lazy module load), stats over frames 5..1999.")
MESH = ("mesh_full = sync, vbg.extract_triangle_mesh(weight_threshold), sync; result device-resident (D2H reported separately "
        "as mesh_d2h_ms, PLY write excluded). cold = first call in the process (on CUDA includes lazy kernel-module load + first "
        "allocations); warm = 3 further calls (median etc.). Over ALL active blocks (VBG has no incremental/dirty-block mesh).")
rows = []
for sd in sorted(glob.glob(f"{RUNS}/*_time_plain_n2000")):
    s = J(f"{sd}/summary.json"); name = os.path.basename(sd)
    if s is None: continue
    dev = "cuda" if s["device"].startswith("CUDA") else "cpu"
    pre = f"{RUNS}/{s['scene']}_v{s['voxel_m']}_{s['row']}"
    size = J(f"{pre}_cuda_size_plain_n2000/summary.json")
    nsd = f"{pre}_cuda_time_nsys_n2000"; nsys = J(f"{nsd}/nsys.json") if dev == "cuda" else None
    nsum = J(f"{nsd}/summary.json") if dev == "cuda" else None
    tv = J(f"{sd}/timev.json"); sm = sampler(f"{sd}/mem.csv"); sc = J(f"{sd}/score.json")
    p = s["params"]
    cfg = (f"Row {s['row']}: VoxelBlockGrid attrs tsdf/weight/color float32, block_resolution 16, voxel {s['voxel_m']} m, "
           f"trunc_voxel_multiplier {p['trunc_voxel_multiplier']} (sdf_trunc {p['sdf_trunc_m']:.3f} m), depth_max {p['depth_max']}, "
           f"depth_scale 6553.5, extract_triangle_mesh weight_threshold {p['mesh_weight_threshold']}, block_count {s['block_count']} "
           f"(= ceil(1.1 x required {size['required_capacity_max_size_plus_frustum'] if size else '?'} / 500) x 500), device {s['device']}, "
           "weight +1 per observation UNCAPPED (VBG has no max_weight), no depth_min in integrate")
    r = dict(library="open3d", component="t.geometry.VoxelBlockGrid", version=s["open3d_version"],
             build=("source build git b6c5e19 (v0.20 release commit), CMAKE_BUILD_TYPE=Release, CUDA 12.6 sm_87, "
                    "ENABLE_CACHED_CUDA_MANAGER=ON, /mnt/data/o3d-cuda-build -> /mnt/data/venvs/o3d-cuda-site, /usr/bin/python3.10, numpy 1.21.5")
                   if dev == "cuda" else "PyPI wheel open3d 0.20.0 aarch64 (CPU only, BUILD_CUDA_MODULE=OFF) in /mnt/data/venvs/tsdf-bench, numpy 2.2.6",
             scene=s["scene"], voxel_m=s["voxel_m"], row=s["row"], device=s["device"], block_size_vox=16, config=cfg,
             params=p, command=T(f"{sd}/command.txt").strip(), timer_coverage=TIMER, mesh_timer_coverage=MESH,
             frames_integrated=s["frames_integrated"], warmup_frames_discarded=s["warmup_frames_discarded"],
             integrate_e2e_ms=s["integrate_e2e_ms"], decode_ms_excluded=s["decode_ms_excluded"], construct_ms=s["construct_ms"],
             n_frustum_blocks_per_frame=s["n_frustum_blocks"],
             mesh_full_cold_ms=s.get("mesh_full_cold_ms"), mesh_full_warm_ms=s.get("mesh_full_warm_ms"), mesh_d2h_ms=s.get("mesh_d2h_ms"),
             mesh_incremental_ms="n/a (VoxelBlockGrid has no incremental / dirty-block meshing; only full extraction over all active blocks)",
             ray_cast=s.get("ray_cast"),
             block_count_capacity=s["block_count"], block_count_required_sizing=size["required_capacity_max_size_plus_frustum"] if size else None,
             block_count_active=s["block_count_active"], hash_capacity_start=s["hash_capacity_start"], hash_capacity_end=s["hash_capacity_end"],
             hashmap_grew_during_timed_run=bool(s["capacity_growth_events"]),
             voxels_allocated=s.get("voxels_allocated"), voxels_observed_weight_gt0=s.get("voxels_observed_weight_gt0"),
             mesh_vertices=s.get("mesh_vertices"), mesh_triangles=s.get("mesh_triangles"), mesh_ply=s.get("mesh_ply"),
             host_memory=dict(peak_rss_kb_getrusage=tv["max_rss_kb"] if tv else None, peak_vmhwm_kb_sampler=sm["vmhwm_kb"] if sm else None,
                              wall_s=tv["wall_s"] if tv else None,
                              method="getrusage(RUSAGE_CHILDREN).ru_maxrss via timev.py (stand-in for /usr/bin/time -v, not installed) + 50 ms /proc VmHWM sidecar"),
             device_memory=None, score=None, clock_start=T(f"{sd}/clock_start.txt"), clock_end=T(f"{sd}/clock_end.txt"),
             waited_for_other_jobs=T(f"{sd}/waited.txt") or "", sources=dict(run_dir=sd, summary=f"{sd}/summary.json",
             frames_csv=f"{sd}/frames.csv", score=f"{sd}/score.json", sizing=f"{pre}_cuda_size_plain_n2000/summary.json"))
    if dev == "cuda":
        r["device_memory"] = dict(
            nsys=nsys.get("device_memory") if nsys else None,
            memgetinfo=s.get("cuda_memgetinfo"),
            sys_used_delta_peak_mb_sampler=sm["sys_used_delta_peak_mb"] if sm else None,
            method=("primary: nsys --cuda-memory-usage running sum of allocs - frees (all CUDA memory kinds; Open3D uses cudaMallocAsync "
                    "+ its cached manager, so freed blocks stay counted until process end) from the separate nsys run of the same config; "
                    "secondary: in-process cudaMemGetInfo free-memory drop from before VBG construction to the minimum sampled (every 50 frames "
                    "+ after each mesh call), which on Orin unified memory is system-wide physical commitment; "
                    "tertiary: MemTotal-MemAvailable sidecar delta (under-reads on this Orin, nvmap page pool)."))
        r["gpu_path_proof"] = dict(nsys_dir=nsd, kernel_launches_total=nsys.get("kernel_launches_total") if nsys else None,
                                   kernels_per_frame=nsys.get("kernels_per_frame") if nsys else None,
                                   kernel_launches_by_phase=nsys.get("kernel_launches_by_phase") if nsys else None,
                                   top_kernels=(nsys or {}).get("top_kernels", [])[:8], memcpy=(nsys or {}).get("memcpy"),
                                   nsys_frame_ms_median=(nsys or {}).get("frame_nvtx_ms_under_nsys"),
                                   nsys_run_integrate_e2e_median_ms=nsum["integrate_e2e_ms"]["median"] if nsum else None,
                                   nsys_run_mesh_triangles=nsum.get("mesh_triangles") if nsum else None)
    else:
        r["device_memory"] = "n/a (CPU:0 run, no CUDA context; the voxel pool is host memory and is inside peak RSS)"
    if sc:
        r["score"] = {k: sc.get(k) for k in ("accuracy_cm", "completeness_cm", "completion_ratio_pct", "precision_pct", "f_score_pct",
                     "accuracy_surf_cm", "completeness_surf_cm", "completion_ratio_surf_pct", "precision_surf_pct", "accuracy_median_cm",
                     "completeness_median_cm", "pred_area_m2", "gt_kept_frac", "empty", "n_points", "seed_pred", "seed_gt", "scorer_sha256")}
        r["score"]["command"] = T(f"{sd}/score_cmd.txt")
    req = r["block_count_required_sizing"]
    r["pool_slack_mb"] = (r["block_count_capacity"] - req) * 16**3 * 20 / 2**20 if req else None
    # in-loop growth cost seen in the (untimed-for-headline) sizing pass: e2e ms of the frames where capacity doubled
    try:
        fr = list(csv.DictReader(open(f"{pre}_cuda_size_plain_n2000/frames.csv"))); g = []
        for a_, b_ in zip(fr, fr[1:]):
            if b_["hash_capacity"] != a_["hash_capacity"]:
                g.append(dict(frame=int(b_["frame"]), new_capacity=int(b_["hash_capacity"]), e2e_ms=float(b_["e2e_ms"])))
        r["sizing_pass_growth_frames_cuda"] = g
    except Exception: r["sizing_pass_growth_frames_cuda"] = None
    if dev == "cuda" and nsys:
        d = nsys["device_memory"]
        r["device_memory_headline_mb"] = max(d["peak_live_during_integrate_loop_mb"], d.get("peak_live_during_mesh_mb") or 0)
        r["device_memory_headline_def"] = ("max(nsys live peak during integrate loop, during the 4 mesh calls); comparable cross-library number. "
                                           "ray_cast peak reported separately (device_memory.nsys.peak_live_during_ray_cast_mb). Do NOT add host RSS: "
                                           "CUDA-row RSS starts at ~280-290 MB of CUDA runtime/driver before VBG construction and it is unverified "
                                           "whether nvmap device pages also appear in RSS on Orin.")
    rows.append(r)
json.dump(rows, open(f"{B}/results.json", "w"), indent=1)
print(len(rows), "rows")

# ---- not-run rows + meta (appended) ----
side_cpu = J(f"{B}/side_checks_cpu_wheel.json"); side_cuda = J(f"{B}/side_checks_cuda_build.json")
for build, sc_ in (("PyPI wheel 0.20.0 aarch64 (CPU)", side_cpu), ("source build 0.20.0+b6c5e19 (CUDA build, legacy code is CPU)", side_cuda)):
    rows.append(dict(library="open3d", component="pipelines.integration.ScalableTSDFVolume (legacy)", version=sc_["open3d"] if sc_ else None,
                     build=build, scene="room0,office0", voxel_m="0.02,0.05", row="M,R", device="CPU:0",
                     not_run=("not run: legacy ScalableTSDFVolume is broken on this aarch64 box: 5 room0 frames at 2 cm give "
                              f"{(sc_ or {}).get('legacy_scalable_5frames_v0.02')} (0 mesh vertices, 0 points); seen in BOTH the PyPI "
                              "wheel and the CUDA source build, so it is not wheel-specific. Not a baseline row (DECISIONS.md)."),
                     sources=dict(side_check=f"{B}/side_checks_{'cpu_wheel' if 'PyPI' in build else 'cuda_build'}.json",
                                  script="/mnt/data/kr-tsdf-spikes/results/baselines/scripts/o3d_side_checks.py")))
rows.append(dict(library="open3d", component="t.geometry.VoxelBlockGrid", axis="multi-sensor fused step (4 cams + 2 LiDARs, headline)",
                 not_run="not run: this step has only single-camera Replica RGB-D data, and VoxelBlockGrid has no LiDAR / spherical integrate path "
                         "(integrate takes one pinhole depth image + intrinsic per call); a multi-camera rig would be N sequential integrate calls. "
                         "No multi-sensor dataset/rig was specified for WF-0b."))
rows.append(dict(library="open3d", component="pipelines.integration.UniformTSDFVolume (legacy)", not_run="not run: optional per DECISIONS.md "
                 "(Baselines: 'UniformTSDFVolume optional'); legacy ScalableTSDFVolume already yields 0 vertices on this box (see the legacy row)."))
json.dump(rows, open(f"{B}/results.json", "w"), indent=1)
meta = dict(
    library="open3d",
    box="Jetson Orin, 7.4 GB RAM, MAXN_SUPER (nvpmodel 2), jetson_clocks: GPU 1020 MHz locked (cur=min=max), CPU 1728 MHz all 6 cores; verified in all 68 clock_start/clock_end snapshots",
    camera_convention="traj.txt c2w AS-IS (OpenCV), extrinsic = inv(c2w); pilot 50-frame room0 5 cm accuracy_surf 0.33 cm confirms",
    depth_range=dict(file="/mnt/data/kr-tsdf-spikes/results/baselines/nvblox/gate/depth_range_count.json",
                     note="VBG integrate has no depth_min (only depth<=0 || depth>depth_max); Row M [0.1,7.0) still met exactly: 0 valid pixels <0.1 m or >=7 m in either scene"),
    weight_cap="Row M weight cap 64 NOT matched: VBG integrate does weight+1 uncapped (VoxelBlockGridImpl.h:302). Not patchable without a rebuild (not allowed).",
    mesh_threshold_semantics="extract_triangle_mesh skips a voxel when weight <= weight_threshold (VoxelBlockGridImpl.h:1215/1226), so thr 1 needs weight >= 2, thr 3 needs >= 4",
    rowR_zero_vertex=dict(files=[f"{B}/side_checks_cpu_wheel.json", f"{B}/side_checks_cuda_build.json"],
                          note="Row R (thr 3): 0 vertices after 1-3 frames, 4272 after 4 (room0 5 cm); Row M thr 1: 0 after 1 frame. Irrelevant for 2000-frame rows (all non-empty)."),
    block_count_sizing="untimed CUDA sizing pass per (scene,voxel,row) with block_count 1000 and auto-grow; required = max_i(size_before_i + n_frustum_i) because HashMap::Activate reserves when size+len(keys) > capacity; timed block_count = ceil(1.1*required/500)*500; capacity verified unchanged in every timed run",
    timing_method=TIMER, mesh_method=MESH,
    raycast_method="ray_cast(frustum blocks of frame 1999, K, inv(c2w_1999), 1200, 680, ['depth','normal','color'], 6553.5, depth_min 0.1, depth_max row, weight_threshold 1 (t_reconstruction_system/ray_casting.py), trunc row, range_map_down_factor 8); 5 warm-up calls discarded, 50 timed, sync before/after each",
    device_memory_headline="device_memory_headline_mb = max(nsys live peak during integrate loop, during mesh); ray_cast peak separate (nvblox has no ray_cast phase); never add host RSS to it",
    memory_method="host: getrusage maxrss (timev.py) + /proc VmHWM sidecar; device (CUDA rows): nsys --cuda-memory-usage live allocation sum in the separate nsys run of the same config (primary), cudaMemGetInfo drop and MemAvailable delta secondary",
    gpu_proof="every CUDA config has an nsys -t cuda,nvtx run: 12 kernel launches per integrated frame (24006 in 2000 'frame' NVTX ranges), ElementWiseKernel_ (integrate + unique), STDGPUInsertKernel, STDGPUFindKernel, CUB static_kernel; H2D 8.16 GB over 14223 memcpys for room0 2 cm (= 2000 x (1.63 MB depth + 2.45 MB colour) + small)",
    cpu_vs_cuda_check="CPU and CUDA rows give identical block counts and triangle counts (room0 2 cm R: 667907 vs 667908) and accuracy within 0.02 cm -> GPU output checked against CPU",
    decode_note="decode is excluded for both devices; decode is slower in the CUDA-build process (system python3.10 + numpy 1.21, ~75 ms/frame) than in the wheel venv (~32 ms); it is outside the timer",
    hygiene="no compile / no other benchmark during any timed run (comm-based check heavy_procs.sh before each run; waited.txt only for one sizing run, caused by my own wait loop matching the old pgrep -f pattern; fixed). The clock_state.sh 'heavy_procs' field lists only shell wrappers/sleep loops (pgrep -f text matches), no compiler or benchmark process. Scoring and nsys parsing ran only after all timed runs.",
    scripts=sorted(glob.glob("/mnt/data/kr-tsdf-spikes/results/baselines/scripts/o3d_*")) + ["/mnt/data/kr-tsdf-spikes/results/baselines/scripts/heavy_procs.sh"])
json.dump(meta, open(f"{B}/meta.json", "w"), indent=1)
print(len(rows), "rows incl. not-run")
