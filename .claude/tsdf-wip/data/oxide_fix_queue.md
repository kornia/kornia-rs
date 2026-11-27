# After WF-1 CUDA gate (apply on feat/tsdf-submaps, each step bit-exact + ptxas -v)
1. mc_count/mc_emit/raycast stack -> 0 B (Hood.bases raw shared ptr, conf() inline bitmask)
2. #[launch_bounds(256)] on all tsdf kernels (+ gate header test)
3. mc_emit: ballot+popc(lanemask_lt) round-robin edges/cubes, no tid-0 serial scans
4. abort bit when assign_misses/rehash CAS loop exhausts (robustness)
5. BLOCKS_PER_SM comment fix (3-4 resident)
6. step params byval .param (drop htod per step)
7. measure first (nsys): touch warp dedup match_any_i64_sync; raycast per-ray block cache; unchecked_indexing after compute-sanitizer; DynamicSharedArray per B
Docs: apply section (C) of oxide_audit_fcd488e9.md to docs/cuda-oxide.md + kornia-code-review skill
Pin bump 6d47ae4 -> 348895e: deferred, own change, ask user
Rebase wip/touch-par (b933d7ae) after fcd488e9 (alloc.rs/integrate.rs overlap)

## WF-0b notes from baseline-builds (2026-11-15)
- InfiniTAM CLI tracks its own pose (ICP), ignores traj.txt: bench must use GT poses (ITMFileBasedTracker / trackerConfig "type=file") or the row is not like-for-like.
- InfiniTAM depth: int16, converter stores depth_m*3276.75 + affine calib (max ~10 m OK).
- InfiniTAM mesh not run-to-run identical (900 B diff); 1.415M tris vs VDBFusion 60k at 5 cm on 3 frames: check voxel size is actually set per row.
- VDBFusion: traj.txt OpenGL axes -> flip y,z; unverified vs GT.

## Done 2026-11-15: e7876efb (items 1-6) + 6d4a793d (docs)
- WF-2 must A/B vs 3b88b65f: raycast 78->86 regs, mc_emit 72->118 regs (launch_bounds), both 3->2 blocks/SM; forcing 3 blocks spills (raycast 24/40 B, mc_emit 12/12 B).
- Options: launch_bounds(256, minBlocks) per kernel, or drop launch_bounds on mc_emit (96 regs), pick by nsys/ncu.
- BLOCKS_PER_SM comment says raycast ~90-120 regs; actual 86 (fix in WF-2).
