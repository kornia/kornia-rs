# TSDF + submaps: paused work state (2026-11-15)

WIP handoff for this draft PR. **Delete `.claude/tsdf-wip/` before marking ready.**
Snapshot of the scratch state at pause time. On the Orin, `/mnt/data/kr-tsdf-spikes/` is the live
copy (raw runs, meshes, nsys traces); this dir keeps the small files so the work survives without it.

## Goal
TSDF with submaps in kornia-3d: scalar / NEON / cuda-oxide backends, bit-exact with each other.
Must beat nvblox v0.0.10 and Open3D (primary: `t.geometry.VoxelBlockGrid` CUDA) on every axis:
integrate ms/frame e2e, mesh ms (full + incremental), peak memory, accuracy/completeness vs GT.
Multi-sensor, multi-agent, flux-agnostic, `no_std` core, 0 allocs per warmed frame, Python bindings,
fusion example. ONE PR (this one). Full plan: `data/PLAN.md`. Frozen numerics: `data/CONTRACT.v1.md`
(v1.2). Frozen bench params: `data/DECISIONS.md`.

## Where it stands
| Step | State |
|---|---|
| WF-0 spikes + contract v1.2 | done |
| WF-1 scalar / NEON / CUDA + bit-exact gate | done (`*_bit_exact_replica` tests pass on Orin) |
| touch-par (parallel block touch, Simd kernel) | done, 3b88b65f |
| cuda-oxide practices (fix queue 1-6) | done, e7876efb + 6d4a793d |
| WF-0b baselines | **paused**: nvblox, Open3D, VDBFusion measured; InfiniTAM, merge, skeptic not run |
| WF-2 bench + optimize loop | not started |
| WF-3 Python, example, docs, reviews, simplify | not started |

Commits on this branch (oldest first): scalar, guards, pin contract, raycast back-off, NEON, neon fixes,
cuda, cuda fixes, touch-par, oxide practices, oxide docs.

## Baseline numbers (room0 / office0, 2000 frames, decode excluded, MAXN_SUPER, GPU 1020 MHz)
Row M = shared params (trunc 4 vox, depth [0.1, 7), weight cap 64, colour on). Row R = library's own config.
Integrate ms/frame mean:

| | room0 2 cm | room0 5 cm | office0 2 cm | office0 5 cm |
|---|---|---|---|---|
| Open3D VBG CUDA, M | **4.09** | **2.82** | 3.27 | 2.58 |
| Open3D VBG CPU, M | 19.5 | 6.56 | 11.0 | 5.06 |
| nvblox v0.0.10 CUDA, M | 30.8 | 7.52 | see json | see json |
| VDBFusion CPU | ~815 (SUSPECT) | ~811 | ~815 | ~813 |
| ours CPU NEON 6t (30-60 frame scratch run, not comparable yet) | 11.6 | 6.6 | - | - |
| ours CUDA | **never timed** | | | |

Accuracy, room0 2 cm, row M: Open3D 1.49 / 1.40 cm (acc / compl), 99.7 % < 5 cm.
VDBFusion 1.50 / 1.48 cm. nvblox at 5 cm: 1.85 / 1.56 cm.
Memory: nvblox about 1067 MB device. Open3D and VDBFusion are in the json files.
Full data: `data/baselines/*/results.json`. Raw runs: `/mnt/data/kr-tsdf-spikes/results/baselines/<lib>/runs/`.

### Open issues in the baseline data (the skeptic step never ran)
- **VDBFusion ~811-815 ms in every config.** A constant this flat smells like a harness or aggregation
  bug, or a fixed per-frame cost such as Python conversion. Re-check `scripts/baselines/vdb_bench.py`
  and `vdb_aggregate.py` before using it.
- **Open3D row R on room0:** completeness 10-11 cm and 77-81 % < 5 cm, so R skips part of the scene
  (likely a depth-max or config difference). Compare against row M. The skeptic must confirm why.
- **nvblox mesh:** the 2 cm incremental and full colour meshes both OOM (`cudaMallocAsync` 485 MB
  marching-cubes scratch). Reported "not run" with gdb and nsys evidence. 5 cm meshes worked (96 ms full).
- **Open3D mesh-extraction ms:** check that `full_mesh` and incremental timings landed in its results.json.
- **InfiniTAM:** built (`/mnt/data/infinitam-build/Apps/InfiniTAM_cli/InfiniTAM_cli`), smoke OK. The
  bench needs GT poses via the file tracker (`ITMTrackerFactory.h:84`) and a verified voxelSize/mu.
  Scripts were in progress: `scripts/baselines/infinitam_run.sh`, `infinitam_parse.py`. Zero runs done.

## Next steps, in order
1. **Finish WF-0b.** Workflow `resumeFromRunId` only works within the same Claude session, so a later
   session copies `workflows/tsdf-wf0b-baselines-*.js`, drops the agents whose results already exist
   (scorer, nvblox, open3d, vdbfusion), and runs: VDBFusion re-check → InfiniTAM → merge into
   `baseline_targets.json` + `SUMMARY.md` → baseline-skeptic. Quiet box only: no builds during timing.
   Run `sudo jetson_clocks` first (ask the user, never sudo inside agents).
2. **WF-2.** Harness `bench_tsdf.rs` + baselines scripts. Time OUR CUDA first (nsys + ncu), then work
   through `data/learn_from_baselines.md` T1-T9 (pinned in-place input staging, GPU touch dedup via
   shuffle, O(m) miss reset, radix rank on cold frames, emit only nt>0 blocks, ...). Also:
   - the register A/B from `data/oxide_fix_queue.md` (raycast 78→86 and mc_emit 72→118 regs vs
     3 blocks/SM);
   - fix the BLOCKS_PER_SM comment (raycast is 86 regs);
   - the multi-sensor headline.
3. **WF-3.** Python bindings, the `tsdf_multisensor_fusion` example, docs, reviews, simplify pass,
   `pixi run pr-check`, then mark the PR ready.

## Environment (Jetson Orin, 7.4 GB RAM)
- **PTX is untracked.** In every worktree run
  `cargo run -p kornia-oxide-kernels --example build_ptx`; otherwise tests fail with `KernelsUnavailable`.
- **Builds:** at most `-j2`, and each worktree gets its own `CARGO_TARGET_DIR` under
  `/mnt/data/kr-tsdf-targets/`. Never share a target dir between worktrees.
- **Simd rule:** the parallel path runs only inside a rayon pool with more than 1 thread. Outside a pool
  it runs serially, because entering a pool allocates and the 0-alloc test would fail.
- **Open3D CUDA:**
  `PYTHONPATH=/mnt/data/venvs/o3d-cuda-site:/mnt/data/venvs/o3d-cuda-deps /usr/bin/python3.10`
  (no `-I`). Open3D CPU: `/mnt/data/venvs/tsdf-bench/bin/python -I`.
  - The legacy `ScalableTSDFVolume` is broken on aarch64.
- **Baseline builds and sources:** `/mnt/data/{nvblox-build,o3d-cuda-build,vdbfusion-build,infinitam-build}`;
  sources in `/mnt/data/src/{Open3D,nvblox-v0.0.10,InfiniTAM,vdbfusion}`. Build recipes are in `scripts/builds/`.
- **Dataset:** Replica (NICE-SLAM), `/mnt/data/datasets/replica/Replica/{room0,office0}`, depth scale 6553.5.
- **Scratch CPU timing harness** (not part of the PR): `scripts/tsdf_scalar_timing.rs.txt`.
- **Rules:**
  - thiserror only, never anyhow; short comments.
  - Commits end with the `Co-Authored-By: Claude` trailer.
  - No committed PTX.
  - Ask the user before any sudo, apt or docker.

## Layout of this dir
- `data/`: PLAN, CONTRACT v1.2, DECISIONS, oxide audit + fix queue, learn_from_baselines, status.json,
  baselines/*.json
- `scripts/baselines/`: scorer (`score_mesh.py`, validated), per-library run, parse and aggregate scripts
- `scripts/builds/`: nvblox, Open3D-CUDA, VDBFusion and InfiniTAM build scripts + the InfiniTAM patch
- `workflows/`: WF-0, WF-1 and WF-0b workflow scripts (agent prompts, frozen rules)
