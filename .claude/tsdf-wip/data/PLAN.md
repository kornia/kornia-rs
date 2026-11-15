# TSDF + submaps in kornia-3d — scalar / NEON / cuda-oxide, multi-sensor, Python, benched vs Open3D + nvblox

Executed as a chain of 4 dynamic workflows (Workflow tool), main loop reads each result and gates the next.

## Context
Motivation (only): flux will later fuse many robots × many sensors into one dense world (furniture
boxes #140, occupancy, frame-to-model ICP). kornia-3d has no TSDF. Build a generic, deterministic,
block-hashed TSDF library with **submaps first-class** (loop closure moves submaps rigidly, no
re-integration), **multi-sensor** (pinhole cameras + spherical LiDARs per step) and **multi-agent**
(many rigs into one map), three backends under one bit-exact contract (scalar, NEON, cuda-oxide),
Python bindings, a fusion example, benches that beat nvblox + Open3D VoxelBlockGrid on this Orin.

## Delivery: ONE PR (hard rule)
Everything lands in a single kornia-rs PR on branch `feat/tsdf-submaps`: kornia-3d `tsdf` module
(scalar + NEON + CUDA launchers), oxide kernels in `kornia-oxide-kernels/device/src/tsdf.rs`,
`io/ply` mesh writer, kornia-py bindings + stubs + tests + benchmark, Rust example + Python twin,
benches (`bench_tsdf.rs`, baselines scripts, `bench_nvblox.cpp`), `docs/benchmark-tsdf.md`,
changelog fragment. No follow-up or stacked PRs; worktree branches from WF-1/WF-3 are merged back into
`feat/tsdf-submaps` locally, never pushed as their own PRs. Commits per layer (core / neon / cuda /
python / example / bench / simplify) for reviewability. Opened as draft once WF-3 is green, marked
ready only after the bench table wins on every axis.

## Flux-agnostic, multi-robot ready (hard rule)
- Zero flux concepts in code, docs, tests, example: no flux crates/types, no zenoh/ROS/topics, no
  site/robot/device names, no flux file formats. Inputs are plain kornia types (`Image`, `DepthImage`,
  `PinholeCamera`, `Pose3d`, numpy) — a flux adapter is a later, separate crate in the flux repo.
- Generic multi-agent model:
  - `AgentId(u32)` / `SensorId(u32)` opaque ids; `SensorRig { sensors: Vec<(SensorId, RangeSensor,
    body_from_sensor: Pose3d)> }` per agent (rigs may differ per agent).
  - `SubmapTsdf::integrate(agent, rig_frame: &[SensorFrame], world_from_body, stamp: u64)`; submaps
    keyed `SubmapId { agent, seq }`, one active submap per agent, spawn policy per agent.
  - Deterministic regardless of arrival interleaving: a step is applied in order
    `(stamp, agent, sensor)`; `integrate_batch(&[AgentStep])` sorts before applying, so two robots
    streaming concurrently give the same map bits as any serial replay.
  - Cross-agent alignment is the caller's: `set_submap_poses` takes poses from any optimizer
    (`kornia_3d::pgo` with intra- and inter-agent edges in the example); the TSDF never owns SLAM.
  - Queries over the union: `extract_mesh(filter: agents/submaps)`, `raycast` across overlapping
    submaps (max-weight / weighted fusion of overlapping submaps, documented rule), `memory_bytes`
    per agent; residency budget shared across agents.
- Example uses **2 synthetic rigs** in Replica (rig A: 4 cams + 1 LiDAR, rig B: 2 cams + 1 LiDAR,
  different trajectories), inter-rig loop closure via `pgo`, `--agents A,B --sensors ...`.
- Tests: multi-agent interleave invariance (shuffled arrival == sorted serial, bitwise); per-agent
  submap isolation; inter-agent pose update moves only that agent's submaps.

## Decisions (user, 2026-11-14)
- Dataset: Replica (NICE-SLAM preprocessed, room0 + office0), `/mnt/data/datasets/replica`, GT meshes.
- Example: multi-camera + multi-LiDAR fusion. Benches: single-cam head-to-head AND multi-sensor.
- Baselines: Open3D CPU wheel; Open3D CUDA source build; nvblox from prebuilt packages (no source build)
  if they exist for R36.4.3 — else Isaac ROS docker, else source build.
- Python bindings in scope.
- 2026-11-14 (user): add **VDBFusion** (OpenVDB, CPU; the LiDAR TSDF reference) and **InfiniTAM v3**
  (CUDA voxel hashing; the classic GPU reference) to WF-0b. Both source builds, no sudo (deps via
  `/mnt/data/localdeps`). VDBFusion runs the LiDAR + single-cam rows on CPU (its own recommended
  `space_carving=False`, `sdf_trunc` = 4 voxels, `preprocess` on); InfiniTAM runs the single-cam rows
  on CUDA (its voxel hashing engine, `mu` = 4 voxels, swapping off, `voxelSize` per row; its block
  size is fixed 8 → compared at B=8). Not built = reported as "not built", never silently dropped.

## Audit — what changed vs the first draft
1. **Spherical LiDAR cannot use `atan2`/`asin`**: PTX gate rejects libdevice (`__nv_*`) and libm ≠
   libdevice bits anyway → breaks bit-exactness. Fix: trig-free projection. Elevation row = binary
   search of `z/r` in a precomputed `sin(elev)` table; azimuth column = one shared polynomial
   `atan2_det(y, x)` written once in plain `f32` mul/add (no FMA), mirrored host/device, tested
   bit-exact. `sqrt` is IEEE on both — spike confirms cuda-oxide lowers it to `sqrt.rn.f32`.
2. **Division → reciprocal multiply everywhere** (`inv_trunc`, `inv_voxel`, `1/meter` as `depth/mod.rs`
   already does), computed once on host, passed to kernels; one documented pixel-rounding rule
   (`floor(u + 0.5)` on `f32`).
3. **Multi-sensor order**: one integrate launch per sensor in fixed sensor order (not a per-voxel
   sensor loop) — same result, simpler kernel, deterministic weighted mean.
4. **Target = beat nvblox AND Open3D on every pre-registered axis** (user, 2026-11-14). Axes, frozen
   in WF-0 before any kernel exists, never redefined mid-loop:
   - integrate end-to-end ms/frame incl. allocation + H2D/zero-copy (single-cam room0/office0; and
     multi-sensor 4 cams + 2 LiDARs);
   - mesh ms in BOTH modes, like-for-like: full extraction AND incremental dirty-block (nvblox meshes
     incrementally);
   - peak memory (host + device);
   - accuracy / completeness vs GT (cm, % < 5 cm): ≤ the best baseline.
   Fairness (skeptic-checked): same frames, voxel, trunc = 4 voxels, max depth, weight cap, colour on;
   each baseline in its OWN documented recommended config, warmed up, synced timing (CUDA events /
   device sync), same clocks. No sandbagged baseline, no per-library param tuning.
   Headline = the multi-sensor fused step (Open3D has no LiDAR path; nvblox integrates sensors one
   by one); single-cam head-to-head reported in full beside it.
4a. **Primary Open3D target = tensor `t.geometry.VoxelBlockGrid` on CUDA** (Open3D's most-used dense
   path; KinectFusion + VoxelHashing engineering). Legacy `ScalableTSDFVolume` and VoxelBlockGrid
   CPU are secondary rows. Run it as its own reconstruction system does (its recommended
   `block_resolution` = 16, attr dtypes tsdf/weight/color float32, `block_count` sized for the scene,
   `compute_unique_block_coordinates` → `integrate` → `extract_triangle_mesh` / `ray_cast`).
   - Consequence: block size is a const generic `B ∈ {8, 16}` (nvblox = 8, Open3D = 16); every
     head-to-head row is run at BOTH so neither baseline is compared at our preferred size only.
   - Drop-in API flavour (kornia's voxel module already mirrors Open3D): Python `TsdfVolume` method
     names/args track VoxelBlockGrid (`integrate(depth, color, intrinsic, extrinsic, depth_scale,
     depth_max)`, `ray_cast`, `extract_triangle_mesh`, `extract_point_cloud`) so a user swaps one
     import; `kornia-py/tests/test_tsdf.py` runs the same script against both and compares meshes
     (chamfer < 0.5 voxel at equal params).
   - Its known costs to attack: per-frame unique of all touched block coords + hashmap `activate`
     (vs our lookup-first 4c), separate attribute buffers (vs our SoA block), per-call Python/tensor
     dispatch (vs one fused frame).
   - Out of scope: FGR / RANSAC+FPFH global registration (submap loop closure takes poses from
     `pgo`, not from global registration).
4b. **FMA, bit-exact**: integrate/raycast math written as explicit scalar `f32::mul_add` (correctly
   rounded on every target) = NEON `vfmaq_f32` = PTX `fma.rn.f32`; those entries go in `FMA_ALLOWED`.
   NEVER `kornia_algebra::simd::F32x4::mul_add` (fuses only on aarch64 → breaks x86 parity); NEON
   kernel uses `core::arch::aarch64` directly, x86 uses scalar `mul_add`.
4c. **Allocation without a per-frame full sort** (biggest structural risk): touched keys → hash
   lookup first; only the misses (new blocks; ~0 after warm-up) are sorted+uniqued and get pool
   indices in ascending key order. Determinism kept, per-frame sort ≈ empty.
5. **Benchmark isolation**: timings only with NO other agent/build running (Open3D-CUDA build finished
   or stopped), MAXN_SUPER, clock state recorded, proof the GPU path ran (kernel launch counter ≠ 0),
   output checked vs CPU before timing (measurement-hygiene rule).
6. **Orin build cap**: 7.4 GB RAM → at most 2 agents compiling at once, each `-j2`, each its own
   worktree AND own `CARGO_TARGET_DIR` under `/mnt/data/kr-tsdf-targets/<agent>` (never share target/).
7. **Submap memory budget**: only active + frustum-visible submaps resident (GPU); others parked on
   host; `max_resident_bytes` config, LRU eviction; bench reports peak.
8. **GPU allocation is a gated branch**, not an assumption: spike decides (a) oxide atomics hash
   insert, (b) reuse `voxel/cuda.rs` radix sort+unique (NVRTC, already there), (c) host allocation.
9. **Unverified facts moved into spikes**: nvblox apt packages on R36.4.3; Open3D aarch64 wheel has
   `RaycastingScene` (else render multiview set with our own scalar raycaster over a GT-mesh TSDF);
   oxide atomics/sqrt/shared-memory support.
10. **System installs (sudo apt / docker pull) run inline by me with user approval between workflows**,
    never inside a workflow agent.
11. Frame-to-model ICP hook: raycast emits vertex+normal maps in `RgbdLevel` layout
    (`registration/rgbd.rs:516`) so existing RGB-D ICP consumes it — constructor only, no ICP changes.
12. Repo rules: read `.claude/skills/kornia-{workflow,coding-style,api-flavor,code-review,bindings}`
    before edits; `changelog.d/<pr>.added.md`; `pixi run pr-check`; thiserror only; comments short.

## Allocation discipline, no_std core, profiling (hard rules)
- **Zero heap allocation per warmed frame** (CPU and CUDA host side): integrate, raycast, incremental
  mesh, submap spawn within budget. All scratch in a reusable `TsdfWorkspace` (touched-key buffer,
  miss list, sort scratch, visible list, MC count/offset buffers) sized at construction or grown
  geometrically on a cold frame only; block pool preallocated (`block_capacity`), growth is an
  explicit `reserve`, never inside a kernel loop. Device buffers allocated once per volume, reused;
  no `cudaMalloc` per frame. Mesh output into caller-owned `TriangleMesh` buffers (`extract_mesh_into`).
- Enforced by tests in their own binaries (pattern `crates/kornia-3d/tests/icp_alloc.rs`, counting
  global allocator, per-thread count): `tests/tsdf_alloc.rs` — warmed frame = 0 allocs for scalar,
  NEON, multi-sensor, multi-agent step, raycast, incremental mesh; CUDA host side 0 allocs.
- **`no_std` core**: all per-voxel / per-pixel math (projection both sensors, `atan2_det`, sdf +
  weighted update, trilinear + gradient, MC tables + edge interpolation, block-key packing, hash
  probe) lives in `tsdf/core/` with `#![no_std]`-clean code: no `alloc`, no `std`, slices + `&mut`
  only, `core::arch::aarch64` for NEON. Checked by `cargo build --target thumbv7em-none-eabihf`-style
  CI-able check or a `#![no_std]` test crate that `include!`s it. WF-0 spike 0.1b: can the SAME
  `core/` source be compiled into the oxide device crate (no_std) so CPU and GPU share one source of
  truth? If packaging allows (`include!` of a file shipped inside both crates, or a tiny
  `kornia-tsdf-core` no_std crate as path dep of `device/`), use it; else mirrored code + bit-exact
  tests (current fallback).
- **Profiling both sides, every round of WF-2**:
  - CPU: `perf stat` (cycles, instructions, IPC, L1/L2 misses, branch misses) + `perf record -g`
    flamegraph per hot path, per backend (scalar, NEON, NEON+rayon); thread scaling 1→6 cores.
  - CUDA: `nsys` timeline per frame (launch gaps, syncs, H2D, CPU-side overhead) + `ncu` per kernel
    (`/usr/local/cuda/bin/ncu`: achieved occupancy, DRAM/L1 throughput, warp stall reasons,
    branch efficiency), via `orin-gpu-profiling` skill.
  - Same protocol on the baselines in WF-0 (0.6) so bottleneck splits are comparable.
  - Alloc count + peak RSS / device memory recorded per round alongside timings.

## Design (unchanged core)
- `crates/kornia-3d/src/tsdf/`: `mod.rs` (config, `TsdfError`), `sensor.rs`
  (`RangeSensor::{Pinhole, Spherical}`, `SphericalLidar`, `range_image_from_points`, `atan2_det`),
  `volume.rs` (SoA 8³ blocks: tsdf f32, weight f32, rgb u8; block hash; pool index assigned in
  ascending key order), `integrate.rs` (scalar), `integrate_neon.rs`, `raycast.rs`, `mesh.rs`
  (per-block marching cubes, sorted block order), `submap.rs` (`SubmapTsdf`, spawn policy,
  `set_submap_poses` fed by `kornia_3d::pgo`, residency budget), `cuda.rs`, `tests.rs`, `cuda_tests.rs`.
- Oxide: `crates/kornia-oxide-kernels/device/src/tsdf.rs` — `tsdf_integrate_pinhole`,
  `tsdf_integrate_spherical`, `tsdf_raycast`, `tsdf_mc_count`, `tsdf_mc_emit` (+ alloc kernels per
  spike). Raw pointers, all loads before first store, FMA only as explicit `mul_add` mirrored on CPU (4b). kornia-3d `cuda` enables
  `kornia-oxide-kernels/{cuda,ptx}`. No NVRTC twin → pre-sm_75 returns `Err`.
- `io/ply`: `write_ply_mesh` (binary). Reuse `PinholeCamera`, `DepthImage` validity/`meter`, `Pose3d`,
  `kornia_algebra::simd::F32x8` (never `mul_add`), `voxel/cuda.rs` sort kernels.
- Integration: allocate blocks in `[d−trunc, d+trunc]` (union over sensors) → sort+unique → per voxel:
  project, `sdf = d − z` (pinhole) / `range − |p|` (LiDAR), skip `< −trunc`, clamp, weighted mean,
  `w = min(w+1, max_w)`.
- Python (`kornia-py/src/tsdf.rs`, registered in `k3d_mod` at `lib.rs:575`, stubs in `k3d.pyi`):
  `PinholeSensor`, `SphericalLidar`, `TsdfVolume(voxel_size, trunc, max_weight, device)` with
  `integrate`, `integrate_multi`, `raycast`, `extract_mesh`, `save_ply`, `num_blocks`, `memory_bytes`;
  `SubmapTsdf` + `submap_ids`, `submap_pose`, `set_submap_poses`. numpy zero-copy, GIL released,
  DLPack device input under `cuda`.
- Example `examples/tsdf_multisensor_fusion` (argh + rerun): 2 rigs (A: 4 cams + 1 LiDAR, B: 2 cams +
  1 LiDAR), `--agents`/`--sensors` toggles, per-submap meshes coloured by agent, intra + inter-rig
  pgo loop-closure snap, `--device cpu|cuda`, `--ply`. Python twin
  `kornia-py/examples/tsdf_fusion.py`.

## Execution: 4 workflows, each step with its own agent roster
Setup inline first (me): clone `kornia/kornia-rs` → `/mnt/data/src/kornia-rs`, branch `feat/tsdf-submaps`.
Every agent returns a schema'd result (`{status, evidence[], files[], numbers{}, blockers[]}`); a claimed
pass must paste command + output tail. Timings never taken inside WF-1/WF-3. Builders that run
concurrently get `isolation: 'worktree'` + own `CARGO_TARGET_DIR`; max 2 compiling at once (`-j2`).
Roster columns = `agent()` opts: `agentType` / `model` / `effort`. Read-only roles use read-only types.
Builders' prompts name the skill they must load (`kornia-cuda-kernel-dev`, `neon-kernels-writer`,
`orin-gpu-profiling`, `multi-view-geometry`, repo `.claude/skills/kornia-*`).

### WF-0 `tsdf-spikes` — Spikes, baselines, contract
| Step | Agent(s) | agentType | model | effort | Writes? |
|---|---|---|---|---|---|
| 0.1b shared no_std `core/` source into oxide device crate: feasible? | `core-share-spike` ×1 | `oh-my-claudecode:executor-high` | opus | high | scratch only |
| 0.1 oxide spike: toolchain, `build_ptx`, throwaway atomics/sqrt/shared kernels through PTX gate → `alloc_path` | `oxide-spike` ×1 | `oh-my-claudecode:executor-high` | opus | high | scratch only |
| 0.2 baseline probe: Open3D wheel + `VoxelBlockGrid`/`RaycastingScene`, nvblox pkg/docker availability (no install), Open3D-CUDA RAM estimate, cudarc CUDA-graph API | `probe-o3d`, `probe-nvblox`, `probe-cudarc` ×3 | `oh-my-claudecode:researcher` | sonnet | medium | no |
| 0.3 data: Replica room0/office0 download + load check | `data` ×1 | `oh-my-claudecode:executor` | sonnet | low | /mnt/data |
| 0.4 contract: op order, rounding, `atan2_det`, mul_add sites, layouts, API (VoxelBlockGrid-flavoured) | `contract-a` (numerics-first), `contract-b` (API-first) ×2 | `oh-my-claudecode:architect` | opus | xhigh | draft text |
| 0.5 contract judge: merge, list non-bit-exact-safe ops | `contract-judge` ×1 | `oh-my-claudecode:critic` | opus | max | `tsdf/CONTRACT.md` |
| gate | me: alloc path, install route (sudo asked), freeze contract + params; installs inline | — | — | — | — |
| 0.6 measure baselines (nvblox, VBG CUDA/CPU, legacy, VDBFusion CPU, InfiniTAM CUDA; full + incremental mesh; nsys) → `baseline_targets.json` | `measure-nvblox`, `measure-o3d`, `measure-vdbfusion`, `measure-infinitam` — **serial** | `oh-my-claudecode:scientist-high` | opus | high | json + logs |
| 0.7 refute baseline numbers (config = recommended? synced? warmed?) | `baseline-skeptic` ×1 | `oh-my-claudecode:critic` | opus | high | no |

### WF-1 `tsdf-core` — Scalar, NEON, CUDA (bit-exact)
| Step | Agent(s) | agentType | model | effort | Writes? |
|---|---|---|---|---|---|
| 1.1 scalar reference + ply writer + submaps + tests | `scalar` ×1 (alone) | `oh-my-claudecode:executor-high` | opus | xhigh | kornia-3d |
| 1.1b alloc tests (`tsdf_alloc.rs`) + no_std check of `tsdf/core/` | `alloc-guard` ×1 | `oh-my-claudecode:tdd-guide` | sonnet | high | tests |
| 1.2 test-gap check on scalar before backends fork | `tdd` ×1 | `oh-my-claudecode:tdd-guide` | sonnet | high | tests only |
| 1.3 NEON integrate (+raycast) | `neon` ×1, worktree | `oh-my-claudecode:executor-high` | opus | xhigh | kornia-3d |
| 1.3 CUDA oxide kernels + launchers + `_bit_exact` tests | `cuda` ×1, worktree | `oh-my-claudecode:executor-high` | opus | xhigh | oxide + kornia-3d |
| 1.4 review per backend, lens 1: contract / bit-exactness (REFUTE) | `rev-exact-{neon,cuda}` | `oh-my-claudecode:architect` | opus | high | no |
| 1.4 review per backend, lens 2: unsafe / bounds / overflow / silent errors | `rev-safety-{neon,cuda}` | `pr-review-toolkit:silent-failure-hunter` | sonnet | high | no |
| 1.5 fix confirmed findings | `fix-{neon,cuda}` | `oh-my-claudecode:executor` | sonnet | high | same worktree |
| 1.6 test matrix + 1 mutation per backend (serial) | `gate-{neon,cuda}` | `oh-my-claudecode:qa-tester-high` | opus | medium | no |
| gate | me: merge worktrees; 3 backends bit-identical on Replica frames or stop | — | — | — | — |

### WF-2 `tsdf-bench-loop` — Beat nvblox + VoxelBlockGrid (strictly serial)
| Step | Agent(s) | agentType | model | effort | Writes? |
|---|---|---|---|---|---|
| 2.1 harness: `bench_tsdf.rs`, `gen_multiview_replica.py`, `bench_tsdf_baselines.py`, `bench_nvblox.cpp` | `harness` ×1 | `oh-my-claudecode:executor` | sonnet | high | benches |
| 2.2 measure all axes vs `baseline_targets.json` | `measure` (per round) | `oh-my-claudecode:scientist` | sonnet | medium | logs |
| 2.3a CPU profile (perf stat/record, flamegraph, 1→6 thread scaling, alloc count) | `profile-cpu` (per round) | `oh-my-claudecode:scientist-high` | opus | xhigh | no |
| 2.3b CUDA profile (nsys timeline + ncu per kernel, stall reasons) | `profile-cuda` (per round, after 2.3a — serial) | `oh-my-claudecode:scientist-high` | opus | xhigh | no |
| 2.4 optimize: one rung, bit-exact tests green | `optimize` (per round) | `oh-my-claudecode:executor-high` | opus | max | kornia-3d/oxide |
| 2.5 A/B vs previous commit, revert regressions | `ab` (per round) | `oh-my-claudecode:scientist` | sonnet | medium | revert only |
| loop | until every axis wins or ladder exhausted → me → user with profiles | — | — | — | — |
| 2.6 refute final table (params, GPU path hit, sync, alloc included, accuracy trade, baseline config) | `skeptic-a`, `skeptic-b` ×2 (different lenses) | `oh-my-claudecode:critic` | opus | max | no |
| 2.7 write `docs/benchmark-tsdf.md` | `bench-doc` ×1 | `oh-my-claudecode:writer` | sonnet | medium | docs |

Ladder (structural levers, roughly in expected-payoff order):
1. Orin unified memory: depth/RGB zero-copy (host-mapped / managed), no H2D memcpy.
2. One allocation pass shared by all N sensors of a step; integrate launches back-to-back, no host
   sync between sensors.
3. CUDA-graph capture of a whole frame if cudarc exposes graphs; else one sync per frame.
4. Visible-block culling: frustum test per block + skip blocks fully behind `d + trunc`.
5. Integrate kernel shape: warp = one x-row pair, depth reads via L1 (128 KB on sm_87), all loads
   before stores, FMA path (4b).
6. Dirty-block incremental meshing + cached per-block meshes; full extraction = scan + emit.
7. Hash: open addressing sized 2× live blocks, packed `u64` key, one probe in the common case.
8. Half-width storage (tsdf `f16`/`i16`, weight `u16`), same quantization on CPU; kept only if
   accuracy axis still wins.
9. CPU: NEON + rayon over blocks, block-major SoA, prefetch; vs VBG CPU and legacy.

### WF-3 `tsdf-surface` — Python, example, docs, review
| Step | Agent(s) | agentType | model | effort | Writes? |
|---|---|---|---|---|---|
| 3.1 Python bindings + stubs + `test_tsdf.py` + `bench_tsdf.py` (VBG-parity script) | `python` ×1, worktree | `oh-my-claudecode:executor-high` | opus | high | kornia-py |
| 3.1 Rust example `tsdf_multisensor_fusion` + Python twin + README | `example` ×1, worktree | `oh-my-claudecode:executor` | sonnet | high | examples |
| 3.1 module docs + changelog fragment | `docs` ×1 | `oh-my-claudecode:writer` | haiku | low | docs |
| 3.2 review: API flavour (`kornia-api-flavor`, VBG parity) | `rev-api` | `pr-review-toolkit:type-design-analyzer` | opus | high | no |
| 3.2 review: bindings rules (`kornia-bindings`), GIL, zero-copy | `rev-bindings` | `oh-my-claudecode:code-reviewer` | sonnet | high | no |
| 3.2 review: whole diff (`kornia-code-review`) | `rev-diff` | `pr-review-toolkit:code-reviewer` | opus | xhigh | no |
| 3.2 review: test coverage | `rev-tests` | `pr-review-toolkit:pr-test-analyzer` | sonnet | medium | no |
| 3.3 fix confirmed findings | `fix` ×1 | `oh-my-claudecode:executor` | sonnet | high | repo |
| 3.4 completeness: what in this plan is missing/unverified → one last fix round | `completeness` ×1 | `oh-my-claudecode:critic` | opus | max | no |
| 3.5 simplify pass (own commit, tests unmodified) | `simplify` ×1 | `code-simplifier:code-simplifier` | sonnet | medium | repo |
| me | `pixi run pr-check`, push `feat/tsdf-submaps`, open THE single draft PR (body: reading order + bench table) | — | — | — | — |

## Verification
- `cargo test -p kornia-3d --release` and `--features cuda`; `scripts/cuda-oxide/orin-test-bench.sh`
  (every `tsdf_*` entry has a passing `_bit_exact`); PTX gate clean; clippy `-D warnings`.
- `cargo bench -p kornia-3d --bench bench_tsdf --features cuda` + `bench_tsdf_baselines.py`, quiet box.
- `cd kornia-py && maturin develop --release [--features cuda] && pytest tests/test_tsdf.py`.
- Run `tsdf_multisensor_fusion` on replica_multiview, inspect rerun (sensor toggles change
  completeness; loop-closure snap), write PLY, score it with the accuracy script.

## Risks
- cuda-oxide gaps → alloc falls back to radix/host (still a valid result, disclosed in bench doc).
- nvblox prebuilt not installable on this JetPack → docker or source build (hours at `-j2`).
- Open3D-CUDA build may not fit 7.4 GB → reported as "not built", Open3D CPU + nvblox remain.
- Goal is to beat both on every axis; the loop does not stop at a plateau. If an axis still loses
  after the full ladder, the user sees it with profile evidence — never hidden or redefined.
- Large single PR: mitigated by per-layer commits and a PR body with a reading order + bench table;
  NOT split (user rule).
