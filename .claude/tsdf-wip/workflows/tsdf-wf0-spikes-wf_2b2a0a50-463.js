export const meta = {
  name: 'tsdf-wf0-spikes',
  description: 'TSDF WF-0a: oxide/no_std spikes, baseline probes, Replica data, bit-exact contract',
  phases: [
    { title: 'Spikes', detail: 'oxide toolchain + atomics/sqrt/shared, then shared no_std core' },
    { title: 'Probes', detail: 'Open3D wheel, nvblox availability, cudarc graphs' },
    { title: 'Data', detail: 'Replica room0/office0' },
    { title: 'Contract', detail: '2 independent drafts + judge' },
  ],
}

const RESULT = {
  type: 'object',
  properties: {
    status: { type: 'string', enum: ['ok', 'partial', 'blocked'] },
    summary: { type: 'string' },
    evidence: { type: 'array', items: { type: 'string' }, description: 'command + output tail per claim' },
    files: { type: 'array', items: { type: 'string' } },
    numbers: { type: 'object', additionalProperties: true },
    decisions: { type: 'object', additionalProperties: true },
    blockers: { type: 'array', items: { type: 'string' } },
  },
  required: ['status', 'summary', 'evidence', 'blockers'],
}

const CTX = `
Context: we are building a deterministic block-hashed TSDF with submaps in kornia-rs crate kornia-3d
(new module crates/kornia-3d/src/tsdf/), backends scalar / NEON / CUDA via cuda-oxide, all bit-exact.
Plan: /home/nvidia/.claude/plans/yes-prototype-it-in-vivid-robin.md (read it first).
Repo checkout: /mnt/data/src/kornia-rs (branch feat/tsdf-submaps). cuda-oxide docs: docs/cuda-oxide.md there.
Box: Jetson Orin Nano, 7.4 GB RAM, 6 cores, CUDA 12.6, JetPack R36.4.3, Ubuntu 22.04, python3.10.
HARD RULES: no sudo, no apt, no docker pull, no git push, no commits on feat/tsdf-submaps. cargo builds
use -j2 and CARGO_TARGET_DIR=/mnt/data/kr-tsdf-targets/<your-label>. Large downloads go under /mnt/data.
Every claim of success must include the exact command and the tail of its output in evidence[].
Nothing flux-specific anywhere.`

phase('Spikes')
const spikes = (async () => {
  const oxide = await agent(`${CTX}
Task 0.1 (oxide spike). Load skill kornia-cuda-kernel-dev first.
1. Install/verify the cuda-oxide toolchain with scripts/cuda-oxide/install-toolchain.sh (default PREFIX ~/.cargo/cuda-oxide; it uses rustup, no sudo). Then run cargo run -p kornia-oxide-kernels --example build_ptx and confirm green.
2. In a SCRATCH copy of crates/kornia-oxide-kernels (copy to /mnt/data/kr-tsdf-spikes/oxide, never edit the repo), add throwaway #[kernel]s that use: atomic CAS + atomic add on u32/u64 global memory (find the cuda-device API for atomics), f32 sqrt, f32 mul_add (does it lower to fma.rn.f32?), floor, shared memory (static/dynamic) with block sync, a u64 hash-probe loop. Build PTX and run the PTX gate on it. Report per feature: compiles? lowers to which PTX instr? passes gate (fma needs FMA_ALLOWED)?
3. If a GPU test is cheap, load the PTX with cudarc and run the atomics kernel once to check results.
Return decisions.alloc_path = 'oxide' (atomics usable) | 'radix' (reuse NVRTC radix sort in crates/kornia-3d/src/voxel/cuda.rs) | 'host', with reasoning, and decisions.features = map feature -> {compiles, ptx, gate}.`,
    { label: 'oxide-spike', phase: 'Spikes', schema: RESULT, agentType: 'oh-my-claudecode:executor-high', model: 'opus', effort: 'high' })
  const core = await agent(`${CTX}
Task 0.1b (shared no_std core spike). The oxide toolchain should now be installed (previous spike result: ${JSON.stringify(oxide && oxide.decisions || {})}).
Question: can ONE source file of pure per-voxel f32 math (e.g. a fn project_pinhole(p:[f32;3], fx,fy,cx,cy)->(f32,f32) using f32::mul_add, and a tsdf_update fn) be compiled both into kornia-3d (std crate) and into the oxide device crate (crates/kornia-oxide-kernels/device, no_std, own workspace, pinned nightly) so CPU and GPU share one source of truth?
Try in scratch copies under /mnt/data/kr-tsdf-spikes/core (never edit the repo): (a) include!() of a file that lives inside kornia-oxide-kernels (shipped by its Cargo include list) from both sides; (b) a tiny no_std path crate 'kornia-tsdf-core' used as dependency of device/ AND kornia-3d — check cargo package of kornia-oxide-kernels would still work (path deps outside crate dir are not packaged); (c) any better option. Check that f32::mul_add / floor / sqrt are usable in no_std (core intrinsics vs libm!) on both sides — in no_std, f32::mul_add may need core::intrinsics or libm; find what works on stable for kornia-3d and on the pinned nightly for the device crate.
Build PTX for the shared fn, pass the gate, and for CPU compile + run a small check that the same inputs give the same bits on CPU and (if cheap) GPU.
Return decisions.core_share = 'include' | 'crate' | 'mirror', with the exact mechanism and packaging caveats.`,
    { label: 'core-share-spike', phase: 'Spikes', schema: RESULT, agentType: 'oh-my-claudecode:executor-high', model: 'opus', effort: 'high' })
  return { oxide, core }
})()

phase('Probes')
const probes = parallel([
  () => agent(`${CTX}
Task 0.2a (Open3D probe). Create a venv /mnt/data/venvs/tsdf-bench (python3.10 -m venv) and pip install open3d (the aarch64 wheel) plus numpy scipy. Check: version; o3d.core.cuda.is_available(); t.geometry.VoxelBlockGrid usable on CPU (integrate a synthetic depth frame, extract_triangle_mesh, ray_cast); t.geometry.RaycastingScene available (cast rays against a mesh); legacy ScalableTSDFVolume. Find Open3D's recommended VoxelBlockGrid config in its reconstruction system / examples (block_resolution, voxel_size, block_count, attr dtypes, trunc multiplier, depth_max) — cite the source file/URL. Then research an Open3D CUDA source build for Jetson Orin (JetPack 6, CUDA 12.6, sm_87): required deps (apt packages needed -> list them, do NOT install), cmake flags, expected build time and peak RAM at -j2, known Jetson issues. Return decisions.o3d_recommended_config and decisions.o3d_cuda_build_recipe.`,
    { label: 'probe-o3d', phase: 'Probes', schema: RESULT, agentType: 'oh-my-claudecode:executor', model: 'sonnet', effort: 'medium' }),
  () => agent(`${CTX}
Task 0.2b (nvblox probe, NO install). Determine the cheapest route to a working nvblox (core library, CUDA) on this box: (1) Isaac ROS apt repo for JetPack 6.1 / R36.4.3 (isaac.download.nvidia.com) — exact repo line, package names that carry libnvblox_core + headers (ros-humble-isaac-ros-nvblox, ros-humble-nvblox* ...), what ROS humble deps they pull, whether Orin Nano is supported; (2) Isaac ROS dev docker image for Jetson — tag + size; (3) nvblox standalone source build (github nvidia-isaac/nvblox): its executables for Replica (fuse_replica / fuse_3dmatch) and their timing output, cmake deps, build time at -j2 on 8GB. You may use read-only commands (apt-cache policy, curl -sI to check URLs exist, docker images, docker manifest inspect is OK). Also extract nvblox's recommended mapper config for Replica (voxel size, projective integrator truncation, max weight, mesh update mode, LiDAR integrator params) with source citations, and how its timers are printed. Return decisions.nvblox_route (ranked) with exact commands I would run (with sudo where needed), and decisions.nvblox_recommended_config.`,
    { label: 'probe-nvblox', phase: 'Probes', schema: RESULT, agentType: 'oh-my-claudecode:executor', model: 'sonnet', effort: 'high' }),
  () => agent(`${CTX}
Task 0.2c (cudarc probe, read-only). Which cudarc version does the kornia-rs workspace pin (Cargo.toml / Cargo.lock in /mnt/data/src/kornia-rs)? In that version's source (~/.cargo/registry/src or docs.rs), does it expose: CUDA graphs (stream capture begin/end, graph instantiate/launch), host-mapped / pinned memory (cuMemHostAlloc with DEVICEMAP, cuMemHostGetDevicePointer) or managed memory (cuMemAllocManaged) for zero-copy on Jetson unified memory, CUDA events for timing, async memset? Also how kornia-tensor wraps device memory (crates/kornia-tensor) and whether a host-mapped buffer can be wrapped. Give exact API names + file:line citations. Return decisions.cudarc = {version, graphs, mapped_memory, managed_memory, events} each with the API path.`,
    { label: 'probe-cudarc', phase: 'Probes', schema: RESULT, agentType: 'oh-my-claudecode:researcher', model: 'sonnet', effort: 'medium' }),
])

phase('Data')
const data = agent(`${CTX}
Task 0.3 (data). Download the NICE-SLAM preprocessed Replica dataset (the standard Replica.zip used by NICE-SLAM / nvblox / many SLAM papers — find the current official URL, e.g. from the NICE-SLAM repo scripts/download_replica.sh) into /mnt/data/datasets/replica, plus the GT meshes (cull_replica_mesh or the *_mesh.ply files the NICE-SLAM eval uses). Keep at least room0 and office0; you may delete other scenes after unzip to save space only if the zip is kept. Verify: frame count, depth png format + scale (expect 6553.5), image size, intrinsics (cam_params.json), traj.txt pose count == frame count, GT mesh loads (python with numpy; if /mnt/data/venvs/tsdf-bench exists use it). Return numbers {frames_room0, frames_office0, width, height, fx, fy, cx, cy, depth_scale} and the exact paths.`,
  { label: 'data', phase: 'Data', schema: RESULT, agentType: 'oh-my-claudecode:executor', model: 'sonnet', effort: 'low' })

phase('Contract')
const CONTRACT_TASK = `Write a DRAFT bit-exact contract for the TSDF (markdown, returned in summary field fully, not abbreviated). Read: the plan; crates/kornia-3d/src/depth/mod.rs (DepthImage meter/validity conventions, f32 reciprocal rule); crates/kornia-3d/src/voxel/mod.rs (key/determinism conventions); docs/cuda-oxide.md (PTX gate, FMA_ALLOWED, raw pointers, loads-before-stores); crates/kornia-algebra/src/simd.rs (why F32x4::mul_add is forbidden); crates/kornia-3d/src/registration/rgbd.rs RgbdLevel. Load skill multi-view-geometry.
Must specify exactly: block key packing (i32 per axis -> u64), voxel centre formula, world->sensor transform op order with every mul_add written out, pinhole projection + the single pixel rounding rule, spherical LiDAR projection WITHOUT atan2/asin (elevation via sin-table binary search; azimuth via a deterministic polynomial atan2_det written as an explicit sequence of f32 mul_add — give the coefficients and max error), sdf for both sensors, truncation/clamp, weighted mean update with weight cap (exact expression order), colour update and rounding to u8, allocation algorithm (lookup-first, sort only misses, pool indices in ascending key order), visible-block set, marching cubes edge interpolation + table ownership + output order, raycast march + trilinear + gradient normals, multi-sensor order (sensor launches in fixed order), multi-agent ordering (stamp, agent, sensor), what is allowed to differ (hash slot positions) vs must be identical, block size const generic B in {8,16}, storage layout SoA per block, NaN/inf/missing-depth rules, and the public Rust API signatures (TsdfVolume, TsdfWorkspace, SubmapTsdf, SensorRig, RangeSensor, AgentId/SensorId, TriangleMesh, extract_mesh_into, integrate/integrate_batch, raycast into RgbdLevel-compatible maps) following kornia API flavour (.claude/skills/kornia-api-flavor) and zero-allocation-per-warmed-frame. Mark every op that is a bit-exactness risk.`
const contract = (async () => {
  const drafts = await parallel([
    () => agent(`${CTX}\nTask 0.4a — NUMERICS-FIRST angle: optimise for provable bit-exactness across scalar/NEON/PTX and for speed of the inner loops.\n${CONTRACT_TASK}`,
      { label: 'contract-a', phase: 'Contract', schema: RESULT, agentType: 'oh-my-claudecode:architect', model: 'opus', effort: 'xhigh' }),
    () => agent(`${CTX}\nTask 0.4b — API-FIRST angle: optimise for a clean kornia + Open3D-VoxelBlockGrid-flavoured API, multi-agent/multi-sensor ergonomics, zero allocation, no_std core boundary.\n${CONTRACT_TASK}`,
      { label: 'contract-b', phase: 'Contract', schema: RESULT, agentType: 'oh-my-claudecode:architect', model: 'opus', effort: 'xhigh' }),
  ])
  const ok = drafts.filter(Boolean)
  const judge = await agent(`${CTX}
Task 0.5 — contract judge. Two independent drafts of the TSDF bit-exact contract follow. Merge into ONE final contract (full markdown in summary, nothing abbreviated), taking the numerics of the better draft and the API of the better draft, grafting good ideas from both. Then adversarially list in blockers[] every op in the merged contract that is NOT safe for bit-exactness across scalar Rust (x86 + aarch64), NEON intrinsics and PTX from cuda-oxide, or that would force a heap allocation per frame, or that leaks any flux concept. Verify claims against the repo files where possible.
DRAFT A:\n${ok[0] ? ok[0].summary : '(missing)'}\n\nDRAFT B:\n${ok[1] ? ok[1].summary : '(missing)'}`,
    { label: 'contract-judge', phase: 'Contract', schema: RESULT, agentType: 'oh-my-claudecode:critic', model: 'opus', effort: 'max' })
  return { drafts: ok.map(d => ({ status: d.status, blockers: d.blockers })), judge }
})()

const [s, p, d, c] = await Promise.all([spikes, probes, data, contract])
return { spikes: s, probes: p, data: d, contract: c }
