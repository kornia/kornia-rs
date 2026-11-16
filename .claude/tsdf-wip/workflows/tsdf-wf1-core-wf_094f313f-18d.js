export const meta = {
  name: 'tsdf-wf1-core',
  description: 'TSDF WF-1: scalar reference, alloc/no_std guards, NEON + cuda-oxide backends, refute reviews, mutation gates',
  phases: [
    { title: 'Scalar', detail: 'reference implementation + tests' },
    { title: 'Guards', detail: 'alloc-count tests, no_std check, test gaps' },
    { title: 'NEON', detail: 'build, review x2, fix, gate' },
    { title: 'CUDA', detail: 'build, review x2, fix, gate' },
  ],
}

const RESULT = {
  type: 'object',
  properties: {
    status: { type: 'string', enum: ['ok', 'partial', 'blocked'] },
    summary: { type: 'string' },
    evidence: { type: 'array', items: { type: 'string' } },
    files: { type: 'array', items: { type: 'string' } },
    commits: { type: 'array', items: { type: 'string' } },
    numbers: { type: 'object', additionalProperties: true },
    blockers: { type: 'array', items: { type: 'string' } },
  },
  required: ['status', 'summary', 'evidence', 'blockers'],
}
const FINDINGS = {
  type: 'object',
  properties: {
    findings: { type: 'array', items: { type: 'object', properties: {
      file: { type: 'string' }, line: { type: 'integer' }, severity: { type: 'string', enum: ['critical', 'major', 'minor'] },
      claim: { type: 'string' }, repro: { type: 'string' }, confirmed: { type: 'boolean' } },
      required: ['file', 'claim', 'severity', 'confirmed'] } },
    summary: { type: 'string' },
  },
  required: ['findings', 'summary'],
}

const CTX = `
Repo: /mnt/data/src/kornia-rs, branch feat/tsdf-submaps (work directly there; you are the only agent writing right now).
READ FIRST: /home/nvidia/.claude/plans/yes-prototype-it-in-vivid-robin.md, /mnt/data/kr-tsdf-spikes/frozen/DECISIONS.md (binding),
/mnt/data/kr-tsdf-spikes/frozen/CONTRACT.v1.md (normative bit-exact contract + public API), repo AGENTS.md and the repo skills it names
(.claude/skills/kornia-coding-style, kornia-api-flavor, kornia-workflow, kornia-simd for SIMD). Spike code to reuse: /mnt/data/kr-tsdf-spikes/core (shared no_std tsdf_core via F32Ops), /mnt/data/kr-tsdf-spikes/oxide (atomics, ptx_asm fma/sqrt, gate edits).
HARD RULES: an Open3D CUDA build is running on this 7.4 GB box -> cargo with -j1 ONLY (CARGO_BUILD_JOBS=1), CARGO_TARGET_DIR=/mnt/data/kr-tsdf-targets/<your-label>; release mode for tests that touch real data. No sudo, no push, no new PRs.
thiserror only (never anyhow). Comments short (invariant + why). Nothing flux-specific anywhere (no flux names/types/formats). Zero heap allocation per warmed frame. Per-voxel/per-pixel math only in the shared no_std tsdf_core (generic over F32Ops), never duplicated.
Commit your layer when green with a conventional message ending in a blank line then: Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Before committing run: cargo fmt; cargo clippy -p <crates you touched> --all-targets -- -D warnings; the tests you added. Evidence = command + output tail.`

phase('Scalar')
const scalar = await agent(`${CTX}
Step 1.1 — SCALAR REFERENCE (load skill multi-view-geometry). Implement exactly the contract:
- crates/kornia-oxide-kernels/device/src/tsdf_core.rs (shared no_std math: F32Ops trait, transforms, pinhole + spherical LiDAR projection with atan2_det and sin-table elevation search, sdf/trunc/weighted update/colour, block key pack/unpack, voxel centre, hash probe helpers, marching cubes tables + edge interpolation, trilinear + gradient), exposed on host via kornia-oxide-kernels lib.rs #[path] module per DECISIONS (no features needed). device/src/main.rs gets mod tsdf_core; (no kernels yet). Keep build_ptx green (run it).
- crates/kornia-3d/src/tsdf/: mod.rs (config, errors, docs), sensor.rs, volume.rs (SoA blocks, const generic B in {8,16}, hash, pool), alloc (lookup-first, sort misses, rank order), integrate (single sensor, multi-sensor step, multi-agent batch sorted by (stamp, agent, sensor)), raycast (RgbdLevel-compatible vertex/normal maps), mesh (full extraction + incremental dirty-block, caller-owned TriangleMesh via extract_mesh_into), submap.rs (SubmapTsdf, SubmapId{agent,seq}, spawn policy, set_submap_poses, residency budget, SubmapFilter), workspace (TsdfWorkspace, zero alloc warmed). StdOps impl of F32Ops. Wire pub mod tsdf in lib.rs. kornia-3d depends on kornia-oxide-kernels (workspace, no features).
- io/ply: binary write_ply_mesh (+ round-trip test).
- tests (crates/kornia-3d/src/tsdf/tests.rs): analytic plane + sphere depth renders (pinhole and spherical LiDAR) -> zero crossing within voxel/4; bitwise determinism over 2 runs; weight cap; missing/invalid depth skipped; NaN never stored; MC sphere closed + area within 2%; incremental mesh == full mesh after same frames; raycast depth vs rendered depth; multi-sensor step == sequential single-sensor in fixed order; multi-agent shuffled arrival == sorted serial (bitwise); submap pose update moves only that agent's submaps; B=8 and B=16 both; LiDAR range_image_from_points deterministic; atan2_det max error bound from contract; StdOps fma matches libm fmaf.
Commit as: feat(3d): tsdf scalar reference with submaps, multi-sensor, multi-agent
Return files, commits, numbers {tests_passed, test_count}, and any contract deviation in blockers (deviations must be minimal and justified).`,
  { label: 'scalar', phase: 'Scalar', schema: RESULT, agentType: 'oh-my-claudecode:executor-high', model: 'opus', effort: 'xhigh' })
if (!scalar || scalar.status === 'blocked') return { scalar }

phase('Guards')
const guard = await agent(`${CTX}
Step 1.1b — ALLOCATION + no_std GUARDS on the scalar TSDF just committed (${(scalar.commits || []).join(', ')}).
1. crates/kornia-3d/tests/tsdf_alloc.rs (own test binary, counting global allocator, per-thread counter; copy the pattern of crates/kornia-3d/tests/icp_alloc.rs): after warm-up frames, ZERO allocations for: single-sensor integrate (B=8,16), multi-sensor step (2 cams + 1 lidar), multi-agent batch (2 agents), raycast, incremental mesh into a reused TriangleMesh, submap spawn within reserved budget. Fix the implementation (not the test) if a path allocates.
2. no_std check: add a test/CI-able check that kornia-oxide-kernels (no features) builds no_std including tsdf_core (e.g. cargo build -p kornia-oxide-kernels --target aarch64-unknown-none if installable via rustup target add, else a no_std check crate under tests that #![no_std] uses it). Document the command.
3. clippy disallowed-methods guard for tsdf code: forbid f32::min/max/clamp, F32x4/F32x8::mul_add, inherent mul_add/sqrt/floor inside tsdf_core (per contract BX-5/BX-6) — via a clippy.toml scoped as the repo allows, or #[deny] attributes + a grep test; pick what works and show it fires on a planted violation (then remove the plant).
Commit: test(3d): tsdf zero-alloc and no_std guards`,
  { label: 'alloc-guard', phase: 'Guards', schema: RESULT, agentType: 'oh-my-claudecode:tdd-guide', model: 'sonnet', effort: 'high' })
const tdd = await agent(`${CTX}
Step 1.2 — TEST-GAP CHECK before backends fork. Read the contract section by section and the scalar tests. For every normative rule (rounding, range gates, NaN, weight cap, colour rounding, alloc ranking, MC order, raycast stop rule, multi-agent order, submap residency/eviction, error cases) make sure a test pins it with an input where a plausible wrong implementation gives different bits. Add the missing tests (they must pass on the current scalar code; if one fails, the CODE is wrong vs contract: fix it and say so). Also add golden-bit tests for LiDAR tables and atan2_det. Commit: test(3d): pin tsdf contract rules`,
  { label: 'tdd', phase: 'Guards', schema: RESULT, agentType: 'oh-my-claudecode:tdd-guide', model: 'sonnet', effort: 'high' })

async function backend(name, buildPrompt, ph) {
  const built = await agent(buildPrompt, { label: name, phase: ph, schema: RESULT, agentType: 'oh-my-claudecode:executor-high', model: 'opus', effort: 'xhigh' })
  if (!built || built.status === 'blocked') return { built }
  const revs = await parallel([
    () => agent(`${CTX}
Step 1.4 lens 1 — try to REFUTE that the ${name} backend is bit-exact with the scalar reference under the contract. Read the diff of commits ${(built.commits || []).join(', ')} (git show). Hunt: op order vs tsdf_core, FMA operand slots, contraction, rounding/conversion without range gates, NaN reaching storage, min/max semantics, signed zero, lane tail handling, B=16 path, multi-sensor ordering, alloc rank order. For each suspicion write a concrete failing input and run it if cheap (cargo -j1). Mark confirmed=true only if you reproduced it. Do NOT edit code.`,
      { label: `rev-exact-${name}`, phase: ph, schema: FINDINGS, agentType: 'oh-my-claudecode:architect', model: 'opus', effort: 'high' }),
    () => agent(`${CTX}
Step 1.4 lens 2 — unsafe / bounds / overflow / silent-failure review of the ${name} backend (commits ${(built.commits || []).join(', ')}). Every unsafe block: is its SAFETY comment true? Index math overflow (u32 pixel/voxel counts, i32 keys), out-of-bounds pointer offsets, tail lanes, error paths that swallow failures or return partial results, CUDA errors ignored, sync missing before host reads. Concrete repro for each; confirmed=true only if reproduced or provable from code. Do NOT edit code.`,
      { label: `rev-safety-${name}`, phase: ph, schema: FINDINGS, agentType: 'pr-review-toolkit:silent-failure-hunter', model: 'sonnet', effort: 'high' }),
  ])
  const confirmed = revs.filter(Boolean).flatMap(r => r.findings).filter(f => f.confirmed)
  let fix = null
  if (confirmed.length) {
    fix = await agent(`${CTX}
Step 1.5 — fix these CONFIRMED findings in the ${name} backend, each with a regression test that fails before the fix:\n${JSON.stringify(confirmed, null, 1)}\nCommit: fix(3d): tsdf ${name} review findings`,
      { label: `fix-${name}`, phase: ph, schema: RESULT, agentType: 'oh-my-claudecode:executor', model: 'sonnet', effort: 'high' })
  }
  const gate = await agent(`${CTX}
Step 1.6 — GATE for the ${name} backend. Run sequentially (-j1): full kornia-3d tsdf tests (release), tsdf_alloc test binary, kornia-oxide-kernels tests, clippy -D warnings on touched crates${name === 'cuda' ? ', cargo test -p kornia-3d --features cuda --release tsdf, scripts/cuda-oxide/orin-test-bench.sh style check that every tsdf_* PTX entry has a passing *_bit_exact test, build_ptx + gate' : ''}.
Then ONE mutation check: flip a single rounding/weight/operand-order line in the ${name} path (not in tests), rebuild, confirm a bit-exact test FAILS, restore the file with git checkout, then touch it and rebuild so the green run is not stale (mutation-restore rule). Report pass/fail per command with output tails. Do NOT commit.`,
    { label: `gate-${name}`, phase: ph, schema: RESULT, agentType: 'oh-my-claudecode:qa-tester-high', model: 'opus', effort: 'medium' })
  return { built, reviews: revs, confirmed: confirmed.length, fix, gate }
}

phase('NEON')
const neon = await backend('neon', `${CTX}

CONTRACT v1.1 (gate decision, commit 29190f43): section 13 raycast now BACKS OFF after a block skip that lands behind the surface (see raycast_pixel in device/src/tsdf_core.rs: fine_until/skipped state, z = (z - block_step) + voxel_size, plain sub/add never fused); default max_steps = 2*ceil(span/voxel)+2. Mirror THAT march exactly. The 404 tilted-plane misses are frustum-edge rays (pinned), not a bug. Replica lives at /mnt/data/datasets/replica/Replica/<scene>/results (depthNNNNNN.png, frameNNNNNN.jpg, traj.txt one level up; cam_params.json in Replica/).
Step 1.3 — NEON BACKEND (load skill neon-kernels-writer; profile-before-tweak applies later, now correctness + sane structure).
crates/kornia-3d/src/tsdf/integrate_neon.rs (and raycast_neon.rs if raycast is hot): #[cfg(target_arch = "aarch64")] core::arch::aarch64 intrinsics, a NeonOps-style lane implementation of the SAME tsdf_core math (vfmaq_f32 with correct operand slots, vsqrtq_f32; floor is the integer floor_i32 of the core, never a float floor such as vrndmq_f32 -- contract section 1), voxel rows of the block as lanes, scalar depth gather per lane, ordered range gates before every conversion, tails handled. Rayon over blocks with deterministic results (each voxel written by exactly one task). Runtime/compile-time dispatch: aarch64 -> NEON, else scalar. Tests *_neon_matches_scalar bitwise on: analytic scenes, 20 Replica room0 frames from /mnt/data/datasets/replica/Replica/room0 (gate behind an env var / ignored test if the path is missing), fused!=unfused stress inputs, B=8/16, pinhole + LiDAR, multi-sensor + multi-agent.
Commit: perf(3d): tsdf NEON integrate (bit-exact with scalar)`, 'NEON')

phase('CUDA')
const cuda = await backend('cuda', `${CTX}

CONTRACT v1.1 (gate decision, commit 29190f43): section 13 raycast now BACKS OFF after a block skip that lands behind the surface (see raycast_pixel in device/src/tsdf_core.rs: fine_until/skipped state, z = (z - block_step) + voxel_size, plain sub/add never fused); default max_steps = 2*ceil(span/voxel)+2. Mirror THAT march exactly. The 404 tilted-plane misses are frustum-edge rays (pinned), not a bug. Replica lives at /mnt/data/datasets/replica/Replica/<scene>/results (depthNNNNNN.png, frameNNNNNN.jpg, traj.txt one level up; cam_params.json in Replica/).
Step 1.3 — CUDA BACKEND via cuda-oxide (load skill kornia-cuda-kernel-dev; read docs/cuda-oxide.md). Toolchain installed at ~/.cargo/cuda-oxide (. ~/.cargo/cuda-oxide/env.sh), set CARGO_BUILD_JOBS=1 for build_ptx too.
- crates/kornia-oxide-kernels/device/src/tsdf.rs + tsdf_ops.rs (PtxOps: ptx_asm! fma.rn.f32, sqrt.rn.f32): kernels tsdf_block_keys, tsdf_alloc_mark (atomics build miss set only), tsdf_rank_misses/assign (sorted misses -> pool index by rank, deterministic), tsdf_integrate_pinhole, tsdf_integrate_spherical, tsdf_raycast, tsdf_mc_count, tsdf_mc_emit (+ incremental dirty-block variants). Raw pointers, all loads before first store. FMA_ALLOWED entries listed. Extend ptx_gate.rs per contract G1/G2/G4 (+ its tests in tests/ptx_gate.rs).
- kornia-3d: Cargo feature cuda adds kornia-oxide-kernels/{cuda,ptx}; crates/kornia-3d/src/tsdf/cuda.rs: CudaTsdfVolume<B>, CudaSubmapTsdf<B>, device pools allocated once, workspace reused, zero host allocs per warmed frame, one stream sync per frame max, sm<75 -> Err. Use kornia_oxide_kernels::__private::kernel(stream, entry::X).
- Tests cuda_tests.rs named <entry>_bit_exact against the CPU path: analytic + 20 Replica frames, B=8/16, multi-sensor, multi-agent, edge sizes (1xN, Nx1, odd), sentinel-filled outputs, degenerate inputs -> Err; incremental == full mesh on device; device result == scalar bit for bit (allowing only hash slot positions to differ, per contract section 14).
Commit: feat(3d): tsdf cuda-oxide kernels (bit-exact with CPU)`, 'CUDA')

return { scalar, guard, tdd, neon, cuda }
