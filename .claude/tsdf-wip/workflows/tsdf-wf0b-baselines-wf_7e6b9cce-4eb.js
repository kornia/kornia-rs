export const meta = {
  name: 'tsdf-wf0b-baselines',
  description: 'Measure TSDF baselines (nvblox, Open3D VBG CUDA/CPU, VDBFusion, InfiniTAM) serially on a quiet box, merge targets, refute',
  phases: [
    { title: 'Scorer', detail: 'shared accuracy/completeness script vs GT' },
    { title: 'Measure', detail: 'one library at a time, never concurrent' },
    { title: 'Merge', detail: 'baseline_targets.json' },
    { title: 'Refute', detail: 'baseline skeptic' },
  ],
}

const R = '/mnt/data/kr-tsdf-spikes/results/baselines'
const COMMON = `
Context: kornia-rs TSDF project (beat nvblox and Open3D on every axis). This step (WF-0b) measures BASELINES ONLY; our own library is not timed here.
Read FIRST: /mnt/data/kr-tsdf-spikes/frozen/DECISIONS.md (section "Frozen bench params" is binding: Row M matched params and Row R each library's recommended config; axes list) and /home/nvidia/.claude/plans/yes-prototype-it-in-vivid-robin.md items 4, 4a, 5 (fairness + hygiene). Load the orin-gpu-profiling skill for GPU timing method.
Data: /mnt/data/datasets/replica/Replica/{room0,office0}/results/{frameNNNNNN.jpg,depthNNNNNN.png}, traj.txt (4x4 camera-to-world, row-major, OpenGL-style axes as NICE-SLAM; verify the convention you use against the GT mesh), GT meshes /mnt/data/datasets/replica/Replica/{room0,office0}_mesh.ply. 2000 frames, 1200x680, fx=fy=600, cx=599.5, cy=339.5, depth scale 6553.5.
Box: Jetson Orin, 7.4 GB RAM, MAXN_SUPER, jetson_clocks set by the user. Record clock state at start and end of your runs (cat /sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/cur_freq or the 17000000.ga10b path, nvpmodel -q). You are the ONLY job on the box: never start compiles, never run two timed processes at once. Before each timed run check no other heavy process (pgrep cargo/cc1plus/python benchmark) is running; if one is, wait and note it.
Rules: no sudo, no apt, no docker, no git push, no pip install into system python (venvs under /mnt/data only if strictly needed and say so). Never edit /mnt/data/src/kornia-rs. Scratch scripts go under ${R}/scripts/. Warm up (discard first 5 frames' timings or a warm-up pass), sync timing (CUDA: device sync / events, never async enqueue time). Prove the GPU path ran (nsys or launch count) for GPU libraries. Each library in its OWN recommended config for Row R; no per-library tuning beyond what DECISIONS.md fixes.
Axes to record per (library, scene, voxel, row): integrate ms/frame end-to-end incl. allocation + H2D (mean, median, p95, min, max over warmed frames; say exactly what the timer covers and what it excludes, e.g. image decode excluded for everyone); full mesh extraction ms; incremental mesh ms where the library has it (else "n/a"); peak host RSS (/usr/bin/time -v) and device memory (Orin is unified: cudaMemGetInfo delta or tegrastats sampled; state method); block/voxel count; triangle count; accuracy and completeness vs GT via the shared scorer ${R}/scripts/score_mesh.py (cm, % < 5 cm). If a run cannot be done, report it as "not run" with the reason, never silently drop it.
Write raw logs under ${R}/<library>/ and a machine-readable ${R}/<library>/results.json (list of rows with the fields above plus config, command, commit/version, clock state). Return a schema'd result; a claimed number must point at its log file.
`
const ROW = {
  type: 'object',
  properties: {
    status: { type: 'string', enum: ['ok', 'partial', 'failed'] },
    summary: { type: 'string' },
    results_json: { type: 'string' },
    evidence: { type: 'array', items: { type: 'string' } },
    not_run: { type: 'array', items: { type: 'string' } },
    fairness_notes: { type: 'array', items: { type: 'string' } },
  },
  required: ['status', 'summary', 'evidence', 'not_run', 'fairness_notes'],
}

phase('Scorer')
const scorer = await agent(`${COMMON}
YOUR TASK (scorer): write ${R}/scripts/score_mesh.py, the ONE accuracy/completeness scorer every library (and later ours) uses. Use the Open3D CPU wheel venv: run with /mnt/data/venvs/tsdf-bench/bin/python -I. Method (NICE-SLAM / standard): sample N=200k points uniformly on predicted and GT meshes (fixed seed), accuracy = mean dist pred->GT (cm), completeness = mean dist GT->pred (cm), completion ratio = % GT points within 5 cm, plus precision % pred within 5 cm. Input: mesh path (ply/stl/obj), GT path, optional --cull-to-frustum using traj.txt + intrinsics over the frames integrated (so a run over N<2000 frames is not penalised for unseen GT; document). Output one JSON line. Validate: GT vs itself gives ~0; a 1 cm shifted GT gives ~1 cm; report both. Also write ${R}/scripts/README.md (one paragraph per script). Do not time anything.`, { label: 'scorer', phase: 'Scorer', schema: ROW, agentType: 'oh-my-claudecode:scientist', effort: 'high' })

const LIBS = [
  { key: 'nvblox', prompt: `nvblox v0.0.10 ONLY. Built at /mnt/data/nvblox-build (source /mnt/data/src/nvblox-v0.0.10); runner /mnt/data/nvblox-build/executables/fuse_replica. Earlier smoke logs (20 frames, room0): /mnt/data/kr-tsdf-spikes/results/nvblox_smoke/ (run.log, run02.log, run05.log, t02.txt) show its timers table and flags. Use its built-in timers AND an outer wall clock; report both. Row M flags per DECISIONS.md (--projective_integrator_weighting_mode=0 constant weight, max_weight 64, trunc 4 vox, depth range). Row R = its defaults. Block size is fixed 8. Mesh: it meshes incrementally; get full-mesh time too if exposed (else n/a with reason). Use nsys (/usr/local/cuda/bin/nsys) on one room0 2 cm run to prove GPU path and capture the kernel/H2D split.` },
  { key: 'open3d', prompt: `Open3D 0.20 tensor VoxelBlockGrid: CUDA build via PYTHONPATH=/mnt/data/venvs/o3d-cuda-site:/mnt/data/venvs/o3d-cuda-deps /usr/bin/python3.10 (no -I, it drops PYTHONPATH), device CUDA:0; CPU via /mnt/data/venvs/tsdf-bench/bin/python -I, device CPU:0. Both are rows (CUDA is the PRIMARY target). An earlier rough script exists: /tmp/claude-1000/-home-nvidia/032ee15f-a6f3-4521-b787-7ab5768564c4/scratchpad/o3d_vbg_timing.py (VBG attrs tsdf/weight/color float32, block 16, compute_unique_block_coordinates + integrate synced, extract_triangle_mesh); copy and extend it into ${R}/scripts/. Timer must INCLUDE the H2D of depth+colour tensors (state it), and sync (o3d.core.cuda.synchronize()). block_count sized for the scene (report). Row R per DECISIONS.md (trunc 8 vox, depth_max 3, weight_threshold 3, note its 0-vertex issue). Also ray_cast timing (one 1200x680 view, synced) for both devices. Legacy ScalableTSDFVolume is broken in the aarch64 wheel: confirm in one line and mark not run. Use nsys on one CUDA room0 2 cm run.` },
  { key: 'vdbfusion', prompt: `VDBFusion 0.1.6 (CPU). Run: PYTHONPATH=/mnt/data/venvs/vdbfusion LD_LIBRARY_PATH=/mnt/data/localdeps/root/usr/lib/aarch64-linux-gnu /usr/bin/python3.10 ; build script /mnt/data/kr-tsdf-spikes/vdbfusion_build.sh; smoke /mnt/data/kr-tsdf-spikes/vdbfusion_smoke.py (camera-frame points from depth + 4x4 pose, flips y,z of traj.txt for OpenCV axes: VERIFY against GT via the scorer before timing). Its recommended config: space_carving False, sdf_trunc = 4 voxels (Row M = Row R except weights; it has no colour: state that, colour off for it), preprocess on. Timer covers depth->points unprojection + integrate (state it, and also report integrate-only). Mesh = extract_triangle_mesh (full only, incremental n/a). If 2000 frames at 2 cm exceeds 60 min wall, run a documented contiguous subset (first K frames) for timing and say so, but accuracy must use the frustum-culled scorer.` },
  { key: 'infinitam', prompt: `InfiniTAM v3 (CUDA), origin/infinitam_v3 c423b89b. Binary /mnt/data/infinitam-build/Apps/InfiniTAM_cli/InfiniTAM_cli (one-line patch: INFINITAM_MESH_OUT env writes the mesh after Run()); build script /mnt/data/kr-tsdf-spikes/infinitam_build.sh; converter /mnt/data/kr-tsdf-spikes/replica_to_infinitam.py <N> writes /mnt/data/datasets/replica/infinitam/room0 (ppm/pgm + calib.txt, depth stored as depth_m*3276.75 with affine calib). Smoke log /mnt/data/infinitam-smoke.log. MUST fix before timing: (a) it tracks its own pose with ICP; switch to its file-based tracker (ITMLib/Trackers/ITMTrackerFactory.h:84, "file" tracker; check how the CLI/ITMLibSettings selects trackerConfig and what pose file format/axes it expects) fed with the GT traj.txt poses, so all libraries integrate at GT poses; if that needs a source edit to the CLI, keep it minimal and in the build script, rebuild with -j1 (you may compile ONLY this, before any timing); (b) set voxelSize per row (0.02/0.05) and mu = 4 voxels, swapping off, block 8 fixed; verify the setting took (the 50-frame smoke mesh had 1.4M tris, suspicious). It has no incremental mesh. It's mesh is not run-to-run identical: run twice, report both. Converter needs office0 too. Use nsys on one room0 2 cm run.` },
]

phase('Measure')
const measured = []
for (const L of LIBS) {
  const r = await agent(`${COMMON}
Shared scorer (already written by the previous step): ${R}/scripts/score_mesh.py — scorer summary: ${scorer ? scorer.summary.slice(0, 1500) : 'scorer step failed; write a minimal one yourself and say so'}
YOUR LIBRARY: ${L.key}. ${L.prompt}
Scenes: room0 and office0. Voxels 0.02 and 0.05. Rows M and R. All 2000 frames unless the library note says otherwise.`, { label: `measure-${L.key}`, phase: 'Measure', schema: ROW, agentType: 'oh-my-claudecode:scientist-high', effort: 'high' })
  measured.push({ lib: L.key, r })
  log(`${L.key}: ${r ? r.status : 'null'}`)
}

phase('Merge')
const merged = await agent(`${COMMON}
YOUR TASK (merge): read every ${R}/*/results.json and the agents' summaries below; write /mnt/data/kr-tsdf-spikes/results/baseline_targets.json: for each axis x scene x voxel x row, the BEST baseline value and which library holds it (the target we must beat), plus the full per-library table, and ${R}/SUMMARY.md (human table, one per axis, with fairness notes and every "not run"). Do not re-run anything except cheap reads.
Agent summaries: ${JSON.stringify(measured.map(m => ({ lib: m.lib, status: m.r && m.r.status, summary: m.r && m.r.summary, not_run: m.r && m.r.not_run, fairness: m.r && m.r.fairness_notes })))}`, { label: 'merge', phase: 'Merge', schema: ROW, agentType: 'oh-my-claudecode:scientist', effort: 'medium' })

phase('Refute')
const skeptic = await agent(`${COMMON}
YOUR TASK (baseline-skeptic, plan step 0.7): try to REFUTE the baseline numbers in /mnt/data/kr-tsdf-spikes/results/baseline_targets.json and ${R}/SUMMARY.md using the raw logs and scripts under ${R}. For each library check: recommended config really used (read its docs/source under /mnt/data/src where present); timing synced and warmed; what the timer includes/excludes is the same across libraries (H2D, allocation, decode); GPU path really hit; poses/axes correct (accuracy sane vs GT); frame counts equal; clocks locked during runs; any sandbagging (a baseline made slower or less accurate than it should be) or the reverse. Read-only: do not run timings. Return findings, each with file evidence and a verdict (sound / suspect / wrong) and what to re-run.`, { label: 'baseline-skeptic', phase: 'Refute', agentType: 'oh-my-claudecode:critic', effort: 'high', schema: {
  type: 'object',
  properties: {
    summary: { type: 'string' },
    findings: { type: 'array', items: { type: 'object', properties: { library: { type: 'string' }, claim: { type: 'string' }, verdict: { type: 'string', enum: ['sound', 'suspect', 'wrong'] }, evidence: { type: 'string' }, rerun: { type: 'string' } }, required: ['library', 'claim', 'verdict', 'evidence'] } },
  },
  required: ['summary', 'findings'],
} })

return { scorer, measured, merged, skeptic }
