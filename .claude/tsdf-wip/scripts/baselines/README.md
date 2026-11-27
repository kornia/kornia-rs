# baselines/scripts

All scripts run with the Open3D 0.20 CPU wheel venv: `/mnt/data/venvs/tsdf-bench/bin/python -I <script> ...`
(numpy 2.2.6, scipy 1.15.3). Nothing here is timed.

**score_mesh.py** is the ONE accuracy/completeness scorer for every library (ours later). Usage:
`score_mesh.py MESH --gt GT.ply [--cull-to-frustum --traj traj.txt --frames 0:2000:1] [-n 200000] [--seed-pred 0 --seed-gt 1]`;
it prints exactly one JSON line (diagnostics on stderr; exit 2 + `"error"` on a bad input; a mesh with 0 vertices and 0
triangles, e.g. Open3D VBG at weight_threshold 3, gives `"empty": true` with null metrics and exit 0; vertices without
triangles is refused as "not a mesh"). Method (NICE-SLAM): N=200k points sampled uniformly by area on pred and GT with a
numpy RNG (pred seed 0, GT seed 1 -- distinct on purpose), `accuracy_cm` = mean pred->GT nearest-sample distance,
`completeness_cm` = mean GT->pred, `completion_ratio_pct` = % GT samples within 5 cm, `precision_pct` = % pred samples
within 5 cm, plus `f_score_pct`. Extra fields `*_surf_*` use point-to-SURFACE distance (Open3D RaycastingScene) for the same
samples. PLY is read by an own reader because Open3D silently drops all quads: on the Replica GT it returns 16,516 of
953,647 faces with only a warning (GT meshes are all quads; they are fan-triangulated here: room0 1,907,294 triangles,
169.5 m^2; office0 1,177,518 triangles). Pred and GT must be in the traj.txt world frame (no alignment is done).
**Sampling floor:** with 200k samples the sample-to-sample metrics have a floor even for a perfect mesh (GT vs itself with
different seeds: room0 1.45 cm, office0 1.12 cm accuracy and completeness; the *_surf fields give 0.00). Every library inherits the
same floor, so compare libraries with the same N and seeds, and read cm values near 1-2 cm with this in mind. Self-tests
(`scorer/validation.log`): same seed -> exactly 0; a rigid 1 cm shift along (1,1,1)/sqrt3 reads 0.56 cm in *_surf (parallel
surfaces do not move: mean |n.d| ~ 0.577 cm) and 1.59 cm sample-to-sample; a 1 cm offset along vertex normals reads 0.99 cm *_surf
(1.81 cm sample-to-sample, room0).

`--cull-to-frustum` (so a run over fewer than 2000 frames, or a library with a shorter range, is not charged for GT it never
saw): a GT sample is KEPT if in at least one frame of `--frames` (python slice of the traj, default all) it has camera z in
[0.1, 7.0) m (the frozen Row M sensor range, NOT a library's depth_max, so Open3D Row R depth_max=3 completeness loss is
still counted), projects inside the 1200x680 image (fx=fy=600, cx=599.5, cy=339.5), and the sensor depth at that pixel
(`results/depthNNNNNN.png / 6553.5`, read from `<traj dir>/results`) is valid and z <= depth + 0.05 m (the depth test:
points behind the observed surface do not count as seen). Only GT is culled by default (`--cull-pred` also culls pred; off by
default so floaters hurt accuracy). The GT mask is cached under `../cache/` keyed by GT hash, frames, seed, N, range,
margin, convention, intrinsics, traj hash (first computation ~1 to 2 min for 2000 frames, then instant). Kept fractions: room0 30.1%
(first 100 frames) vs 86.7% (all 2000); office0 26.1% vs 78.3%. The JSON reports `gt_points_kept/total`, `gt_kept_frac`.
**Camera convention (binding for all runners):** the Replica `traj.txt` c2w matrices are used AS-IS, i.e. OpenCV axes
(x right, y down, z forward). Verified by projecting GT samples with each convention against the sensor depth over 8 frames:
no flip gives median |z-depth| 3.0 mm (room0) / 1.7 mm (office0) with 94-96% of points within 2 cm; flipping the y/z columns
(the "OpenGL/NICE-SLAM" reading of the task text) gives ~1 m median error and 1-3% within 2 cm. `--cam-convention opencv` is the default.

**validate_scorer.py** reproduces the scorer's validation: (1) the camera-convention test above on room0 and office0,
(2) the self-tests (same seed, different seed = sampling floor, 1 cm rigid shift, 1 cm normal offset), (3) the cull
sanity (kept fraction for 100 vs 2000 frames; this also warms the mask cache for the full 2000-frame runs), (4) degenerate
inputs (empty mesh, vertex-only cloud, ascii single triangle, missing file). Run it as
`/mnt/data/venvs/tsdf-bench/bin/python -I validate_scorer.py > ../scorer/validation.log` (about 5 min, CPU only, no GPU).
