import json
B = "/mnt/data/kr-tsdf-spikes/results/baselines/vdbfusion/runs"
asis = json.load(open(f"{B}/room0_v0.05_n50_conv_asis/score_M.json")); flip = json.load(open(f"{B}/room0_v0.05_n50_conv_flip/score_M.json"))
pil = json.load(open(f"{B}/room0_v0.02_n50_pilot/bench.json"))
json.dump(dict(
  convention_check=dict(note="room0, voxel 0.05, first 50 frames, Row M mesh, scorer culled to frames 0:50. VDBFusion integrate takes WORLD points + uses extrinsic[:3,3] only as ray origin; the supplied /mnt/data/kr-tsdf-spikes/vdbfusion_smoke.py passed CAMERA-frame points and flipped y/z, both wrong.",
    as_is=dict(run=f"{B}/room0_v0.05_n50_conv_asis", accuracy_cm=asis["accuracy_cm"], completeness_cm=asis["completeness_cm"], completion_ratio_pct=asis["completion_ratio_pct"]),
    flip_yz=dict(run=f"{B}/room0_v0.05_n50_conv_flip", accuracy_cm=flip["accuracy_cm"], completeness_cm=flip["completeness_cm"], completion_ratio_pct=flip["completion_ratio_pct"])),
  pilot_2cm_50f=dict(run=f"{B}/room0_v0.02_n50_pilot", e2e_median_ms=pil["integrate_e2e_ms"]["median"]),
  subset_decision="not needed: ~0.81 s/frame -> ~29.6 min wall per 2000-frame run (< 60 min), all 2000 frames run for every config",
  deviations=["no weight cap (Row M cap 64 not expressible; constant +1 uncapped, same as Open3D VBG)", "no colour", "single-threaded integrate (upstream sequential std::for_each)",
              "trunc fixed 4 voxels for Row R too (upstream RGB-D configs 3/7 voxels)", "depth filter on camera z [0.1,7.0) (upstream filters ray norm)",
              "Row M and Row R share ONE integration run per scene x voxel (identical integration params; only mesh min_weight 1 vs 5 differs)",
              "Row R mesh 'cold' call follows Row M's three calls in the same process"],
  sampling_floor_cm=dict(room0=1.45, office0=1.12)), open("/mnt/data/kr-tsdf-spikes/results/baselines/vdbfusion/meta.json", "w"), indent=1)
