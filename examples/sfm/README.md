# Structure-from-Motion from Video

End-to-end SfM example for the kornia-rs ecosystem: read a video file,
extract per-frame features, match frames, chain the matches into multi-view
tracks, reconstruct a sparse 3D point cloud, and export it as a binary PLY
file (XYZ + RGB + normals).

## Pipeline

```
MP4 ─► RGB+gray frames ─► features (ORB/SIFT) ─► pairwise matches
   ─► build_tracks ─► kornia_calib::reconstruct ─► PLY (XYZ + RGB + normals)
```

| Stage | Module |
|---|---|
| Frame decoding (GStreamer) | `video.rs` |
| Feature extraction (ORB / SIFT, sync or parallel) | `features.rs` |
| Sliding-window matching (sync or parallel) | `matching.rs` |
| Track building | `kornia_calib::build_tracks` |
| Incremental SfM | `reconstruction.rs` → `kornia_calib::reconstruct` |
| PLY export + normals | `ply_writer.rs` |
| Rerun visualization | `ply_viewer.rs` |

## Usage

```sh
cargo run -p sfm -- <video> <output.ply> --fx <fx> --fy <fy> --cx <cx> --cy <cy> [options]
```

Camera intrinsics (`--fx`, `--fy`, `--cx`, `--cy`, in pixels) are required.

### Example

```sh
cargo run -p sfm -- sample.mp4 out.ply \
    --fx 600 --fy 600 --cx 320 --cy 240 \
    --detector orb --n-features 2000 --match-window 5 --frame-step 1
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--detector` | `orb` | Feature detector: `orb` (binary, fast) or `sift` (float, robust). |
| `--n-features` | `2000` | Max features per frame. |
| `--match-window` | `5` | Match each frame against this many following frames. |
| `--ratio` | `0.8` | Lowe's ratio-test threshold (lower = stricter). |
| `--frame-step` | `1` | Process every Nth frame (1 = all frames). |
| `--threads` | `0` | Worker threads for parallel feature extraction and matching (`0` = auto-detect CPU count). |
| `--view` | off | Open the output PLY in the rerun viewer after writing. |
| `--max-ba-iterations` | `100` | Bundle-adjustment LM iterations. Lower = faster but less accurate. |
| `--min-registration-inliers` | `30` | Min PnP inliers to register a view. Lower admits more cameras (looser). |
| `--motion-prior-sigma` | `0.0` | Constant-velocity motion prior (`0.0` = off). Use for smooth walkthroughs. |
| `--up-prior-sigma` | `0.0` | Camera-up prior (`0.0` = off). Use for handheld upright capture. |
| `--max-reprojection-error` | `0.01` | Reprojection-error threshold (normalized units). |
| `--geo-verify` | off | Verify matches with epipolar RANSAC after matching (rejects false matches). |
| `--geo-threshold` | `3.0` | Epipolar RANSAC inlier threshold (pixels). |
| `--geo-min-inliers` | `15` | Min inliers for a pair's fundamental matrix to be trusted. Values below 15 warn — at 8 the 8-point algorithm's minimal sample validates itself. |
| `--cuda` | off | Use CUDA for SIFT extraction (requires an NVIDIA GPU). |
| `--sprt` | off | Enable Wald's SPRT for PnP registration (rejects bad candidate poses early). |
| `--sprt-epsilon` | `0.5` | SPRT expected inlier ratio (capped at 0.3 until a consensus exists). |
| `--sprt-delta` | `0.05` | SPRT Type-I error: probability of rejecting a good pose. |
| `--refine-intrinsics` | off | Refine focal + radial/tangential distortion against the reconstruction. |
| `--wide-baseline` | `0` | Also match frame `i` against `i+K, i+2K, …` (`0` = off). |

### Recommended flags by capture type

- **SIFT speed**: add `--cuda` to run SIFT on the GPU (needs the CUDA runtime
  on `LD_LIBRARY_PATH`). NVRTC kernels are JIT-compiled on the first frame.
- **Noisy matches / poor ORB reconstruction**: add `--geo-verify`.
- **Long videos**: raise `--frame-step` (fewer cameras to register).

## Matching model and future upgrades

Every frame pair is matched with **mutual nearest neighbours (cross-check)**: a
pair is kept only when each descriptor is the other's nearest neighbour. This is
a hard requirement of the track builder — `kornia_calib::build_tracks` assumes
one-to-one pair matches and discards any track that reaches one camera at two
different pixels. A forward-only matcher can emit many-to-one matches (two
keypoints of one camera sharing a neighbour), which silently culls whole
multi-view tracks; the ORB path regressed to that in `98ecd0b` and was reverted
to a cross-checked matcher here.

Planned upgrades (each belongs in a core crate, not this example):

| Idea | Where | Why |
|---|---|---|
| Orientation-histogram filtering on top of mutual NN | `kornia-imgproc` (`OrbMatchConfig`) | `98ecd0b`'s rotation-consistency check is useful but was shipped without cross-check; combine both. |
| Per-target injective matching | `kornia-imgproc` | Weaker than mutual NN: kills the shared-neighbour fan-in while keeping strictly more matches. |
| Cross-octave ORB keypoint de-duplication | `kornia-imgproc` (ORB extractor) | ORB emits near-duplicate keypoints 1–3 px apart across scales; removing them attacks the fan-in at its source. |
| `build_tracks` conflict recovery (split instead of drop) | `kornia-calib` | Recover the consistent sub-track instead of discarding the whole component. |
| Pose-guided matching | `kornia-imgproc` (`match_orb_by_projection`) | Already exists; use it once incremental poses are available. |

## Performance notes

- Feature extraction and frame-pair matching run on a rayon thread pool;
  `--threads` sizes it (`0` = auto-detect).
- Video *decode* is paced by GStreamer's real-time clock (`sync=true` in
  `kornia-io`'s `VideoReader`), so reading a 21 s clip takes ~21 s. Use
  `--frame-step` to keep fewer frames. Overlapping decode with downstream work
  is tracked as future work (it needs a fast-read path in `kornia-io`).
- Results are deterministic for a given input: matching, track building, and
  the reconstruction do not depend on run-to-run ordering.

## Benchmarks

`swiss_knife.mp4` — 628 frames, 1280x720 HEVC orbit. Intrinsics
`--fx 895 --fy 895 --cx 640 --cy 360`, `--n-features 500`, `--frame-step 10`
(63 frames) unless noted. CUDA rows use `--cuda` (device SIFT extraction *and*
matching). Decode is clock-paced at ~24 s and is included in `Total`.

| Config | Detector | Extract (s) | Match (s) | Geo (s) | Matches (raw→verified) | Tracks | Recon (s) | Points | Views | RMSE (px) | Total (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `orb_base` | ORB | 2.6 | 4.3 | — | 19810 | 4048 | 1.8 | 125 | 2/63 | 0.761 | 32.3 |
| `orb_geo` | ORB | 2.5 | 4.4 | 3.6 | 19810→14286 | 4011 | 0.7 | 137 | 2/63 | 0.803 | 34.7 |
| `orb_geo --geo-min-inliers 8` | ORB | 2.7 | 4.6 | 3.7 | 19810→15385 | 4148 | 3.7 | 670 | 10/63 | 1.072 | 38.3 |
| `orb_geo --wide-baseline 20` | ORB | 2.9 | 5.0 | 4.4 | 20973→14428 | 3985 | 4.6 | 697 | 12/63 | 1.111 | 40.5 |
| `sift_base` | SIFT (CUDA) | 0.8 | 0.1 | — | 21667 | 3483 | 3.1 | 520 | 9/63 | 2.244 | 27.9 |
| `sift_geo` | SIFT (CUDA) | 1.1 | 0.1 | 2.1 | 21667→16935 | 3774 | 8.0 | 806 | 16/63 | 1.842 | 35.1 |
| `sift_geo_sprt` (champion) | SIFT (CUDA) | 0.8 | 0.1 | 2.2 | 21667→16935 | 3774 | 8.4 | 805 | 16/63 | 1.848 | 35.1 |
| champion, no `--cuda` | SIFT (CPU) | 18.8 | 12.7 | 2.1 | 21667→16935 | 3774 | 7.8 | 805 | 16/63 | 1.848 | 64.9 |
| champion + `--refine-intrinsics` | SIFT (CUDA) | 0.9 | 0.1 | 2.3 | 21667→16935 | 3774 | 9.2 | 805 | 16/63 | 1.831 | 36.1 |
| champion + `--wide-baseline 20` | SIFT (CUDA) | 0.8 | 0.1 | 2.3 | 22643→17147 | 3707 | 13.4 | 914 | 19/63 | 1.910 | 40.2 |
| champion, `--frame-step 5` | SIFT (CUDA) | 1.3 | 0.2 | 3.2 | 70314→60355 | 6510 | 29.7 | 1703 | 31/126 | 3.380 | 58.0 |

Observations:

- **Deterministic**: two identical champion runs produce byte-identical PLY.
- **`--cuda` speeds up both stages with identical output**: extraction
  18.8 s → 0.8 s and matching 12.7 s → 0.1 s; the reconstruction is
  byte-identical to the CPU path.
- **`--geo-verify` is the SIFT accuracy lever**: 9 → 16 views, RMSE 2.24 →
  1.84 px.
- **`--wide-baseline 20`** maximises coverage at this frame step (914 pts,
  19/63 views).
- **`--refine-intrinsics`** is near-neutral here (the EXIF focal is already
  accurate: it fits `gamma≈1.00`).
- **`--frame-step 5`** gives the densest map (1703 pts, 31/126 views) at higher
  RMSE.
- **ORB sensitivity**: with the default `--geo-min-inliers 15` ORB registers
  2/63 views on this clip; `--geo-min-inliers 8` keeps enough weak pairs to
  register 10/63. SIFT has denser per-pair inliers and is unaffected.

Reproduce the champion (device SIFT, requires an NVIDIA GPU with NVRTC on the
library path):

```sh
export LD_LIBRARY_PATH=<dir containing libnvrtc.so>
cargo run -p sfm -- swiss_knife.mp4 out.ply \
    --fx 895 --fy 895 --cx 640 --cy 360 --n-features 500 --frame-step 10 \
    --detector sift --geo-verify --sprt --cuda
```

## Output

The PLY uses the `XYZRgbNormals` layout that `kornia_3d::io::ply::read_ply_binary`
can read back: `x y z` (f32), `red green blue` (u8), `nx ny nz` (f32), written
little-endian at 27 bytes per vertex. Colours are sampled from the source RGB
frames per raw observation (before reconstruction), and each point takes the
colour of a surviving observation; normals are estimated from the k nearest
neighbours via PCA and oriented toward the point's own observing cameras.

View the result in MeshLab, CloudCompare, the `ply_rerun` example, or with
this example's `--view` flag (requires the [rerun](https://rerun.io) viewer,
`pip install rerun-sdk`).

## Notes and limitations

- **Up to scale**: without an AprilTag anchor, the reconstruction is recovered
  up to an unknown global scale (`ScaleSource::UpToScale`). Shape is correct,
  units are arbitrary.
- **Sparse cloud**: only tracked feature points are reconstructed, not a dense
  surface.
- **Requires `gstreamer`**: the `kornia-io` gstreamer feature must be enabled
  (it is in this example's `Cargo.toml`).
- **Intrinsics must match the decoded frame.** Phone videos shot in portrait are
  often coded 1280x720 landscape with a `rotation=-90` metadata tag, and
  GStreamer decodes them unrotated — so supply intrinsics for the *decoded*
  resolution (principal point at its centre), not the displayed portrait frame.
  The focal length can be derived from the EXIF 35 mm-equivalent:
  `fx = f_35mm / 36 * frame_width`.
- **Frame width must satisfy `3*W % 4 == 0`.** GStreamer pads RGB rows to a
  4-byte boundary and `kornia-io` currently ignores the stride, so other widths
  (e.g. 854x480) would be read sheared with no error; the example rejects them
  until the core fix lands.
- **No distortion model**: the supplied intrinsics are treated as an ideal
  pinhole (zero distortion).

## Tests

```sh
cargo test -p sfm --bin sfm
```

Tests use fabricated inputs (synthetic checkerboards, descriptor sets, and a
projected 3D scene) plus a PLY write→read round-trip against
`kornia_3d::io::ply::read_ply_binary`. No external video files are required.
