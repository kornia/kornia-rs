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
| Frame decoding (GStreamer, sync or async) | `video.rs` |
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

### Fast path (async + parallel)

```sh
cargo run -p sfm -- sample.mp4 out.ply \
    --fx 600 --fy 600 --cx 320 --cy 240 \
    --async-video --buffer-size 64 --threads 8
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--detector` | `orb` | Feature detector: `orb` (binary, fast) or `sift` (float, robust). |
| `--n-features` | `2000` | Max features per frame. |
| `--match-window` | `5` | Match each frame against this many following frames. |
| `--ratio` | `0.8` | Lowe's ratio-test threshold (lower = stricter). |
| `--frame-step` | `1` | Process every Nth frame (1 = all frames). |
| `--async-video` | off | Enable async video reading plus parallel feature extraction and matching. |
| `--threads` | `0` | Worker threads for parallel stages (`0` = auto-detect CPU count). |
| `--buffer-size` | `32` | Channel buffer (frames) for async video reading; larger = less backpressure, more memory. |
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

### Performance notes

- **Sync mode** is the original sequential pipeline (useful as a baseline).
- **Async mode** (`--async-video`) decodes video on a tokio task with a
  bounded channel and runs feature extraction and frame-pair matching on a
  rayon thread pool. For many frames this can give a near-linear speedup on
  the extraction/matching stages.
- Video *decode* itself is paced by GStreamer's real-time clock (`sync=true`
  in `kornia-io`'s `VideoReader`), so reading a 21 s clip takes ~21 s in
  either mode. Use `--frame-step` to reduce the number of frames kept.
- The parallel and sequential paths produce byte-identical results
  (deterministic ordering).

## Output

The PLY uses the `XYZRgbNormals` layout that `kornia_3d::io::ply::read_ply_binary`
can read back: `x y z` (f32), `red green blue` (u8), `nx ny nz` (f32), written
little-endian at 27 bytes per vertex. Colours are sampled from the source RGB
frames at each point's first track observation; normals are estimated from the
k nearest neighbours via PCA and oriented toward the cameras.

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
- **No distortion model**: the supplied intrinsics are treated as an ideal
  pinhole (zero distortion).

## Tests

```sh
cargo test -p sfm --bin sfm
```

Tests use fabricated inputs (synthetic checkerboards, descriptor sets, and a
projected 3D scene) plus a PLY write→read round-trip against
`kornia_3d::io::ply::read_ply_binary`. No external video files are required.
