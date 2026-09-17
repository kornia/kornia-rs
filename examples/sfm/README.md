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
| `--orb-no-orientation-check` | off | Disable ORB-SLAM3 orientation-histogram filtering. Helps on orbit/turntable captures where the camera rotates systematically. |
| `--max-ba-iterations` | `100` | Bundle-adjustment LM iterations. Lower = faster but less accurate. |
| `--min-registration-inliers` | `30` | Min PnP inliers to register a view. Lower admits more cameras (looser). |
| `--motion-prior-sigma` | `0.0` | Constant-velocity motion prior (`0.0` = off). Use for smooth walkthroughs. |
| `--up-prior-sigma` | `0.0` | Camera-up prior (`0.0` = off). Use for handheld upright capture. |
| `--max-reprojection-error` | `0.01` | Reprojection-error threshold (normalized units). |
| `--geo-verify` | off | Verify matches with epipolar RANSAC after matching (rejects false matches). |
| `--geo-threshold` | `3.0` | Epipolar RANSAC inlier threshold (pixels). |
| `--geo-min-inliers` | `8` | Min inliers for a pair's fundamental matrix to be trusted. |
| `--cuda` | off | Use CUDA for SIFT extraction (requires an NVIDIA GPU). |

### Recommended flags by capture type

- **Orbit/turntable captures** (camera circles a static object): add
  `--orb-no-orientation-check` for ORB.
- **SIFT speed**: add `--cuda` to run SIFT on the GPU (needs the CUDA runtime
  on `LD_LIBRARY_PATH`). NVRTC kernels are JIT-compiled on the first frame.
- **Noisy matches / poor ORB reconstruction**: add `--geo-verify`.
- **Long videos**: raise `--frame-step` (fewer cameras to register).

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
