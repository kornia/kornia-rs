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
| Feature extraction (ORB / SIFT) | `features.rs` |
| Sliding-window matching | `matching.rs` |
| Track building | `kornia_calib::build_tracks` |
| Incremental SfM | `reconstruction.rs` → `kornia_calib::reconstruct` |
| PLY export + normals | `ply_writer.rs` |

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

## Output

The PLY uses the `XYZRgbNormals` layout that `kornia_3d::io::ply::read_ply_binary`
can read back: `x y z` (f32), `red green blue` (u8), `nx ny nz` (f32), written
little-endian at 27 bytes per vertex. Colours are sampled from the source RGB
frames at each point's first track observation; normals are estimated from the
k nearest neighbours via PCA and oriented toward the cameras.

View the result in MeshLab, CloudCompare, or with the `ply_rerun` example.

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