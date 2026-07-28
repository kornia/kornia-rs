# AprilTag board multi-camera calibration

This example estimates the pose of two or more cameras from images of the same
stationary MakerWorld 6x6 AprilTag board. The first image is camera 0 and becomes
the reference frame for the reported relative poses and baselines.

## Run

Pass one image path per camera. At least two paths are required:

```bash
pixi run cargo run -p apriltag_board_multicam \
  -- \
  /path/to/camera_0.png \
  /path/to/camera_1.png
```

Add more paths for larger rigs:

```bash
pixi run cargo run -p apriltag_board_multicam \
  -- \
  /path/to/camera_0.png \
  /path/to/camera_1.png \
  /path/to/camera_2.png
```

Use `--help` to show the command-line help:

```bash
pixi run cargo run -p apriltag_board_multicam -- --help
```

## Save the Rerun recording

By default, the example opens the Rerun viewer. Set `RERUN_SAVE` to save an
`.rrd` file instead:

```bash
RERUN_SAVE=/tmp/apriltag_board_multicam.rrd \
  pixi run cargo run -p apriltag_board_multicam \
  -- \
  /path/to/camera_0.png \
  /path/to/camera_1.png
```

The 2D views show detected tag boundaries in cyan and reprojected board
boundaries in green. The 3D view shows the board, camera poses, and image
planes.

## Input assumptions

- Every image observes the same stationary board.
- The image order defines the camera indices.
- Camera 0 is the reference for relative poses and baselines.
- The board dimensions and tag layout match the constants in
  `src/main.rs`.
- Each camera starts with an assumed 24 mm full-frame-equivalent focal length.
- Intrinsics are estimated from the board because calibrated intrinsics are not
  supplied.

## Implementation

The example uses the existing `kornia-apriltag` decoder and the
`BoardGeometry`, `estimate_focal`, and `calibrate_board` APIs from
`kornia-calib`. It does not implement a separate detector or calibration
solver.

## Test

```bash
pixi run cargo test -p apriltag_board_multicam
```
