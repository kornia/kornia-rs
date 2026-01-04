An example showing how to stream camera feed to Foxglove WebSocket server using the `kornia_io` module with V4L. This example captures frames from a webcam and sends them as compressed images to Foxglove for visualization.

NOTE: This example is only supported on Linux.

```bash
Usage: foxglove [OPTIONS]

Options:
  -c, --camera-id <CAMERA_ID>  the camera id to use [default: 0]
  -f, --fps <FPS>              the frames per second to record [default: 30]
  -h, --help                   Print help
```

Example:

```bash
cargo run --bin foxglove --release -- --camera-id 0 --fps 30
```
