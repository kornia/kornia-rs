Example showing how to write a video using different background tasks.

NOTE: This example requires the gstremer backend to be enabled. To enable the gstreamer backend, use the `gstreamer` feature flag when building the `kornia` crate and its dependencies.

```bash
Usage: video_write_tasks [OPTIONS] --output <OUTPUT>

Options:
  -o, --output <OUTPUT>
  -c, --camera-id <CAMERA_ID>  [default: 0]
  -f, --fps <FPS>              [default: 30]
  -d, --duration <DURATION>
  -h, --help                   Print help              Print help
```

Example:

```bash
cargo run --bin video_write_tasks --release -- --output output.mp4
```
