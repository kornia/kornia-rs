An example showing how to write a video file using the `kornia_io` module along with the webcam capture example. Visualizes the webcam feed in a [`rerun`](https://github.com/rerun-io/rerun) window.

NOTE: This example requires the gstremer backend to be enabled. To enable the gstreamer backend, use the `gstreamer` feature flag when building the `kornia` crate and its dependencies.

```bash
Usage: video_write [OPTIONS] --output <OUTPUT>

Options:
  -o, --output <OUTPUT>
  -c, --camera-id <CAMERA_ID>  [default: 0]
  -f, --fps <FPS>              [default: 30]
  -h, --help                   Print help
```

Example:

```bash
cargo run --bin video_write --release -- --output ~/output.mp4
```
