An example showing how to write a video file using the `kornia::io` module along with the webcam capture example. Visualizes the webcam feed in a [`rerun`](https://github.com/rerun-io/rerun) window.

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

![Screenshot from 2024-08-28 18-33-56](https://github.com/user-attachments/assets/783619e4-4867-48bc-b7d2-d32a133e4f5a)
