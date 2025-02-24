An example showing how to use the webcam with the `kornia_io` module with the ability to cancel the feed after a certain amount of time. This example will display the webcam feed in a [`rerun`](https://github.com/rerun-io/rerun) window.

NOTE: This example requires the gstremer backend to be enabled. To enable the gstreamer backend, use the `gstreamer` feature flag when building the `kornia` crate and its dependencies.

```bash
Usage: webcam [OPTIONS]

Options:
  -c, --camera-id <CAMERA_ID>  [default: 0]
  -f, --fps <FPS>              [default: 30]
  -d, --duration <DURATION>
  -h, --help                   Print help
```

Example:

```bash
cargo run --bin webcam --release -- --camera-id 0 --duration 5 --fps 30
```

![Screenshot from 2024-08-28 18-33-56](https://github.com/user-attachments/assets/783619e4-4867-48bc-b7d2-d32a133e4f5a)
