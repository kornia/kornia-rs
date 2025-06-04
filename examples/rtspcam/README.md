An example showing how to use the RTSP camera with the `kornia_io` module with the ability to cancel the feed after a certain amount of time. This example will display the webcam feed in a [`rerun`](https://github.com/rerun-io/rerun) window.

NOTE: This example requires the gstremer backend to be enabled. To enable the gstreamer backend, use the `gstreamer` feature flag when building the `kornia` crate and its dependencies.

```bash
Usage: rtspcam [-r <rtsp-url>] [-u <username>] [-p <password>] [--camera-ip <camera-ip>] [--camera-port <camera-port>] [-s <stream>]

RTSP Camera Capture and stream to ReRun

Options:
  -r, --rtsp-url    the full RTSP URL (e.g., rtsp://user:pass@ip:port/stream)
  -u, --username    the username to access the camera
  -p, --password    the password to access the camera
  --camera-ip       the camera ip address
  --camera-port     the camera port
  -s, --stream      the camera stream
  --help, help      display usage information
```

NOTE: You can also pass the rtsp url as an argument to the program. For example, you can use the following testing url: `rtsp://rtspstream:4j_U0tJ6fGtiaKREnuVnH@zephyr.rtsp.stream/movie`

Or check out the [RTSP Streams](https://www.rtsp.stream/) website for more streams to try more streams.

![Screenshot from 2024-08-28 18-30-11](https://github.com/user-attachments/assets/2a9a80f4-4933-4614-930a-061ec2463227)
