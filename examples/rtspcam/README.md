An example showing how to use the RTSP camera with the `kornia_io` module with the ability to cancel the feed after a certain amount of time. This example will display the webcam feed in a [`rerun`](https://github.com/rerun-io/rerun) window.

NOTE: This example requires the gstremer backend to be enabled. To enable the gstreamer backend, use the `gstreamer` feature flag when building the `kornia` crate and its dependencies.

```bash
Usage: rtspcam [OPTIONS] --username <USERNAME> --password <PASSWORD> --camera-ip <CAMERA_IP> --camera-port <CAMERA_PORT> --stream <STREAM>

Options:
  -u, --username <USERNAME>
  -p, --password <PASSWORD>
      --camera-ip <CAMERA_IP>
      --camera-port <CAMERA_PORT>
  -s, --stream <STREAM>
  -d, --duration <DURATION>
  -h, --help                       Print help
```

![Screenshot from 2024-08-28 18-30-11](https://github.com/user-attachments/assets/2a9a80f4-4933-4614-930a-061ec2463227)
