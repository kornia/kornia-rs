An example showing how to use V4L to drive a webcam with the `kornia_io` module with the ability to cancel the feed after a certain amount of time. This example will display the webcam feed in a [`rerun`](https://github.com/rerun-io/rerun) window.

NOTE: This example is only supported on Linux.

```bash
Usage: v4l [OPTIONS]

Options:
  -c, --camera-id <CAMERA_ID>  [default: 0]
  -f, --fps <FPS>              [default: 30]
  -d, --duration <DURATION>
  -h, --help                   Print help
```

Example:

```bash
cargo run --bin smol_vlm_video --release -- --camera-id 0 --fps 30
```

![Screenshot from 2024-08-28 18-33-56](https://github.com/user-attachments/assets/783619e4-4867-48bc-b7d2-d32a133e4f5a)




Notes on connecting via SSH from local (i.e., laptop) to remote (i.e., workstation) on Linux:
```bash
# feel free to put it in modprobe.d to automatically load it after every startup
sudo modprobe usbip_core; sudo modprobe usbip_host   ### ON REMOTE + LOCAL

# start USB/IP daemon
sudo usbipd -D   ### ON REMOTE + LOCAL

# find device with
lsusb
usbip list -l

# binding device
sudo usbip bind -b <device/bus-id>
sudo usbip attach -r <ip_addr_of_local> -b <device/bus-id>  ### ON REMOTE


# when restarting
sudo usbip unbind -b <device/bus-id>
sudo usbip bind -b <device/bus-id>
sudo usbip attach -r <ip_addr_of_local> -b <device/bus-id>  ### ON REMOTE

```

