# Video Understanding Example

## Usage

```bash
Usage: smol_vlm2_video [OPTIONS]
  --prompt <PROMPT>              Custom prompt for the model (required)
  -i, --ip-address <IP_ADDRESS>  (required)
  -P, --port <PORT>              (required)
  --debug                        Useful debug info (optional)
  -h, --help                     Print help

Options for video file mode:
  --video-file <VIDEO_FILE>
  -f, --fps <FPS>                For customized sampling of the video [default: 30]

Options live webcam mode:
  -c, --camera-id <CAMERA_ID>    [default: 0]
  -f, --fps <FPS>                [default: 30]
  -p, --pixel-format <PIXEL_FORMAT> [default: MJPG]

```

## Examples

Examples:

Camera mode:
```bash
cargo run --bin smol_vlm2_video --release -- --camera-id 0 --fps 30 --ip-address 192.168.1.4 --port 9999 --prompt "Describe the scene." --debug
```

Video file mode:
```bash
cargo run --bin smol_vlm2_video --release --features cuda -- --video-file ./example_video.mp4 --ip-address <viewer's IP address> --port 9999 --prompt "What direction is Earth moving?"
```

![GIF](https://private-user-images.githubusercontent.com/39780709/485829152-384ae9d2-a78b-4748-a2b5-ef05ef006621.gif?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTg3NDE4ODEsIm5iZiI6MTc1ODc0MTU4MSwicGF0aCI6Ii8zOTc4MDcwOS80ODU4MjkxNTItMzg0YWU5ZDItYTc4Yi00NzQ4LWEyYjUtZWYwNWVmMDA2NjIxLmdpZj9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA5MjQlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwOTI0VDE5MTk0MVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTA2OTMxNDk4Y2VlZjA5YjZhZjE0ZmRmY2ZkMTk0ZTQ4YmE2ZTRhYTAxMzM4MTg5YjA5ODkwYzVkYjY4YTBmZDUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.4AEO9xXsR87z5bDCEh_qpGUZDBqNwOVJUekVrWTJ3v4)


## Running Rerun

```sh
# to find your client IP address
hostname -I
# to reset logs
rerun reset
# to start your rerun on the client
rerun --port 9999
```

## (Specific Usage) Connecting Webcam
If you want to run this example on a headless server, you first need a source of webcam (i.e., laptop camera). To connect your external (over the network) webcam to your headless server, we will need `usbip`. We will refer the laptop as the client and the headless server as the server.

```bash
# ------ Run on server + client

# feel free to put it in modprobe.d to automatically load it after every startup
sudo modprobe usbip_core; sudo modprobe usbip_host

# start USB/IP daemon
sudo usbipd -D

# ------ Run only on client

# find device with
lsusb
usbip list -l



# binding device
sudo usbip bind -b <device/bus-id>
# ------ Run only on server
sudo usbip attach -r <ip_addr_of_local> -b <device/bus-id>

# when restarting
sudo usbip unbind -b <device/bus-id>
sudo usbip bind -b <device/bus-id>
sudo usbip attach -r <ip_addr_of_local> -b <device/bus-id>


# to get your rerun viewer's device IP address, which is likely to be different then the IP address of your server.
ip addr show
```
