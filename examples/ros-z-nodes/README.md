# ROS-Z Nodes Example

A modular ROS2-style camera processing pipeline using [ros-z](https://github.com/ZettaScaleLabs/ros-z) (Zenoh-based middleware). This example demonstrates a multi-node architecture for camera capture, JPEG decoding, image statistics computation, and visualization via Foxglove.

NOTE: This example is only supported on Linux.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│ CameraNode  │────▶│ DecoderNode  │────▶│ ComputeNode  │
│  (V4L/JPEG) │     │ (JPEG→RGB)   │     │ (mean/std)   │
└─────────────┘     └──────────────┘     └──────────────┘
      │                   │                    │
      │                   │ (Zenoh SHM)        │
      ▼                   ▼                    ▼
camera/{name}/      camera/{name}/      camera/{name}/
  compressed          raw_shm              stats
      │                                       │
      │         ┌───────────────┐             │
      └────────▶│ FoxgloveNode  │◀────────────┘
                │  (WebSocket)  │
                └───────────────┘
                        │
                        ▼
                 Foxglove Studio
                  (port 8765)
```

## Nodes

| Node | Description |
|------|-------------|
| **CameraNode** | Captures JPEG frames from V4L camera and publishes `CompressedImage` |
| **DecoderNode** | Decodes JPEG to RGB and publishes via Zenoh Shared Memory |
| **ComputeNode** | Computes per-channel mean and standard deviation |
| **LoggerNode** | Logs image statistics to console |
| **FoxgloveNode** | Bridges messages to Foxglove WebSocket for visualization |

## Usage

```bash
Usage: ros-z-camera [OPTIONS]

Options:
  -n, --camera-name <CAMERA_NAME>  camera name for topics (e.g., "front", "back") [default: front]
  -d, --device-id <DEVICE_ID>      V4L device ID (e.g., 0 for /dev/video0) [default: 0]
  -f, --fps <FPS>                  frames per second [default: 30]
  -h, --help                       Print help
```

## Example

```bash
cargo run -p ros-z-nodes --bin ros-z-camera --release -- --camera-name front --device-id 0 --fps 30
```
<img width="1044" height="1271" alt="Screenshot from 2025-12-22 14-25-10" src="https://github.com/user-attachments/assets/4fdbe4d6-6396-4ddb-b9bc-331ca49e48ef" />

## Foxglove Visualization

1. Start the example (Foxglove WebSocket server runs on port 8765)
2. Open [Foxglove Studio](https://foxglove.dev/studio)
3. Connect via WebSocket: `ws://localhost:8765`
4. Add panels:
   - **Image** panel for `camera/front/compressed`
   - **Raw Messages** or **Plot** panel for `camera/front/stats`

## Topics

| Topic | Message Type | Description |
|-------|--------------|-------------|
| `camera/{name}/compressed` | `CompressedImage` | JPEG compressed frames |
| `camera/{name}/raw_shm` | `RawImage` | Decoded RGB frames (via Zenoh SHM) |
| `camera/{name}/stats` | `ImageStats` | Per-channel mean and std |

## Protobuf Messages

The example uses custom protobuf messages defined in `protos/`:

- `header.proto` - Common header with timestamps and sequence
- `camera.proto` - `CompressedImage` and `RawImage` messages
- `stats.proto` - `ImageStats` with per-channel statistics
