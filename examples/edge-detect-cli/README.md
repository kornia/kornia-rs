# Edge Detection CLI Tool

A command-line utility for applying Sobel edge detection to images. Designed for robotics applications like line-following, obstacle detection, and general computer vision tasks.

## Features

- 🚀 Fast Sobel edge detection (10-15ms for 1080p)
- 📊 Real-time performance statistics
- 🎯 Configurable kernel sizes (1, 3, 5, 7)
- 🔧 Optional edge magnitude thresholding
- 📁 Support for common image formats (PNG, JPEG, etc.)

## Build

From the kornia-rs root directory:

```bash
cargo build --example edge-detect-cli
```

## Usage

### Basic Usage

```bash
cargo run --example edge-detect-cli -- \
  --input tests/data/robotics_image.png \
  --output target/edges.png \
  --show-stats
```

### Advanced Options

```bash
cargo run --example edge-detect-cli -- \
  --input input.png \
  --output edges.png \
  --kernel-size 3 \
  --threshold 50.0 \
  --show-stats
```

### Command-Line Arguments

- `--input` (required): Path to input image
- `--output` (required): Path to output edge map
- `--kernel-size` (optional, default=3): Sobel kernel size (1, 3, 5, or 7)
- `--threshold` (optional): Edge magnitude threshold for binary output (0-255)
- `--show-stats` (optional, default=true): Display performance statistics

## Use Cases

### 1. Robotics - Line Following

```bash
# detect edges for line-following robot
cargo run --example edge-detect-cli -- \
  --input camera_frame.png \
  --output edge_map.png \
  --kernel-size 3 \
  --threshold 100
```

### 2. Medical Imaging - Feature Detection

```bash
# detect boundaries in medical scan
cargo run --example edge-detect-cli -- \
  --input ct_scan.png \
  --output edges.png \
  --kernel-size 5
```

### 3. Quality Control - Defect Detection

```bash
# detect product defects
cargo run --example edge-detect-cli -- \
  --input product_image.png \
  --output defects.png \
  --kernel-size 3
```

## Output Statistics

Example output:

```
🚀 Starting edge detection pipeline...
📁 Input: camera_frame.png
✅ Image loaded: 1920x1080 pixels in 2.15ms
✅ Preprocessing complete in 4.32ms
✅ Sobel edge detection complete in 8.75ms
✅ Post-processing complete in 1.25ms
✅ Output saved: edge_map.png in 3.20ms

📊 Statistics:
─────────────────────────────────────────
  Input size: 1920x1080 (2M pixels)
  Kernel size: 3x3
─────────────────────────────────────────
  Total processing time: 19.67ms
    ├─ Load: 2.15ms
    ├─ Preprocess: 4.32ms
    ├─ Sobel: 8.75ms
    ├─ Post-process: 1.25ms
    └─ Save: 3.20ms
─────────────────────────────────────────
  Performance:
    ├─ Edge pixels: 45,230 (2.18%)
    ├─ Pixels/second: 219M
    └─ FPS (at 1080p): 51.1
─────────────────────────────────────────
```

## Performance Targets

- **1080p @ 3x3 kernel**: ~10-12ms (80-100 FPS equivalent)
- **720p @ 3x3 kernel**: ~5-6ms (150-166 FPS equivalent)
- **4K @ 3x3 kernel**: ~40-50ms (20-25 FPS equivalent)

## Algorithm

The tool applies the **Sobel operator**, which computes image gradients using:

```
Gx = [-1  0  1]    (vertical edges)
     [-2  0  2]
     [-1  0  1]

Gy = [-1 -2 -1]    (horizontal edges)
     [ 0  0  0]
     [ 1  2  1]

Magnitude = sqrt(Gx² + Gy²)
```

Larger kernel sizes produce smoother, less noisy edges but are slightly slower.
Smaller kernel sizes are faster but more sensitive to noise.

## Example Results

### Input: Robotics Line

```
Original:             After Edge Detection:
███████████████      ░░░░░░░░░░░░░░░░░░
█████████  ████  →   ░░░░▓░░░▓░░░░░░░░░
███  █████████       ░░░░░░░░░░░░░░░░░
█████████ ████       ░░░░░░░░░░░░░░░░░░
███████████████      ░░░░░░░░░░░░░░░░░
```

## Integration with Robotics Systems

This tool can be used as a preprocessing step for:

- **Line tracking**: Convert camera frame → edges → contour detection → motor control
- **Obstacle detection**: Detect sudden edge discontinuities → trigger avoidance
- **Lane detection**: Multiple passes at different thresholds
- **Pattern matching**: Match detected edges against templates

## Related Documentation

- [Kornia Image Processing Docs](https://docs.rs/kornia-imgproc/)
- [Sobel Edge Detection](https://en.wikipedia.org/wiki/Sobel_operator)
- [Kornia GitHub Repository](https://github.com/kornia/kornia)

## License

Apache License 2.0 (same as Kornia)
