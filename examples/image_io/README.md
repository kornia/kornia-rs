# Image I/O Example

This example demonstrates how to use the image I/O functionality in kornia-rs to read images in various formats and write them in specific formats with different channel configurations.

## Features

- Read images in any format supported by the library
- Write images in PNG format with different channel configurations:
  - Grayscale (1 channel)
  - RGB (3 channels)
  - RGBA (4 channels)
- Write images in JPEG format:
  - Grayscale (1 channel)
  - RGB (3 channels)
- Decode images directly from raw bytes:
  - Any image format (using format hint or automatic detection)
  - Specialized JPEG decoding using libjpeg-turbo
  - RGB and grayscale output formats

## Usage

### Main I/O Example

Run the example with:

```bash
cargo run -- --input-path <INPUT_IMAGE_PATH> --output-path <OUTPUT_IMAGE_PATH> --format <FORMAT> --channels <CHANNELS>
```

Where:

- `<INPUT_IMAGE_PATH>` is the path to the input image.
- `<OUTPUT_IMAGE_PATH>` is the path where the output image will be saved.
- `<FORMAT>` is the output format, can be "png" or "jpeg" (default: "png").
- `<CHANNELS>` is the number of channels for the output image:
  - 1 for grayscale
  - 3 for RGB (default)
  - 4 for RGBA (PNG only)

### Decode from Bytes Example (Rust)

This example demonstrates decoding images directly from raw bytes:

```bash
cargo run --bin decode_from_bytes -- --image-path <INPUT_IMAGE_PATH>
```

Where:

- `<INPUT_IMAGE_PATH>` is the path to the input image.

### Decode from Bytes Example (Python)

```bash
python src/python/decode_from_bytes.py --image-path <INPUT_IMAGE_PATH>
```

Where:

- `<INPUT_IMAGE_PATH>` is the path to the input image.

## Examples

### Convert an image to grayscale PNG:

```bash
cargo run -- --input-path ../../tests/data/dog.jpeg --output-path dog_gray.png --format png --channels 1
```

### Convert an image to RGB PNG:

```bash
cargo run -- --input-path ../../tests/data/dog.jpeg --output-path dog_rgb.png --format png --channels 3
```

### Convert an image to RGBA PNG:

```bash
cargo run -- --input-path ../../tests/data/dog.jpeg --output-path dog_rgba.png --format png --channels 4
```

### Convert an image to grayscale JPEG:

```bash
cargo run -- --input-path ../../tests/data/dog.jpeg --output-path dog_gray.jpeg --format jpeg --channels 1
```

### Convert an image to RGB JPEG:

```bash
cargo run -- --input-path ../../tests/data/dog.jpeg --output-path dog_rgb.jpeg --format jpeg --channels 3
```

### Decode an image from raw bytes (Rust):

```bash
cargo run --bin decode_from_bytes -- --image-path ../../tests/data/dog.jpeg
```

### Decode an image from raw bytes (Python):

```bash
python src/python/decode_from_bytes.py --image-path ../../tests/data/dog.jpeg
```

## Using in Python

```python
import kornia_rs as K
import numpy as np

# Load an image from a file
img = K.read_image_any("path/to/image.jpg")

# Alternatively, read the image bytes and decode
with open("path/to/image.jpg", "rb") as f:
    image_data = f.read()

# Decode with automatic format detection
img_rgb = K.decode_image_bytes(image_data)
img_gray = K.decode_image_bytes_gray(image_data)

# Decode with format hint
img_png = K.decode_image_bytes(image_data, "png")

# For JPEG images, use specialized JPEG decoder for better performance
img_jpeg = K.decode_jpeg_bytes(image_data)
img_jpeg_gray = K.decode_jpeg_bytes_gray(image_data)
```
