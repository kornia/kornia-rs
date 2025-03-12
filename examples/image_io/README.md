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

## Usage

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
