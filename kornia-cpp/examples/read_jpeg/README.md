# Read JPEG Example

Simple example demonstrating JPEG image reading with kornia-cpp.

## Building

```bash
pixi run cpp-build-examples
./kornia-cpp/build/examples/read_jpeg/read_jpeg_example path/to/image.jpg
```

## Usage

```bash
./read_jpeg_example <path_to_jpeg_image>
```

Example output:
```
Kornia C++ Library v0.1.0
Reading JPEG image from: dog.jpeg

✓ Successfully loaded image!
  Dimensions: 258 x 195
  Channels: 3
  Data size: 150930 bytes

  First 10 bytes: 188 179 174 188 179 174 188 179 174 188
  Pixel (0,0) R channel: 188
```

## Code

The example demonstrates:

1. Loading a JPEG image
   ```cpp
   auto image = kornia::io::jpeg::read_image_jpeg_rgb8(file_path);
   ```

2. Accessing image properties
   ```cpp
   image.width(), image.height(), image.channels()
   ```

3. Zero-copy data access
   ```cpp
   auto data = image.data();  // rust::Slice<const uint8_t>
   uint8_t pixel = data[idx];
   ```

4. Error handling
   ```cpp
   try { ... } catch (const std::exception& e) { ... }
   ```
