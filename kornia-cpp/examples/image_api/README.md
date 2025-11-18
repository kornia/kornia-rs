# Image API Example

Comprehensive example demonstrating the kornia-cpp Image API, including:
- Creating images from fill values or data
- Zero-copy data access
- Error handling
- Different image types (U8, F32) and channels (1, 3, 4)

## Building

```bash
cd kornia-cpp
just build-examples
./build/examples/image_api/image_api_example
```

Or build standalone:

```bash
cd examples/image_api
mkdir build && cd build
cmake ..
make
./image_api_example
```

## Example Output

```
╔════════════════════════════════════════════╗
║   Kornia C++ Image API Examples           ║
╚════════════════════════════════════════════╝

Example 1: Creating images from fill values
================================================
✓ Created 640x480x3 image
  Fill value: 128
  Total bytes: 921600

Example 2: Creating images from data vectors
================================================
✓ Created 10x10 grayscale image from data
  First pixel: 0
  Last pixel: 255

Example 3: Error handling (wrong data size)
================================================
✓ Caught expected error:
  Data length (100) does not match the image size (300)
...
```

## API Features Demonstrated

### 1. Creating Images from Fill Values

```cpp
// Create 640x480 RGB image filled with value 128
auto img = kornia::ImageU8C3(640, 480, 128);
```

### 2. Creating Images from Data Vectors

```cpp
std::vector<uint8_t> data(100);
// ... fill data ...
auto img = kornia::ImageU8C1(10, 10, data);
```

### 3. Zero-Copy Data Access

```cpp
auto img = kornia::ImageU8C3(100, 100, 42);
auto data = img.data();  // rust::Slice<const uint8_t> - zero copy!
uint8_t pixel = data[0];
```

### 4. Owned Copy of Data

```cpp
auto img = kornia::ImageF32C3(100, 100, 0.5f);
auto owned_copy = img.to_vec();  // std::vector<float>
// Can modify independently of original
```

### 5. Error Handling

```cpp
try {
    std::vector<uint8_t> wrong_size(100);  // Need 300
    auto img = kornia::ImageU8C3(10, 10, wrong_size);
} catch (const std::exception& e) {
    // Catches: "Data length (100) does not match the image size (300)"
    std::cerr << "Error: " << e.what() << "\n";
}
```

### 6. Image Properties

```cpp
auto img = kornia::ImageU8C3(640, 480, 0);
size_t w = img.width();       // 640
size_t h = img.height();      // 480
size_t c = img.channels();    // 3
auto size = img.size();       // ImageSize{width: 640, height: 480}
```

## Supported Image Types

| Type | Description | C++ Type | Channels |
|------|-------------|----------|----------|
| `ImageU8C1` | Grayscale 8-bit | `uint8_t` | 1 |
| `ImageU8C3` | RGB 8-bit | `uint8_t` | 3 |
| `ImageU8C4` | RGBA 8-bit | `uint8_t` | 4 |
| `ImageF32C1` | Grayscale float | `float` | 1 |
| `ImageF32C3` | RGB float | `float` | 3 |
| `ImageF32C4` | RGBA float | `float` | 4 |

## Memory Management

- **Ownership**: Images are owned by Rust via `rust::Box`
- **Zero-copy**: `data()` returns `rust::Slice` pointing to Rust memory
- **RAII**: Automatic cleanup when C++ object goes out of scope
- **Thread-safe**: Safe for concurrent reads
- **Move semantics**: Supports C++ move operations

## Data Layout

Images use **row-major interleaved** format:

```
For a 3x2 RGB image:
[R G B] [R G B] [R G B]  <- Row 0
[R G B] [R G B] [R G B]  <- Row 1

Pixel at (row=1, col=2): data[(1 * width + 2) * channels]
```

## Error Types

All constructor errors are returned as `std::exception` with descriptive messages:

- **Data length mismatch**: "Data length (X) does not match the image size (Y)"
- **Allocation failure**: Details from underlying tensor allocation
- **Invalid operations**: Contextual error messages

## Performance

- Zero-copy data access: O(1)
- Image creation: O(n) where n = width × height × channels
- Move operations: O(1)
- Copy operations: O(n) via `to_vec()`

