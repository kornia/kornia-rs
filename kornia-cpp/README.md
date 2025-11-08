# kornia-cpp

Thin C++ wrapper for the Kornia Rust library using [CXX](https://cxx.rs/).

## Overview

Lightweight, zero-overhead C++ interface to kornia-rs. All functions are inline wrappers with no data copies or performance penalties.

## Features

- ✅ Zero-copy via `rust::Slice` (direct pointer into Rust memory)
- ✅ Header-only C++ wrapper
- ✅ Exception-based error handling
- ✅ CMake integration with `find_package` support
- ✅ OpenCV-style image types (`ImageU8C3`, `ImageF32C1`, etc.)
- ✅ Thread-safe concurrent reads (standard C++ object lifetime rules apply)

## Quick Start

```cpp
#include <kornia.hpp>
#include <iostream>

int main() {
    auto image = kornia::io::read_jpeg_rgb8("image.jpg");
    std::cout << image.width() << "x" << image.height() << std::endl;
    
    auto data = image.data();  // rust::Slice<const uint8_t>
    std::cout << "First pixel R: " << (int)data[0] << std::endl;
    
    return 0;
}
```

## Building

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build --prefix ~/local  # optional
```

### Using Just

> **Note:** If you are running commands from the `kornia-cpp` directory, use the local commands below. If running from the project root, use the `cpp-` prefixed commands (e.g., `just cpp-test`), which delegate to the local justfile.

```bash
just test            # Build and run tests
just build-examples  # Build examples
just clean           # Clean build
```

## API

### Image Types

```cpp
// OpenCV-style naming: ImageU8C3 = Unsigned 8-bit, 3 Channels
ImageU8C1   // Grayscale u8 (1 channel)
ImageU8C3   // RGB u8 (3 channels)
ImageU8C4   // RGBA u8 (4 channels)
ImageF32C1  // Grayscale f32 (1 channel)
ImageF32C3  // RGB f32 (3 channels)
ImageF32C4  // RGBA f32 (4 channels)
```

### Image Methods

```cpp
auto image = kornia::io::read_jpeg_rgb8("image.jpg");

size_t w = image.width();
size_t h = image.height();
size_t c = image.channels();
ImageSize size = image.size();
rust::Slice<const uint8_t> data = image.data();  // Zero-copy!
```

### I/O Functions

```cpp
namespace kornia::io {
    ImageU8C3 read_jpeg_rgb8(const std::string& file_path);
    ImageU8C1 read_jpeg_mono8(const std::string& file_path);
}
```

## Integration

### Option 1: CMake Subdirectory

```cmake
add_subdirectory(kornia-cpp)
target_link_libraries(myapp PRIVATE kornia_cpp)
```

### Option 2: find_package

```cmake
find_package(kornia REQUIRED PATHS ~/local)
target_link_libraries(myapp PRIVATE kornia::kornia_cpp)
```

## Testing

```bash
just cpp-test           # Run all tests
just cpp-test-verbose   # Verbose output
just cpp-test-filter "JPEG"  # Run specific tests
```

## Examples

See `examples/read_jpeg/` for a complete example.

## License

Apache-2.0
