# kornia-cpp

Thin C++ wrapper for the Kornia Rust library using [CXX](https://cxx.rs/).

## Overview

Lightweight, zero-overhead C++ interface to kornia-rs. All functions are inline wrappers with no data copies or performance penalties.

## Features

- ✅ Zero-copy via `rust::Slice` (direct pointer into Rust memory)
- ✅ Header-only C++ wrapper (inline functions only)
- ✅ Exception-based error handling (Rust `Result` → C++ exceptions)
- ✅ CMake integration with `find_package` support
- ✅ Namespaced image types (`kornia::image::ImageU8C3`, `kornia::image::ImageF32C1`, etc.)
- ✅ Thread-safe concurrent reads (standard C++ object lifetime rules)
- ✅ Move semantics (no copy constructors for image types)
- ✅ Cross-platform (Linux, macOS, Windows)

## Quick Start

```cpp
#include <kornia.hpp>
#include <iostream>

int main() {
    // Read JPEG image with zero-copy data access
    auto image = kornia::io::read_jpeg_rgb8("image.jpg");
    
    std::cout << "Loaded: " << image.width() << "x" << image.height() 
              << " (" << image.channels() << " channels)\n";
    
    // Access raw pixel data (rust::Slice - no copy!)
    auto data = image.data();
    std::cout << "First pixel R: " << (int)data[0] << "\n";
    
    return 0;
}
```

## Building

### Using CMake directly

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
just format          # Format code with clang-format
just format-check    # Check format without modifying
just clean           # Clean build artifacts
```

## API

### Image Types

```cpp
// Image types in kornia::image namespace
kornia::image::ImageU8C1   // Grayscale u8 (1 channel)
kornia::image::ImageU8C3   // RGB u8 (3 channels)
kornia::image::ImageU8C4   // RGBA u8 (4 channels)
kornia::image::ImageF32C1  // Grayscale f32 (1 channel)
kornia::image::ImageF32C3  // RGB f32 (3 channels)
kornia::image::ImageF32C4  // RGBA f32 (4 channels)
```

### Image Methods

All methods are `const` and thread-safe for concurrent reads:

```cpp
auto image = kornia::io::read_jpeg_rgb8("image.jpg");

size_t w = image.width();          // Image width in pixels
size_t h = image.height();         // Image height in pixels
size_t c = image.channels();       // Number of channels
ImageSize size = image.size();     // {width, height} struct
rust::Slice<const uint8_t> data = image.data();  // Zero-copy data view!
```

**Data Layout:** Row-major, interleaved channels (e.g., `RGBRGBRGB...`)

**Pixel Access:**
```cpp
// Access pixel at (row, col), channel ch:
size_t idx = (row * width + col) * channels + ch;
uint8_t value = data[idx];
```

### I/O Functions

```cpp
namespace kornia::io {
    // Read JPEG as RGB u8 (uses libjpeg-turbo)
    image::Image<uint8_t, 3> read_jpeg_rgb8(const std::string& file_path);
    
    // Read JPEG as grayscale u8 (auto-converts if needed)
    image::Image<uint8_t, 1> read_jpeg_mono8(const std::string& file_path);
}
```

All I/O functions throw `rust::Error` on failure (file not found, invalid format, decode error, etc.).

## Integration

### Option 1: CMake Subdirectory

```cmake
add_subdirectory(kornia-cpp)
target_link_libraries(myapp PRIVATE kornia_cpp)
```

### Option 2: find_package (after installation)

```cmake
find_package(kornia REQUIRED PATHS ~/local)
target_link_libraries(myapp PRIVATE kornia::kornia_cpp)
```

## Thread Safety & Memory Model

### Thread Safety
- ✅ **Safe:** Concurrent reads from multiple threads
- ✅ **Safe:** Concurrent calls to I/O functions
- ⚠️ **Unsafe:** Accessing image after move or destruction
- ⚠️ **Unsafe:** Concurrent writes (not supported - images are immutable)

### Memory Management
- **Ownership:** Rust owns all image data via `rust::Box`
- **Views:** C++ accesses data via `rust::Slice` (non-owning pointer)
- **Lifetime:** Data valid as long as image wrapper exists
- **Cleanup:** Automatic via RAII (`rust::Box` destructor)

### Move Semantics
Image types support move but not copy:
```cpp
auto img1 = kornia::io::read_jpeg_rgb8("a.jpg");
auto img2 = std::move(img1);  // ✅ OK - move
// auto img3 = img2;           // ❌ Error - copy deleted
```

## Testing

```bash
just cpp-test           # Run all tests (from project root)
just test              # Run all tests (from kornia-cpp/)
just test-sanitizers   # Run with ASAN/UBSAN enabled
```

Tests use [Catch2](https://github.com/catchorg/Catch2) v3 framework.

## Examples

See `examples/read_jpeg/` for a complete example demonstrating:
- Reading JPEG images
- Accessing image properties
- Zero-copy data access
- Error handling with try/catch

## Requirements

- CMake 3.15 or later
- C++14 compatible compiler (GCC 5+, Clang 3.4+, MSVC 2015+)
- Rust toolchain (for building the Rust library)
- clang-format (optional, for code formatting)

## Formatting

The project uses `.clang-format` (LLVM style, 100 column limit):

```bash
just format        # Auto-format all C++ files
just format-check  # Check format without modifying
```

CI enforces formatting - make sure to run `just format` before committing.

## Design Principles

1. **Zero-copy:** No data copies on FFI boundary (use `rust::Slice`)
2. **Thin wrapper:** All C++ code is inline, delegates to Rust
3. **Type-safe:** CXX bridge ensures FFI safety at compile time
4. **Header-only:** C++ wrapper is header-only (except Rust lib)
5. **Const-correct:** All accessor methods are `const`
6. **Modern C++:** Use RAII, move semantics, `= delete` for safety

## Contributing

Please read [CONTRIBUTING.md](../CONTRIBUTING.md) for:
- Coding standards and style guidelines
- How to add new image types or I/O functions
- Testing requirements
- CI checks

## License

Apache-2.0
