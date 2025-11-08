# kornia-cpp

**Thin** C++ wrapper for the Kornia Rust library using [CXX](https://cxx.rs/).

## Overview

This is a **lightweight, zero-overhead** C++ interface to kornia-rs. All functions are inline wrappers that directly expose the Rust implementation with no data copies or performance penalties. It's essentially just namespace organization and exception handling over the raw FFI.

## Features

- ✅ **Zero-copy** - Returns Rust types directly (rust::Vec, rust::String)
- ✅ **Header-only** C++ wrapper
- ✅ Exception-based error handling
- ✅ Namespace organization (`kornia::io`)
- ✅ CMake integration
- ✅ Minimal overhead (inline functions)

## Building

### Using CMake (Recommended)

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

### Using Cargo (for development)

```bash
cargo build --release
```

## API

### ImageResult (from Rust)

```cpp
// Directly from Rust - zero copy!
struct ImageResult {
    rust::Vec<uint8_t> data;    // Raw pixel data (row-major, interleaved)
    size_t width;                // Image width in pixels
    size_t height;               // Image height in pixels
    size_t channels;             // Number of channels (1 or 3)
    bool success;                // Always true (throws on error)
    rust::String error_message;  // Empty (throws on error)
};
```

### Image Type

```cpp
namespace kornia {

// Zero-copy alias to Rust ImageResult
using Image = ::ImageResult;

}
```

### I/O Functions

Available in `kornia/io/jpeg.hpp`:

```cpp
namespace kornia::io {

// Read RGB JPEG (throws std::runtime_error on failure)
Image read_jpeg_rgb(const std::string& file_path);

// Read grayscale JPEG (throws std::runtime_error on failure)  
Image read_jpeg_gray(const std::string& file_path);

}
```

## Quick Start Example

```cpp
#include <kornia/kornia.hpp>
#include <iostream>

int main() {
    try {
        // Read RGB image - zero copy, returns Rust types directly
        kornia::Image image = kornia::io::read_jpeg_rgb("path/to/image.jpg");
        
        std::cout << "Loaded: " << image.width << "x" << image.height 
                  << " (" << image.channels << " channels)" << std::endl;
        
        // Access pixel data directly from Rust Vec
        // Format: row-major, interleaved channels
        size_t idx = (row * image.width + col) * image.channels + channel;
        uint8_t pixel_value = image.data[idx];
        
        // Or iterate
        for (size_t i = 0; i < image.data.size(); ++i) {
            uint8_t value = image.data[i];
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
```

## Modular Headers

You can also include specific functionality:

```cpp
#include <kornia/image.hpp>      // Just the Image type
#include <kornia/io/jpeg.hpp>    // Just JPEG I/O
#include <kornia/io.hpp>         // All I/O functionality
#include <kornia/kornia.hpp>     // Everything (recommended)
```

## Version Management

The C++ library version is **automatically synchronized** with `Cargo.toml`:

```cpp
std::cout << kornia::version() << std::endl;  // "0.1.0"
```

Update version in one place (`Cargo.toml`) and both Rust and C++ stay in sync!
See [VERSION.md](VERSION.md) for details.

See `examples/read_jpeg_example.cpp` for a complete working example.

## Testing

### Rust Tests

```bash
cargo test
```

### C++ Tests (with CMake)

```bash
cd build
ctest
```

All tests include:
- ✅ Reading valid JPEG images
- ✅ Error handling for invalid paths
- ✅ Pixel access and bounds checking
- ✅ CXX bridge compilation verification

## Architecture

The binding uses CXX to create a safe bridge between Rust and C++:

1. **Rust Implementation**: Uses `read_image_jpeg_mono8` and `read_image_jpeg_rgb8` from `kornia-io`
2. **FFI Bridge**: `#[cxx::bridge]` macro defines the C++-compatible interface
3. **Wrapper Functions**: Convert Rust `Result<Image>` to C++-friendly `ImageResult` struct
4. **Code Generation**: CXX automatically generates C++ headers and FFI glue code
5. **C++ Usage**: Include generated headers and link against the Rust library

## Integration into C++ Projects

### Method 1: Using CMake (Recommended)

Add Kornia as a subdirectory:

```cmake
add_subdirectory(path/to/kornia-cpp)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE kornia)
```

### Method 2: Find Package

After installing Kornia:

```cmake
find_package(kornia REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE kornia::kornia)
```

### Method 3: Manual Integration

1. Build the library: `cargo build --release`
2. Add include directory: `include/`
3. Link: `target/release/libkornia_cpp.a`
4. Link system libraries: `pthread`, `dl`, `m` (Linux)

```cmake
add_executable(my_app main.cpp)
target_include_directories(my_app PRIVATE 
    path/to/kornia-cpp/include
    path/to/kornia-cpp/target/cxxbridge
)
target_link_libraries(my_app PRIVATE 
    path/to/kornia-cpp/target/release/libkornia_cpp.a
    pthread dl m
)
```

## License

Apache-2.0

