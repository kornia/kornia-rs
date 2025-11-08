# Read JPEG Example

A simple example demonstrating how to use kornia-cpp to read JPEG images.

## Building

### Option 1: As Part of kornia-cpp Build

This happens automatically when building kornia-cpp:

```bash
cd kornia-cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
./examples/read_jpeg/read_jpeg_example path/to/image.jpg
```

### Option 2: Standalone Build

After installing kornia-cpp, you can build this example independently:

```bash
# Install kornia-cpp first
cd kornia-cpp/build
cmake --install . --prefix /usr/local

# Then build this example standalone
cd ../examples/read_jpeg
mkdir build && cd build
cmake .. -DSTANDALONE_BUILD=ON
cmake --build .
./read_jpeg_example path/to/image.jpg
```

### Option 3: Copy as Template

Copy this directory as a template for your own project:

```bash
cp -r kornia-cpp/examples/read_jpeg my_project
cd my_project

# Edit CMakeLists.txt to set STANDALONE_BUILD ON by default
sed -i 's/option(STANDALONE_BUILD "Build as standalone example" OFF)/option(STANDALONE_BUILD "Build as standalone example" ON)/' CMakeLists.txt

# Build
mkdir build && cd build
cmake ..
cmake --build .
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

## Code Structure

The example demonstrates:

1. **Include kornia headers**
   ```cpp
   #include <kornia/kornia.hpp>
   ```

2. **Load JPEG image**
   ```cpp
   kornia::Image image = kornia::io::read_jpeg_rgb(file_path);
   ```

3. **Access image properties**
   ```cpp
   image.width, image.height, image.channels
   ```

4. **Access pixel data**
   ```cpp
   image.data[idx]  // Direct access to rust::Vec
   ```

5. **Error handling**
   ```cpp
   try { ... } catch (const std::exception& e) { ... }
   ```

## Minimal Integration Example

Here's the absolute minimum to integrate kornia-cpp:

**CMakeLists.txt:**
```cmake
cmake_minimum_required(VERSION 3.15)
project(my_app)

find_package(kornia REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE kornia)
```

**main.cpp:**
```cpp
#include <kornia/kornia.hpp>
#include <iostream>

int main() {
    auto img = kornia::io::read_jpeg_rgb("image.jpg");
    std::cout << img.width << "x" << img.height << std::endl;
}
```

That's it! No manual include paths or library linking needed.

