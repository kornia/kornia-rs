# Kornia: kornia-cpp

[![License](https://img.shields.io/crates/l/kornia-cpp.svg)](https://github.com/kornia/kornia/blob/main/LICENSE)

> **C++ bindings for the Kornia Rust computer‑vision library.**

## 🚀 Overview
`kornia-cpp` provides a thin C++ wrapper around the core Kornia Rust crates, exposing the same high‑performance vision primitives to native C++ projects via the `cxx` bridge.

## 🔑 Key Features
- Zero‑copy interoperability between Rust and C++ data structures
- Access to all core Kornia functionality (tensor ops, image processing, 3‑D utilities)
- Simple CMake integration
- Cross‑platform support (Linux, macOS, Windows)

## 📦 Installation
```bash
# Clone the repository and build the C++ bindings
git clone https://github.com/kornia/kornia-rs.git
cd kornia-rs/kornia-cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

## 🛠️ Usage
```cpp
#include "kornia_cpp/kornia_cpp.h"
int main() {
    // Example: create a tensor and run a convolution
    auto tensor = kornia::tensor::from_vec({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    auto result = kornia::ops::conv2d(tensor, /*kernel*/ ...);
    return 0;
}
```

## 🤝 Contributing
Contributions are welcome! See the main repository's [Contributing Guidelines](https://github.com/kornia/kornia/blob/main/CONTRIBUTING.md).

## 📄 License
Apache-2.0
