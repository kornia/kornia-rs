#pragma once

/// @file kornia.hpp
/// @brief Main header for Kornia C++ library - thin wrapper around kornia-rs
///
/// This is a lightweight, zero-overhead C++ interface over the Rust implementation.
/// All functions are inline wrappers - no data copies, no performance penalty.
///
/// @example
/// ```cpp
/// #include <kornia/kornia.hpp>
///
/// auto image = kornia::io::read_jpeg_rgb("image.jpg");
/// std::cout << image.width << "x" << image.height << std::endl;
/// ```

// Version (auto-generated from Cargo.toml)
#include "kornia/version.hpp"

// Core types
#include "kornia/image.hpp"

// I/O functionality
#include "kornia/io.hpp"

namespace kornia {

/// @brief Get library version string (synced with Cargo.toml)
/// @return Version string in format "major.minor.patch"
inline const char* version() {
    return detail::get_version();
}

} // namespace kornia
