#pragma once

#include "kornia/version.hpp"

// Core Image types
#include "kornia/image.hpp"

// Core I/O functionality
#include "kornia/io.hpp"

/// @file kornia.hpp
/// @brief Main header for kornia-cpp C++ bindings
///
/// This is a thin, zero-overhead wrapper around the Rust kornia library.
/// All functions are inline and delegate directly to Rust via CXX FFI bridge.
///
/// Features:
/// - Zero-copy data access via rust::Slice
/// - Header-only C++ wrapper (inline functions only)
/// - Exception-based error handling (Rust Result -> C++ exception)
/// - CMake integration with find_package support
/// - Thread-safe concurrent reads
///
/// @see https://github.com/kornia/kornia-rs

namespace kornia {

/// @brief Get kornia-cpp library version string
///
/// @return Version string in format "MAJOR.MINOR.PATCH" (e.g., "0.1.0")
///
/// The version is automatically generated from Cargo.toml during build.
inline const char* version() {
    return detail::get_version();
}

} // namespace kornia
