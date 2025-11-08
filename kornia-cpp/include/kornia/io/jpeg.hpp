#pragma once

#include "kornia/image.hpp"
#include <string>

namespace kornia {
namespace io {

/// @brief Read an RGB JPEG image from file (u8, 3 channels)
/// 
/// @param file_path Path to JPEG file
/// @return ImageU8C3 containing RGB data
/// @throws rust::Error if file cannot be read or is invalid
/// 
/// @note Memory is managed by Rust. Zero-copy data access.
inline ImageU8C3 read_jpeg_rgb8(const std::string& file_path) {
    return ImageU8C3(::read_jpeg_rgb8(file_path));  // Call FFI function from global namespace
}

/// @brief Read a grayscale JPEG image from file (u8, 1 channel)
/// 
/// @param file_path Path to JPEG file
/// @return ImageU8C1 containing grayscale data
/// @throws rust::Error if file cannot be read or is invalid
/// 
/// @note Memory is managed by Rust. Zero-copy data access.
inline ImageU8C1 read_jpeg_mono8(const std::string& file_path) {
    return ImageU8C1(::read_jpeg_mono8(file_path));  // Call FFI function from global namespace
}

} // namespace io
} // namespace kornia
