#pragma once

#include "kornia/image.hpp"
#include <stdexcept>
#include <string>

namespace kornia {
namespace io {

/// @brief Read an RGB JPEG image from file
///
/// @param file_path Path to JPEG file
/// @return Image containing RGB data (3 channels)
/// @throws std::runtime_error if file cannot be read or is invalid
///
/// @note This is a zero-copy operation. The returned Image contains
///       a rust::Vec that directly manages the pixel data.
inline Image read_jpeg_rgb(const std::string& file_path) {
    auto result = read_jpeg_rgb8(file_path);
    if (!result.success) {
        throw std::runtime_error(std::string(result.error_message));
    }
    return result;
}

/// @brief Read a grayscale JPEG image from file
///
/// @param file_path Path to JPEG file
/// @return Image containing grayscale data (1 channel)
/// @throws std::runtime_error if file cannot be read or is invalid
///
/// @note This is a zero-copy operation. The returned Image contains
///       a rust::Vec that directly manages the pixel data.
inline Image read_jpeg_gray(const std::string& file_path) {
    auto result = read_jpeg_mono8(file_path);
    if (!result.success) {
        throw std::runtime_error(std::string(result.error_message));
    }
    return result;
}

} // namespace io
} // namespace kornia
