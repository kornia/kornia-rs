#pragma once

#include "kornia/image.hpp"
#include <string>

namespace kornia {
namespace io {

/// @brief Read an RGB JPEG image from file (u8, 3 channels)
///
/// Uses libjpeg-turbo for high-performance JPEG decoding.
/// Image data is owned by Rust and accessed via zero-copy rust::Slice.
///
/// @param file_path Path to JPEG file (absolute or relative)
/// @return Image<uint8_t, 3> containing RGB data in row-major, interleaved format
/// @throws rust::Error if:
///   - File does not exist or cannot be opened
///   - File is not a valid JPEG
///   - JPEG decode fails
///   - Out of memory
///
/// Thread safety: Safe to call concurrently from multiple threads.
/// Memory: Image memory is managed by Rust (automatic cleanup).
/// Performance: Zero-copy - no data copies on FFI boundary.
///
/// @code{.cpp}
/// auto image = kornia::io::read_jpeg_rgb8("photo.jpg");
/// std::cout << image.width() << "x" << image.height() << "\n";
/// auto data = image.data();  // rust::Slice<const uint8_t>
/// uint8_t r = data[0];  // First pixel, R channel
/// @endcode
inline image::Image<uint8_t, 3> read_jpeg_rgb8(const std::string& file_path) {
    return image::Image<uint8_t, 3>(::read_jpeg_rgb8(file_path));
}

/// @brief Read a grayscale JPEG image from file (u8, 1 channel)
///
/// Uses libjpeg-turbo for high-performance JPEG decoding.
/// If source is color, it will be converted to grayscale.
/// Image data is owned by Rust and accessed via zero-copy rust::Slice.
///
/// @param file_path Path to JPEG file (absolute or relative)
/// @return Image<uint8_t, 1> containing grayscale data in row-major format
/// @throws rust::Error if:
///   - File does not exist or cannot be opened
///   - File is not a valid JPEG
///   - JPEG decode fails
///   - Out of memory
///
/// Thread safety: Safe to call concurrently from multiple threads.
/// Memory: Image memory is managed by Rust (automatic cleanup).
/// Performance: Zero-copy - no data copies on FFI boundary.
///
/// @note Known issue: May fail on some color JPEGs due to data length mismatch.
///       This is tracked in the test suite with [!mayfail] tag.
///
/// @code{.cpp}
/// auto image = kornia::io::read_jpeg_mono8("photo.jpg");
/// std::cout << image.width() << "x" << image.height() << "\n";
/// auto data = image.data();  // rust::Slice<const uint8_t>
/// uint8_t gray = data[0];  // First pixel
/// @endcode
inline image::Image<uint8_t, 1> read_jpeg_mono8(const std::string& file_path) {
    return image::Image<uint8_t, 1>(::read_jpeg_mono8(file_path));
}

} // namespace io
} // namespace kornia
