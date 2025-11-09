#pragma once

#include "kornia/image.hpp"
#include <string>
#include <vector>

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

/// @brief Encode an RGB JPEG image to bytes (returns std::vector)
///
/// @param image RGB u8 image to encode
/// @param quality JPEG quality (0-100, where 100 is highest quality)
/// @return std::vector<uint8_t> containing JPEG-encoded bytes
/// @throws rust::Error if encoding fails
///
/// @note Copies data from rust::Vec to std::vector for C++ interop
inline std::vector<uint8_t> encode_jpeg_rgb8(const ImageU8C3& image, uint8_t quality) {
    auto bytes = ::encode_jpeg_rgb8(image.inner(), quality);
    return std::vector<uint8_t>(bytes.begin(), bytes.end());
}

/// @brief Decode JPEG bytes to RGB u8 image
///
/// @param jpeg_bytes JPEG-encoded bytes (borrowed via const reference)
/// @return ImageU8C3 containing decoded RGB data
/// @throws rust::Error if decoding fails or not RGB
///
/// @note Zero-copy: creates rust::Slice view of std::vector. Perfect for Unreal/OpenCV interop.
inline ImageU8C3 decode_jpeg_rgb8(const std::vector<uint8_t>& jpeg_bytes) {
    // Zero-copy: rust::Slice is just a view into the std::vector
    rust::Slice<const uint8_t> slice(jpeg_bytes.data(), jpeg_bytes.size());
    return ImageU8C3(::decode_jpeg_rgb8(slice));
}

} // namespace io
} // namespace kornia
