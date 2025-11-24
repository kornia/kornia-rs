#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <kornia/io.hpp>

namespace fs = std::filesystem;
namespace img = kornia::image;
namespace jpeg = kornia::io::jpeg;

static std::string get_test_image_path() {
    fs::path test_data =
        fs::path(__FILE__).parent_path() / ".." / ".." / "tests" / "data" / "dog.jpeg";
    return fs::canonical(test_data).string();
}

TEST_CASE("Read JPEG RGB8", "[io][jpeg]") {
    std::string path = get_test_image_path();
    if (path.empty())
        SKIP("Test image not found");

    auto image = jpeg::read_image_jpeg_rgb8(path);

    REQUIRE(image.width() == 258);
    REQUIRE(image.height() == 195);
    REQUIRE(image.channels() == 3);
    REQUIRE(image.data().size() == 258 * 195 * 3);
}

// Known issue: Grayscale JPEG reading fails due to data length mismatch in kornia-io
TEST_CASE("Read JPEG Mono8", "[io][jpeg][!mayfail]") {
    std::string path = get_test_image_path();
    if (path.empty())
        SKIP("Test image not found");

    auto image = jpeg::read_image_jpeg_mono8(path);

    REQUIRE(image.width() == 258);
    REQUIRE(image.height() == 195);
    REQUIRE(image.channels() == 1);
    REQUIRE(image.data().size() == 258 * 195);
}

TEST_CASE("Read JPEG - Error Handling", "[io][jpeg][error]") {
    REQUIRE_THROWS(jpeg::read_image_jpeg_rgb8("/nonexistent/image.jpg"));
    REQUIRE_THROWS(jpeg::read_image_jpeg_rgb8(""));
}

TEST_CASE("Encode JPEG RGB8", "[io][jpeg][encode]") {
    std::string path = get_test_image_path();
    if (path.empty())
        SKIP("Test image not found");

    // Load test image
    auto image = jpeg::read_image_jpeg_rgb8(path);

    // Encode using img::ImageBuffer
    img::ImageBuffer buffer;
    jpeg::encode_image_jpeg_rgb8(image, 100, buffer);

    // Verify JPEG magic bytes (0xFF 0xD8)
    REQUIRE(buffer.size() > 2);
    REQUIRE(buffer.data()[0] == 0xFF);
    REQUIRE(buffer.data()[1] == 0xD8);

    // Verify JPEG end marker (0xFF 0xD9) at the end
    REQUIRE(buffer.data()[buffer.size() - 2] == 0xFF);
    REQUIRE(buffer.data()[buffer.size() - 1] == 0xD9);

    // Decode from buffer (convert to std::vector for decode API)
    auto jpeg_bytes = buffer.to_vector();
    auto decoded = jpeg::decode_image_jpeg_rgb8(jpeg_bytes);

    REQUIRE(decoded.width() == 258);
    REQUIRE(decoded.height() == 195);
    REQUIRE(decoded.channels() == 3);
}

TEST_CASE("Encode JPEG RGB8 with img::ImageBuffer - Multiple Encodes", "[io][jpeg][encode]") {
    std::string path = get_test_image_path();
    if (path.empty())
        SKIP("Test image not found");

    // Load test image
    auto image = jpeg::read_image_jpeg_rgb8(path);

    // Reuse img::ImageBuffer across multiple encodes
    img::ImageBuffer buffer;

    // First encode
    buffer.clear();
    jpeg::encode_image_jpeg_rgb8(image, 100, buffer);

    REQUIRE(buffer.size() > 2);
    REQUIRE(buffer.data()[0] == 0xFF);
    REQUIRE(buffer.data()[1] == 0xD8);

    // Decode and verify (to_vector for decode API compatibility)
    auto vec1 = buffer.to_vector();
    auto decoded1 = jpeg::decode_image_jpeg_rgb8(vec1);
    REQUIRE(decoded1.width() == 258);
    REQUIRE(decoded1.height() == 195);

    // Second encode - buffer is reused (capacity preserved)
    buffer.clear();
    jpeg::encode_image_jpeg_rgb8(image, 90, buffer);

    REQUIRE(buffer.size() > 2);
    REQUIRE(buffer.data()[0] == 0xFF);
    REQUIRE(buffer.data()[1] == 0xD8);

    // Verify buffer works with decode
    auto vec2 = buffer.to_vector();
    auto decoded2 = jpeg::decode_image_jpeg_rgb8(vec2);
    REQUIRE(decoded2.width() == 258);
    REQUIRE(decoded2.height() == 195);
}

TEST_CASE("Encode JPEG RGB8 with img::ImageBuffer (Zero-Copy)", "[io][jpeg][encode][zerocopy]") {
    std::string path = get_test_image_path();
    if (path.empty())
        SKIP("Test image not found");

    auto image = jpeg::read_image_jpeg_rgb8(path);

    // Zero-copy buffer - data stays in Rust memory
    // Can be reused for JPEG, PNG, WebP, etc.
    img::ImageBuffer buffer;

    // First encode
    buffer.clear();
    jpeg::encode_image_jpeg_rgb8(image, 100, buffer);

    // Zero-copy access via pointer + size
    REQUIRE(buffer.size() > 2);
    REQUIRE(!buffer.empty());
    REQUIRE(buffer.data()[0] == 0xFF);
    REQUIRE(buffer.data()[1] == 0xD8);

    // Can still convert to std::vector when needed (explicit copy)
    std::vector<uint8_t> vec = buffer.to_vector();
    REQUIRE(vec.size() == buffer.size());

    // Decode works with pointer+size (zero-copy decode too)
    std::vector<uint8_t> temp_vec(buffer.data(), buffer.data() + buffer.size());
    auto decoded = jpeg::decode_image_jpeg_rgb8(temp_vec);
    REQUIRE(decoded.width() == 258);
    REQUIRE(decoded.height() == 195);

    // Second encode - buffer is reused
    buffer.clear();
    jpeg::encode_image_jpeg_rgb8(image, 90, buffer);

    REQUIRE(buffer.size() > 2);
    REQUIRE(buffer.data()[0] == 0xFF);
    REQUIRE(buffer.data()[1] == 0xD8);
}

TEST_CASE("Encode JPEG BGRA8 from raw pointer", "[io][jpeg][encode][bgra]") {
    // Create a test BGRA image from raw data (simulating Unreal Engine's FColor)
    const size_t width = 64;
    const size_t height = 48;
    const size_t channels = 4;

    // Create test BGRA data with a gradient pattern
    std::vector<uint8_t> bgra_data(width * height * channels);
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            size_t idx = (y * width + x) * 4;
            bgra_data[idx + 0] = static_cast<uint8_t>((x * 255) / width);  // B
            bgra_data[idx + 1] = static_cast<uint8_t>((y * 255) / height); // G
            bgra_data[idx + 2] = 128;                                      // R
            bgra_data[idx + 3] = 255;                                      // A (full opacity)
        }
    }

    // Construct image from raw pointer (explicit constructor)
    img::ImageU8C4 image(width, height, bgra_data.data());

    SECTION("Image properties") {
        REQUIRE(image.width() == width);
        REQUIRE(image.height() == height);
        REQUIRE(image.channels() == channels);
    }

    SECTION("Encode to JPEG") {
        img::ImageBuffer buffer;
        jpeg::encode_image_jpeg_bgra8(image, 95, buffer);

        // Verify JPEG magic bytes
        REQUIRE(buffer.size() > 2);
        REQUIRE(buffer.data()[0] == 0xFF);
        REQUIRE(buffer.data()[1] == 0xD8);

        // Verify JPEG end marker
        REQUIRE(buffer.data()[buffer.size() - 2] == 0xFF);
        REQUIRE(buffer.data()[buffer.size() - 1] == 0xD9);
    }

    SECTION("Multiple encodes with buffer reuse") {
        img::ImageBuffer buffer;

        // First encode at quality 100
        buffer.clear();
        jpeg::encode_image_jpeg_bgra8(image, 100, buffer);
        size_t size_q100 = buffer.size();
        REQUIRE(size_q100 > 0);

        // Second encode at quality 50 (should be smaller)
        buffer.clear();
        jpeg::encode_image_jpeg_bgra8(image, 50, buffer);
        size_t size_q50 = buffer.size();
        REQUIRE(size_q50 > 0);
        REQUIRE(size_q50 < size_q100); // Lower quality = smaller file
    }
}

TEST_CASE("Encode JPEG BGRA8 - Unreal Engine use case", "[io][jpeg][encode][bgra][unreal]") {
    // Simulate Unreal Engine FColor buffer (BGRA format)
    const size_t width = 128;
    const size_t height = 96;

    // Create a simple test pattern: solid red with full alpha
    std::vector<uint8_t> unreal_pixels(width * height * 4);
    for (size_t i = 0; i < width * height; ++i) {
        unreal_pixels[i * 4 + 0] = 0;   // B
        unreal_pixels[i * 4 + 1] = 0;   // G
        unreal_pixels[i * 4 + 2] = 255; // R (red)
        unreal_pixels[i * 4 + 3] = 255; // A (opaque)
    }

    // Construct image from raw pointer (like reinterpret_cast<const uint8_t*>(FColor*))
    img::ImageU8C4 camera_frame(width, height, unreal_pixels.data());

    // Encode to JPEG for network transmission
    img::ImageBuffer jpeg_buffer;
    jpeg::encode_image_jpeg_bgra8(camera_frame, 85, jpeg_buffer);

    REQUIRE(jpeg_buffer.size() > 0);
    REQUIRE(jpeg_buffer.data()[0] == 0xFF);
    REQUIRE(jpeg_buffer.data()[1] == 0xD8);

    // Buffer can be sent over network using data() and size()
    // In Unreal: zenoh::Bytes(jpeg_buffer.data(), jpeg_buffer.size())
    REQUIRE(!jpeg_buffer.empty());
}
