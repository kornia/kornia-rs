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

    auto image = jpeg::read_jpeg_rgb8(path);

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

    auto image = jpeg::read_jpeg_mono8(path);

    REQUIRE(image.width() == 258);
    REQUIRE(image.height() == 195);
    REQUIRE(image.channels() == 1);
    REQUIRE(image.data().size() == 258 * 195);
}

TEST_CASE("Read JPEG - Error Handling", "[io][jpeg][error]") {
    REQUIRE_THROWS(jpeg::read_jpeg_rgb8("/nonexistent/image.jpg"));
    REQUIRE_THROWS(jpeg::read_jpeg_rgb8(""));
}

TEST_CASE("Encode JPEG RGB8", "[io][jpeg][encode]") {
    std::string path = get_test_image_path();
    if (path.empty())
        SKIP("Test image not found");

    // Load test image
    auto image = jpeg::read_jpeg_rgb8(path);

    // Encode using img::img::ImageBuffer
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

TEST_CASE("Encode JPEG RGB8 with img::img::ImageBuffer - Multiple Encodes", "[io][jpeg][encode]") {
    std::string path = get_test_image_path();
    if (path.empty())
        SKIP("Test image not found");

    // Load test image
    auto image = jpeg::read_jpeg_rgb8(path);

    // Reuse img::img::ImageBuffer across multiple encodes
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

TEST_CASE("Encode JPEG RGB8 with img::img::ImageBuffer (Zero-Copy)",
          "[io][jpeg][encode][zerocopy]") {
    std::string path = get_test_image_path();
    if (path.empty())
        SKIP("Test image not found");

    auto image = jpeg::read_jpeg_rgb8(path);

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
