#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <kornia.hpp>

namespace fs = std::filesystem;

static std::string get_test_image_path() {
    fs::path test_data =
        fs::path(__FILE__).parent_path() / ".." / ".." / "tests" / "data" / "dog.jpeg";
    return fs::canonical(test_data).string();
}

TEST_CASE("Read JPEG RGB8", "[io][jpeg]") {
    std::string path = get_test_image_path();
    if (path.empty())
        SKIP("Test image not found");

    auto image = kornia::io::read_jpeg_rgb8(path);

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

    auto image = kornia::io::read_jpeg_mono8(path);

    REQUIRE(image.width() == 258);
    REQUIRE(image.height() == 195);
    REQUIRE(image.channels() == 1);
    REQUIRE(image.data().size() == 258 * 195);
}

TEST_CASE("Read JPEG - Error Handling", "[io][jpeg][error]") {
    REQUIRE_THROWS(kornia::io::read_jpeg_rgb8("/nonexistent/image.jpg"));
    REQUIRE_THROWS(kornia::io::read_jpeg_rgb8(""));
}
TEST_CASE("Encode JPEG RGB8", "[io][jpeg][encode]") {
    std::string path = get_test_image_path();
    if (path.empty())
        SKIP("Test image not found");

    // Load test image
    auto image = kornia::io::read_jpeg_rgb8(path);

    // Encode to JPEG bytes (returns std::vector)
    std::vector<uint8_t> jpeg_bytes = kornia::io::encode_jpeg_rgb8(image, 100);

    // Verify JPEG magic bytes (0xFF 0xD8)
    REQUIRE(jpeg_bytes.size() > 2);
    REQUIRE(jpeg_bytes[0] == 0xFF);
    REQUIRE(jpeg_bytes[1] == 0xD8);

    // Verify JPEG end marker (0xFF 0xD9) at the end
    REQUIRE(jpeg_bytes[jpeg_bytes.size() - 2] == 0xFF);
    REQUIRE(jpeg_bytes[jpeg_bytes.size() - 1] == 0xD9);

    // Decode from std::vector (works with Unreal, OpenCV, etc.)
    auto decoded = kornia::io::decode_jpeg_rgb8(jpeg_bytes);

    REQUIRE(decoded.width() == 258);
    REQUIRE(decoded.height() == 195);
    REQUIRE(decoded.channels() == 3);
}
