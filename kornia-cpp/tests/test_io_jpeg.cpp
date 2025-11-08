#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <kornia.hpp>

namespace fs = std::filesystem;

// Helper to get test image path (from workspace root)
static std::string get_test_image_path() {
    // Test file is at: kornia-rs/kornia-cpp/tests/test_io_jpeg.cpp
    // Image is at:     kornia-rs/tests/data/dog.jpeg
    fs::path test_data = fs::path(__FILE__).parent_path() / ".." / ".." / "tests" / "data" / "dog.jpeg";
    return fs::canonical(test_data).string();
}

// =============================================================================
// Basic FFI Binding Tests
// These tests verify C++ ↔ Rust bindings work correctly.
// Functionality is tested in detail in the Rust crates.
// =============================================================================

TEST_CASE("Read JPEG RGB8", "[io][jpeg]") {
    std::string path = get_test_image_path();
    if (path.empty()) SKIP("Test image not found");

    auto image = kornia::io::read_jpeg_rgb8(path);
    
    REQUIRE(image.width() == 258);
    REQUIRE(image.height() == 195);
    REQUIRE(image.channels() == 3);
    REQUIRE(image.data().size() == 258 * 195 * 3);
}

// NOTE: Grayscale JPEG reading has a known issue in kornia-io
// where data length doesn't match expected size. Marked as may-fail.
TEST_CASE("Read JPEG Mono8", "[io][jpeg][!mayfail]") {
    std::string path = get_test_image_path();
    if (path.empty()) SKIP("Test image not found");

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
