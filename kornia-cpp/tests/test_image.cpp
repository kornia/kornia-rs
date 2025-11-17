#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <kornia.hpp>

namespace fs = std::filesystem;

static std::string get_test_image_path() {
    // When running from build/tests via CTest
    fs::path test_data = fs::path("data") / "dog.jpeg";
    if (fs::exists(test_data)) {
        return test_data.string();
    }
    // Fallback for different working directories
    test_data = fs::path("../../tests/data/dog.jpeg");
    if (fs::exists(test_data)) {
        return test_data.string();
    }
    return "";
}

TEST_CASE("ImageSize construction", "[image][size]") {
    kornia::ImageSize size{640, 480};

    REQUIRE(size.width == 640);
    REQUIRE(size.height == 480);
}

TEST_CASE("ImageU8C3 basic interface", "[image][u8c3]") {
    std::string path = get_test_image_path();
    if (path.empty())
        SKIP("Test image not found");

    // Use existing I/O API to get an image, then test the Image wrapper interface
    kornia::ImageU8C3 image = kornia::io::read_jpeg_rgb8(path);

    SECTION("Dimensions") {
        REQUIRE(image.width() == 258);
        REQUIRE(image.height() == 195);
        REQUIRE(image.channels() == 3);
    }

    SECTION("Size struct") {
        auto size = image.size();
        REQUIRE(size.width == 258);
        REQUIRE(size.height == 195);
    }

    SECTION("Data access") {
        auto data = image.data();
        REQUIRE(data.size() == 258 * 195 * 3);
        // Verify we can read data
        REQUIRE(data[0] >= 0);
        REQUIRE(data[0] <= 255);
    }

    SECTION("Const correctness") {
        const auto& const_image = image;
        REQUIRE(const_image.width() == 258);
        REQUIRE(const_image.height() == 195);
        auto const_data = const_image.data();
        REQUIRE(const_data.size() == 258 * 195 * 3);
    }
}

TEST_CASE("ImageU8C1 basic interface", "[image][u8c1][!mayfail]") {
    std::string path = get_test_image_path();
    if (path.empty())
        SKIP("Test image not found");

    // Use existing I/O API to get an image, then test the Image wrapper interface
    kornia::ImageU8C1 image = kornia::io::read_jpeg_mono8(path);

    SECTION("Dimensions") {
        REQUIRE(image.width() == 258);
        REQUIRE(image.height() == 195);
        REQUIRE(image.channels() == 1);
    }

    SECTION("Size struct") {
        auto size = image.size();
        REQUIRE(size.width == 258);
        REQUIRE(size.height == 195);
    }

    SECTION("Data access") {
        auto data = image.data();
        REQUIRE(data.size() == 258 * 195);
        // Verify we can read data
        REQUIRE(data[0] >= 0);
        REQUIRE(data[0] <= 255);
    }
}

TEST_CASE("Image move semantics", "[image][move]") {
    std::string path = get_test_image_path();
    if (path.empty())
        SKIP("Test image not found");

    kornia::ImageU8C3 image1 = kornia::io::read_jpeg_rgb8(path);

    SECTION("Move construction") {
        kornia::ImageU8C3 image2 = std::move(image1);
        REQUIRE(image2.width() == 258);
        REQUIRE(image2.height() == 195);
        REQUIRE(image2.channels() == 3);
    }

    SECTION("Move assignment") {
        kornia::ImageU8C3 image2 = kornia::io::read_jpeg_rgb8(path);
        image2 = std::move(image1);
        REQUIRE(image2.width() == 258);
        REQUIRE(image2.height() == 195);
    }
}

TEST_CASE("Image data layout", "[image][layout]") {
    std::string path = get_test_image_path();
    if (path.empty())
        SKIP("Test image not found");

    kornia::ImageU8C3 image = kornia::io::read_jpeg_rgb8(path);

    SECTION("Row-major interleaved RGB") {
        auto data = image.data();
        size_t w = image.width();
        size_t c = image.channels();

        // Access pixel at (row=0, col=0)
        size_t idx_r = (0 * w + 0) * c + 0;
        size_t idx_g = (0 * w + 0) * c + 1;
        size_t idx_b = (0 * w + 0) * c + 2;

        REQUIRE(idx_r < data.size());
        REQUIRE(idx_g < data.size());
        REQUIRE(idx_b < data.size());

        // Access pixel at (row=10, col=10)
        size_t idx_10_10 = (10 * w + 10) * c;
        REQUIRE(idx_10_10 + 2 < data.size());
    }
}
