#include <catch2/catch_test_macros.hpp>
#include <kornia.hpp>

TEST_CASE("ImageSize construction", "[image][size]") {
    kornia::ImageSize size{640, 480};

    REQUIRE(size.width == 640);
    REQUIRE(size.height == 480);
}

TEST_CASE("ImageU8C3 basic interface", "[image][u8c3]") {
    // Create a test image directly using Image API (100x80, filled with value 42)
    kornia::ImageU8C3 image = kornia::image::image_u8c3_from_size_val(100, 80, 42);

    SECTION("Dimensions") {
        REQUIRE(image.width() == 100);
        REQUIRE(image.height() == 80);
        REQUIRE(image.channels() == 3);
    }

    SECTION("Size struct") {
        auto size = image.size();
        REQUIRE(size.width == 100);
        REQUIRE(size.height == 80);
    }

    SECTION("Data access") {
        auto data = image.data();
        REQUIRE(data.size() == 100 * 80 * 3);
        // Verify data is filled with expected value
        REQUIRE(data[0] == 42);
        REQUIRE(data[100] == 42);
        REQUIRE(data[data.size() - 1] == 42);
    }

    SECTION("Const correctness") {
        const auto& const_image = image;
        REQUIRE(const_image.width() == 100);
        REQUIRE(const_image.height() == 80);
        auto const_data = const_image.data();
        REQUIRE(const_data.size() == 100 * 80 * 3);
    }
}

TEST_CASE("ImageU8C1 basic interface", "[image][u8c1]") {
    // Create a grayscale test image directly using Image API (50x60, filled with value 128)
    kornia::ImageU8C1 image = kornia::image::image_u8c1_from_size_val(50, 60, 128);

    SECTION("Dimensions") {
        REQUIRE(image.width() == 50);
        REQUIRE(image.height() == 60);
        REQUIRE(image.channels() == 1);
    }

    SECTION("Size struct") {
        auto size = image.size();
        REQUIRE(size.width == 50);
        REQUIRE(size.height == 60);
    }

    SECTION("Data access") {
        auto data = image.data();
        REQUIRE(data.size() == 50 * 60);
        // Verify data is filled with expected value
        REQUIRE(data[0] == 128);
        REQUIRE(data[100] == 128);
        REQUIRE(data[data.size() - 1] == 128);
    }
}

TEST_CASE("Image move semantics", "[image][move]") {
    kornia::ImageU8C3 image1 = kornia::image::image_u8c3_from_size_val(100, 80, 42);

    SECTION("Move construction") {
        kornia::ImageU8C3 image2 = std::move(image1);
        REQUIRE(image2.width() == 100);
        REQUIRE(image2.height() == 80);
        REQUIRE(image2.channels() == 3);
    }

    SECTION("Move assignment") {
        kornia::ImageU8C3 image2 = kornia::image::image_u8c3_from_size_val(50, 50, 0);
        image2 = std::move(image1);
        REQUIRE(image2.width() == 100);
        REQUIRE(image2.height() == 80);
    }
}

TEST_CASE("Image data layout", "[image][layout]") {
    // Create test image with known dimensions
    kornia::ImageU8C3 image = kornia::image::image_u8c3_from_size_val(100, 80, 42);

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
