#include <catch2/catch_test_macros.hpp>
#include <kornia/image.hpp>

TEST_CASE("ImageSize construction", "[image][size]") {
    kornia::image::ImageSize size{640, 480};

    REQUIRE(size.width == 640);
    REQUIRE(size.height == 480);
}

TEST_CASE("ImageU8C3 basic interface", "[image][u8c3]") {
    // Create a test image directly using constructor (100x80, filled with value 42)
    kornia::image::ImageU8C3 image(100, 80, 42);

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
    // Create a grayscale test image directly using constructor (50x60, filled with value 128)
    kornia::image::ImageU8C1 image(50, 60, 128);

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
    kornia::image::ImageU8C3 image1(100, 80, 42);

    SECTION("Move construction") {
        kornia::image::ImageU8C3 image2 = std::move(image1);
        REQUIRE(image2.width() == 100);
        REQUIRE(image2.height() == 80);
        REQUIRE(image2.channels() == 3);
    }

    SECTION("Move assignment") {
        kornia::image::ImageU8C3 image2(50, 50, uint8_t(0));
        image2 = std::move(image1);
        REQUIRE(image2.width() == 100);
        REQUIRE(image2.height() == 80);
    }
}

TEST_CASE("Image data layout", "[image][layout]") {
    // Create test image with known dimensions
    kornia::image::ImageU8C3 image(100, 80, 42);

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

TEST_CASE("Image from vector", "[image][from_vector]") {
    // Create image from vector of data
    std::vector<uint8_t> data(10 * 8 * 3, 99);
    kornia::image::ImageU8C3 image(10, 8, data);

    SECTION("Dimensions") {
        REQUIRE(image.width() == 10);
        REQUIRE(image.height() == 8);
        REQUIRE(image.channels() == 3);
    }

    SECTION("Data values") {
        auto img_data = image.data();
        REQUIRE(img_data.size() == 10 * 8 * 3);
        REQUIRE(img_data[0] == 99);
        REQUIRE(img_data[img_data.size() - 1] == 99);
    }
}

TEST_CASE("Image from raw pointer", "[image][from_pointer]") {
    // Create test data in a raw array
    const size_t width = 10;
    const size_t height = 8;
    const size_t channels = 3;
    std::vector<uint8_t> raw_data(width * height * channels, 77);

    // Construct image from raw pointer (explicit constructor)
    kornia::image::ImageU8C3 image(width, height, raw_data.data());

    SECTION("Dimensions") {
        REQUIRE(image.width() == width);
        REQUIRE(image.height() == height);
        REQUIRE(image.channels() == channels);
    }

    SECTION("Data values") {
        auto img_data = image.data();
        REQUIRE(img_data.size() == width * height * channels);
        REQUIRE(img_data[0] == 77);
        REQUIRE(img_data[img_data.size() - 1] == 77);
    }

    SECTION("Data is copied - modifying source doesn't affect image") {
        // Modify the source data
        raw_data[0] = 123;

        // Image data should be unchanged (data was copied during construction)
        auto img_data = image.data();
        REQUIRE(img_data[0] == 77);
    }
}

TEST_CASE("Image from raw pointer - BGRA", "[image][from_pointer][bgra]") {
    // Test with 4-channel BGRA image (common in graphics APIs like Unreal/DirectX)
    const size_t width = 16;
    const size_t height = 12;
    const size_t channels = 4;

    // Create BGRA data: [B, G, R, A, B, G, R, A, ...]
    std::vector<uint8_t> bgra_data(width * height * channels);
    for (size_t i = 0; i < width * height; ++i) {
        bgra_data[i * 4 + 0] = 255; // B (blue)
        bgra_data[i * 4 + 1] = 128; // G (green)
        bgra_data[i * 4 + 2] = 64;  // R (red)
        bgra_data[i * 4 + 3] = 255; // A (alpha)
    }

    // Construct BGRA image from raw pointer
    kornia::image::ImageU8C4 image(width, height, bgra_data.data());

    SECTION("Dimensions") {
        REQUIRE(image.width() == width);
        REQUIRE(image.height() == height);
        REQUIRE(image.channels() == channels);
    }

    SECTION("Data values preserved") {
        auto img_data = image.data();
        REQUIRE(img_data.size() == width * height * channels);

        // Verify first pixel BGRA values
        REQUIRE(img_data[0] == 255); // B
        REQUIRE(img_data[1] == 128); // G
        REQUIRE(img_data[2] == 64);  // R
        REQUIRE(img_data[3] == 255); // A
    }
}
