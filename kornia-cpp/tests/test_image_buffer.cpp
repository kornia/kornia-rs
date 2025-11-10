#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <kornia/image.hpp>
#include <vector>

using namespace kornia;
using namespace kornia::image;

TEST_CASE("ImageBuffer - Default Construction", "[image][buffer]") {
    ImageBuffer buffer;

    REQUIRE(buffer.empty());
    REQUIRE(buffer.size() == 0);
    REQUIRE(buffer.data() != nullptr); // Even empty buffer has valid pointer
}

TEST_CASE("ImageBuffer - Data Access", "[image][buffer]") {
    ImageBuffer buffer;

    // Manually write some data through rust_vec() (simulating encoder)
    auto& rust_vec = buffer.rust_vec();
    rust_vec.push_back(0xFF);
    rust_vec.push_back(0xD8);
    rust_vec.push_back(0xFF);
    rust_vec.push_back(0xE0);

    // Zero-copy pointer access
    REQUIRE_FALSE(buffer.empty());
    REQUIRE(buffer.size() == 4);
    REQUIRE(buffer.data()[0] == 0xFF);
    REQUIRE(buffer.data()[1] == 0xD8);
    REQUIRE(buffer.data()[2] == 0xFF);
    REQUIRE(buffer.data()[3] == 0xE0);
}

TEST_CASE("ImageBuffer - Clear and Reuse", "[image][buffer]") {
    ImageBuffer buffer;

    // First use
    auto& rust_vec = buffer.rust_vec();
    rust_vec.push_back(0x01);
    rust_vec.push_back(0x02);
    REQUIRE(buffer.size() == 2);

    // Clear
    buffer.clear();
    REQUIRE(buffer.empty());
    REQUIRE(buffer.size() == 0);

    // Reuse - capacity should be preserved
    rust_vec.push_back(0x03);
    rust_vec.push_back(0x04);
    rust_vec.push_back(0x05);
    REQUIRE(buffer.size() == 3);
    REQUIRE(buffer.data()[0] == 0x03);
    REQUIRE(buffer.data()[1] == 0x04);
    REQUIRE(buffer.data()[2] == 0x05);
}

TEST_CASE("ImageBuffer - to_vector Conversion", "[image][buffer]") {
    ImageBuffer buffer;

    // Add data
    auto& rust_vec = buffer.rust_vec();
    for (uint8_t i = 0; i < 10; ++i) {
        rust_vec.push_back(i);
    }

    // Convert to std::vector (explicit copy)
    std::vector<uint8_t> vec = buffer.to_vector();

    REQUIRE(vec.size() == 10);
    for (size_t i = 0; i < 10; ++i) {
        REQUIRE(vec[i] == i);
    }

    // Original buffer should be unchanged
    REQUIRE(buffer.size() == 10);
    REQUIRE(buffer.data()[0] == 0);
}

TEST_CASE("ImageBuffer - Zero-Copy Semantics", "[image][buffer]") {
    ImageBuffer buffer;

    // Add data
    auto& rust_vec = buffer.rust_vec();
    rust_vec.push_back(0xAA);
    rust_vec.push_back(0xBB);
    rust_vec.push_back(0xCC);

    // Get pointer (zero-copy)
    const uint8_t* ptr1 = buffer.data();
    size_t size1 = buffer.size();

    // Verify we're accessing the same memory
    REQUIRE(ptr1[0] == 0xAA);
    REQUIRE(ptr1[1] == 0xBB);
    REQUIRE(ptr1[2] == 0xCC);
    REQUIRE(size1 == 3);

    // Add more data
    rust_vec.push_back(0xDD);

    // Pointer might change due to reallocation, but still zero-copy
    const uint8_t* ptr2 = buffer.data();
    REQUIRE(ptr2[0] == 0xAA);
    REQUIRE(ptr2[3] == 0xDD);
    REQUIRE(buffer.size() == 4);
}

TEST_CASE("ImageBuffer - Multiple Formats Reuse", "[image][buffer]") {
    ImageBuffer buffer;

    // Simulate JPEG encoding
    buffer.clear();
    auto& rust_vec = buffer.rust_vec();
    rust_vec.push_back(0xFF); // JPEG magic
    rust_vec.push_back(0xD8);
    REQUIRE(buffer.size() == 2);
    REQUIRE(buffer.data()[0] == 0xFF);

    // Reuse for PNG encoding
    buffer.clear();
    rust_vec.push_back(0x89); // PNG magic
    rust_vec.push_back(0x50);
    rust_vec.push_back(0x4E);
    rust_vec.push_back(0x47);
    REQUIRE(buffer.size() == 4);
    REQUIRE(buffer.data()[0] == 0x89);

    // Reuse for WebP encoding
    buffer.clear();
    rust_vec.push_back(0x52); // RIFF
    rust_vec.push_back(0x49);
    rust_vec.push_back(0x46);
    rust_vec.push_back(0x46);
    REQUIRE(buffer.size() == 4);
    REQUIRE(buffer.data()[0] == 0x52);
}

TEST_CASE("ImageBuffer - Large Data", "[image][buffer]") {
    ImageBuffer buffer;

    // Simulate encoding a large image (1MB)
    const size_t large_size = 1024 * 1024;
    auto& rust_vec = buffer.rust_vec();

    for (size_t i = 0; i < large_size; ++i) {
        rust_vec.push_back(static_cast<uint8_t>(i & 0xFF));
    }

    REQUIRE(buffer.size() == large_size);
    REQUIRE_FALSE(buffer.empty());

    // Verify some data points
    REQUIRE(buffer.data()[0] == 0);
    REQUIRE(buffer.data()[255] == 255);
    REQUIRE(buffer.data()[256] == 0);

    // Clear should handle large buffer
    buffer.clear();
    REQUIRE(buffer.empty());
    REQUIRE(buffer.size() == 0);
}

TEST_CASE("ImageBuffer - Copy to C Array", "[image][buffer]") {
    ImageBuffer buffer;

    // Add test data
    auto& rust_vec = buffer.rust_vec();
    rust_vec.push_back(0x01);
    rust_vec.push_back(0x02);
    rust_vec.push_back(0x03);
    rust_vec.push_back(0x04);

    // Copy to C-style array (zero-copy read, explicit copy write)
    uint8_t c_array[4];
    std::memcpy(c_array, buffer.data(), buffer.size());

    REQUIRE(c_array[0] == 0x01);
    REQUIRE(c_array[1] == 0x02);
    REQUIRE(c_array[2] == 0x03);
    REQUIRE(c_array[3] == 0x04);
}

TEST_CASE("ImageBuffer - Move Semantics", "[image][buffer]") {
    ImageBuffer buffer1;

    // Add data to buffer1
    auto& rust_vec1 = buffer1.rust_vec();
    rust_vec1.push_back(0xAA);
    rust_vec1.push_back(0xBB);

    // Move to buffer2
    kornia::image::ImageBuffer buffer2 = std::move(buffer1);

    // buffer2 should have the data
    REQUIRE(buffer2.size() == 2);
    REQUIRE(buffer2.data()[0] == 0xAA);
    REQUIRE(buffer2.data()[1] == 0xBB);

    // Note: buffer1 state after move is implementation-defined
    // We just ensure buffer2 works correctly
}
