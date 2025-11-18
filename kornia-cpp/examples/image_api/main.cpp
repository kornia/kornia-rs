#include <iostream>
#include <kornia.hpp>

void print_separator() {
    std::cout << "================================================\n";
}

void example_create_from_value() {
    std::cout << "Example 1: Creating images from fill values\n";
    print_separator();

    try {
        // Create a 640x480 RGB image filled with value 128
        auto img = kornia::ImageU8C3(640, 480, 128);
        std::cout << "✓ Created " << img.width() << "x" << img.height() << "x" << img.channels()
                  << " image\n";
        std::cout << "  Fill value: " << (int)img.data()[0] << "\n";
        std::cout << "  Total bytes: " << img.data().size() << "\n";

    } catch (const std::exception& e) {
        std::cerr << "✗ Error: " << e.what() << "\n";
    }

    std::cout << "\n";
}

void example_create_from_data() {
    std::cout << "Example 2: Creating images from data vectors\n";
    print_separator();

    try {
        // Create data: 10x10 grayscale gradient
        std::vector<uint8_t> data(10 * 10);
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = static_cast<uint8_t>((i * 255) / data.size());
        }

        auto img = kornia::ImageU8C1(10, 10, data);
        std::cout << "✓ Created " << img.width() << "x" << img.height()
                  << " grayscale image from data\n";
        std::cout << "  First pixel: " << (int)img.data()[0] << "\n";
        std::cout << "  Last pixel: " << (int)img.data()[img.data().size() - 1] << "\n";

    } catch (const std::exception& e) {
        std::cerr << "✗ Error: " << e.what() << "\n";
    }

    std::cout << "\n";
}

void example_error_handling() {
    std::cout << "Example 3: Error handling (wrong data size)\n";
    print_separator();

    try {
        // Deliberately create wrong-sized data
        std::vector<uint8_t> data(100); // Need 300 for 10x10x3 image
        auto img = kornia::ImageU8C3(10, 10, data);
        std::cout << "✓ Created image (should not reach here)\n";

    } catch (const std::exception& e) {
        std::cout << "✓ Caught expected error:\n";
        std::cout << "  " << e.what() << "\n";
    }

    std::cout << "\n";
}

void example_zero_copy_access() {
    std::cout << "Example 4: Zero-copy data access\n";
    print_separator();

    try {
        // Create small RGB image
        auto img = kornia::ImageU8C3(5, 5, 42);

        // Zero-copy access to underlying Rust data
        auto data = img.data(); // rust::Slice<const uint8_t>

        std::cout << "✓ Image dimensions: " << img.width() << "x" << img.height() << "x"
                  << img.channels() << "\n";
        std::cout << "  Data is zero-copy view into Rust memory\n";
        std::cout << "  Total elements: " << data.size() << "\n";

        // Access specific pixels (row-major, interleaved RGB)
        size_t pixel_idx = (2 * img.width() + 3) * img.channels(); // Row 2, Col 3
        std::cout << "  Pixel (2,3) RGB: (" << (int)data[pixel_idx + 0] << ", "
                  << (int)data[pixel_idx + 1] << ", " << (int)data[pixel_idx + 2] << ")\n";

    } catch (const std::exception& e) {
        std::cerr << "✗ Error: " << e.what() << "\n";
    }

    std::cout << "\n";
}

void example_copy_data() {
    std::cout << "Example 5: Creating owned copy of image data\n";
    print_separator();

    try {
        // Create image
        auto img = kornia::ImageF32C3(3, 3, 0.5f);

        // Zero-copy view
        auto data_view = img.data();
        std::cout << "✓ Zero-copy view size: " << data_view.size() << " elements\n";

        // Create owned copy
        auto data_copy = img.to_vec();
        std::cout << "✓ Owned copy size: " << data_copy.size() << " elements\n";
        std::cout << "  First element: " << data_copy[0] << "\n";
        std::cout << "  Can modify copy independently of original\n";

    } catch (const std::exception& e) {
        std::cerr << "✗ Error: " << e.what() << "\n";
    }

    std::cout << "\n";
}

void example_different_types() {
    std::cout << "Example 6: Different image types\n";
    print_separator();

    try {
        // U8 images (8-bit unsigned)
        auto gray_u8 = kornia::ImageU8C1(100, 100, 255);
        auto rgb_u8 = kornia::ImageU8C3(100, 100, 128);
        auto rgba_u8 = kornia::ImageU8C4(100, 100, 64);

        // F32 images (32-bit float, common for ML/processing)
        auto gray_f32 = kornia::ImageF32C1(100, 100, 1.0f);
        auto rgb_f32 = kornia::ImageF32C3(100, 100, 0.5f);
        auto rgba_f32 = kornia::ImageF32C4(100, 100, 0.25f);

        std::cout << "✓ Created 6 different image types:\n";
        std::cout << "  U8:  Grayscale (C1), RGB (C3), RGBA (C4)\n";
        std::cout << "  F32: Grayscale (C1), RGB (C3), RGBA (C4)\n";
        std::cout << "\nAll types support:\n";
        std::cout << "  - width(), height(), channels(), size()\n";
        std::cout << "  - data() for zero-copy access\n";
        std::cout << "  - to_vec() for owned copy\n";

    } catch (const std::exception& e) {
        std::cerr << "✗ Error: " << e.what() << "\n";
    }

    std::cout << "\n";
}

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════╗\n";
    std::cout << "║   Kornia C++ Image API Examples           ║\n";
    std::cout << "╚════════════════════════════════════════════╝\n";
    std::cout << "\n";

    example_create_from_value();
    example_create_from_data();
    example_error_handling();
    example_zero_copy_access();
    example_copy_data();
    example_different_types();

    std::cout << "╔════════════════════════════════════════════╗\n";
    std::cout << "║   All examples completed successfully!     ║\n";
    std::cout << "╚════════════════════════════════════════════╝\n";
    std::cout << "\n";

    return 0;
}
