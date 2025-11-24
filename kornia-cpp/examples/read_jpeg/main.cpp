#include <exception>
#include <iostream>
#include <kornia.hpp>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_jpeg_image>" << std::endl;
        return 1;
    }

    try {
        std::cout << "Kornia C++ Library v" << kornia::version() << std::endl;
        std::cout << "Reading JPEG image from: " << argv[1] << std::endl;

        // Read RGB image - wraps kornia_image::Image<u8, 3> (zero-copy)
        // Use fully qualified name to avoid ambiguity with CXX bridge types
        kornia::image::ImageU8C3 img = kornia::io::jpeg::read_jpeg_rgb8(argv[1]);

        // Print image information
        std::cout << "\n✓ Successfully loaded image!" << std::endl;
        std::cout << "  Dimensions: " << img.width() << " x " << img.height() << std::endl;
        std::cout << "  Channels: " << img.channels() << std::endl;

        auto data = img.data();
        std::cout << "  Data size: " << data.size() << " bytes" << std::endl;

        // Access pixel data directly (from Rust - no copy!)
        std::cout << "\n  First 10 bytes: ";
        for (size_t i = 0; i < std::min(size_t(10), data.size()); ++i) {
            std::cout << static_cast<int>(data[i]) << " ";
        }
        std::cout << std::endl;

        // Pixel access example (row-major, interleaved channels)
        size_t idx = (0 * img.width() + 0) * img.channels() + 0; // pixel (0,0), channel R
        std::cout << "  Pixel (0,0) R channel: " << static_cast<int>(data[idx]) << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
