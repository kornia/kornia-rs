#include <exception>
#include <iostream>
#include <kornia/kornia.hpp>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_jpeg_image>" << std::endl;
        return 1;
    }

    try {
        std::cout << "Kornia C++ Library v" << kornia::version() << std::endl;
        std::cout << "Reading JPEG image from: " << argv[1] << std::endl;

        // Read RGB image - returns Rust ImageResult directly (zero-copy)
        kornia::Image image = kornia::io::read_jpeg_rgb(argv[1]);

        // Print image information
        std::cout << "\n✓ Successfully loaded image!" << std::endl;
        std::cout << "  Dimensions: " << image.width << " x " << image.height << std::endl;
        std::cout << "  Channels: " << image.channels << std::endl;
        std::cout << "  Data size: " << image.data.size() << " bytes" << std::endl;

        // Access pixel data directly (from Rust Vec - no copy!)
        std::cout << "\n  First 10 bytes: ";
        for (size_t i = 0; i < std::min(size_t(10), image.data.size()); ++i) {
            std::cout << static_cast<int>(image.data[i]) << " ";
        }
        std::cout << std::endl;

        // Pixel access example (row-major, interleaved channels)
        size_t idx = (0 * image.width + 0) * image.channels + 0; // pixel (0,0), channel R
        std::cout << "  Pixel (0,0) R channel: " << static_cast<int>(image.data[idx]) << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
