// Native CPU reference worker for bench_adaptive_dataset.py.
// Image loading, command transport and mask writes are outside timed regions.
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgproc.hpp>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

/// Serve native OpenCV warmup, timing and mask requests without timing IPC.
int main(int argc, char** argv) {
    try {
        cv::setNumThreads(1);
        cv::ocl::setUseOpenCL(false);
        if (argc == 2) {
            const std::string option = argv[1];
            if (option == "--info") {
                std::cout << cv::getBuildInformation();
            } else if (option == "--version") {
                std::cout << CV_VERSION << '\n';
            } else if (option == "--threads") {
                std::cout << cv::getNumThreads() << '\n';
            } else {
                throw std::runtime_error("unknown option");
            }
            return 0;
        }
        if (argc != 6 || std::string(argv[5]) != "--server") {
            throw std::runtime_error("usage: worker INPUT.raw WIDTH HEIGHT MASK_DIRECTORY --server");
        }
        const int width = std::stoi(argv[2]);
        const int height = std::stoi(argv[3]);
        if (width <= 0 || height <= 0) {
            throw std::runtime_error("dimensions must be positive");
        }
        std::ifstream input(argv[1], std::ios::binary);
        if (!input) {
            throw std::runtime_error("cannot open input");
        }
        std::vector<unsigned char> pixels((std::istreambuf_iterator<char>(input)), {});
        if (pixels.size() != static_cast<size_t>(width) * height) {
            throw std::runtime_error("input byte count does not match dimensions");
        }
        const cv::Mat source(height, width, CV_8UC1, pixels.data());
        cv::Mat destination(height, width, CV_8UC1);
        const std::filesystem::path output = argv[4];
        std::filesystem::create_directories(output);
        using Clock = std::chrono::steady_clock;
        const auto apply = [&](int window) {
            cv::adaptiveThreshold(source, destination, 255, cv::ADAPTIVE_THRESH_MEAN_C,
                                  cv::THRESH_BINARY, window, 10);
            std::atomic_signal_fence(std::memory_order_seq_cst);
        };
        std::string command;
        int window;
        while (std::cin >> command >> window) {
            if (command == "W") {
                const auto start = Clock::now();
                size_t calls = 0;
                do {
                    apply(window);
                    ++calls;
                } while (calls < 10 || Clock::now() - start < std::chrono::milliseconds(250));
                std::cout << "ok\n";
            } else if (command == "R") {
                int iterations;
                if (!(std::cin >> iterations) || iterations <= 0) {
                    throw std::runtime_error("iterations must be positive");
                }
                const auto start = Clock::now();
                for (int i = 0; i < iterations; ++i) {
                    apply(window);
                }
                const auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start);
                std::cout << std::fixed << std::setprecision(9) << elapsed.count() / iterations << '\n';
            } else if (command == "M") {
                apply(window);
                std::ofstream mask(output / (std::to_string(window) + ".raw"), std::ios::binary);
                mask.write(reinterpret_cast<const char*>(destination.data), pixels.size());
                if (!mask) {
                    throw std::runtime_error("mask write failed");
                }
                std::cout << "ok\n";
            } else {
                throw std::runtime_error("unknown command");
            }
            std::cout.flush();
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
