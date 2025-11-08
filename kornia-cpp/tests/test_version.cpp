#include <catch2/catch_test_macros.hpp>
#include <kornia.hpp>
#include <string>

TEST_CASE("Library Version", "[version]") {
    const char* version = kornia::version();
    REQUIRE(version != nullptr);
    REQUIRE(std::string(version).length() > 0);
}

