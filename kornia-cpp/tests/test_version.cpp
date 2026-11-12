#include <catch2/catch_test_macros.hpp>
#include <kornia.hpp>
#include <string>

// The numeric parts must be integer literals: usable in #if and as int.
static_assert(KORNIA_VERSION_MAJOR >= 0 && KORNIA_VERSION_MINOR >= 0 && KORNIA_VERSION_PATCH >= 0,
              "version parts must be integers");
static_assert(KORNIA_VERSION_IS_PRERELEASE == 0 || KORNIA_VERSION_IS_PRERELEASE == 1,
              "IS_PRERELEASE must be 0 or 1");

TEST_CASE("Library Version", "[version]") {
    const char* version = kornia::version();
    REQUIRE(version != nullptr);
    REQUIRE(std::string(version).length() > 0);
}

TEST_CASE("Version parts rebuild the version string", "[version]") {
    std::string expected = std::to_string(KORNIA_VERSION_MAJOR) + "." +
                           std::to_string(KORNIA_VERSION_MINOR) + "." +
                           std::to_string(KORNIA_VERSION_PATCH);
    const std::string pre = KORNIA_VERSION_PRERELEASE;
    REQUIRE((KORNIA_VERSION_IS_PRERELEASE == 1) == !pre.empty());
    if (!pre.empty()) {
        expected += "-" + pre;
    }
    REQUIRE(std::string(kornia::version()) == expected);
}
