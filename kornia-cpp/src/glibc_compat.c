// Compatibility shim for mixing C++ standard libraries
//
// CONTEXT:
// - kornia-cpp uses Rust + CXX bridge (C++ code compiled with libstdc++)
// - Some consuming projects may use libc++ instead of libstdc++
// - libstdc++ depends on glibc-specific symbols that may not be available
//
// SOLUTION:
// Provide weak definitions for glibc-specific symbols. These act as fallbacks
// when the real glibc isn't available. The weak attribute ensures that if the
// actual glibc is present, its strong definitions will be used instead.
//
// SYMBOLS:
// - __libc_single_threaded: Thread-safety optimization hint used by libstdc++
//   Setting to 0 (multi-threaded) is the safe default.
//
// NOTE:
// This is a pragmatic workaround for C++ stdlib mixing. If you have a better
// solution (e.g., pure C API, proper stdlib isolation), please open a PR!

#ifdef __linux__

__attribute__((weak)) char __libc_single_threaded = 0;

#endif

