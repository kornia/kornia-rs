#!/bin/bash
# Activation script for Rust feature
# Sets up environment variables for native library dependencies

export PKG_CONFIG_PATH="${CONDA_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH}"

# OpenSSL configuration - use conda's openssl instead of system
export OPENSSL_DIR="${CONDA_PREFIX}"
export OPENSSL_INCLUDE_DIR="${CONDA_PREFIX}/include"
export OPENSSL_LIB_DIR="${CONDA_PREFIX}/lib"
