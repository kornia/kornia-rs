#!/bin/bash
set -euo pipefail

# Build script for Python 3.14t (free-threaded) environment.
#
# maturin 1.10.x can falsely detect cross-compilation when the conda-forge
# rust package's activation scripts set CARGO_BUILD_TARGET or
# CARGO_TARGET_*_LINKER variables. This causes it to reject the uv-provided
# Python 3.14t interpreter. We clear these variables and pin PYO3_PYTHON
# before invoking maturin to ensure a native build.

# Clear cargo cross-compilation variables set by conda-forge rust
unset CARGO_BUILD_TARGET 2>/dev/null || true
while IFS='=' read -r key _; do
    case "$key" in
        CARGO_TARGET_*_LINKER) unset "$key" ;;
    esac
done < <(env)

# Pin the interpreter explicitly
export PYO3_PYTHON="${VIRTUAL_ENV}/bin/python"

exec maturin develop -m Cargo.toml --extras dev
