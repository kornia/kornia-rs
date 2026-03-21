#!/bin/bash
set -euo pipefail

# Build script for Python 3.14t (free-threaded) environment.
#
# maturin 1.10.x cannot reliably detect the platform for uv-provided
# Python 3.14t (free-threaded), causing it to enter its cross-compilation
# code path and reject the interpreter. Using `maturin build` with an
# explicit --interpreter flag followed by pip install avoids the
# `maturin develop` interpreter discovery entirely.

# Clear cargo cross-compilation variables set by conda-forge rust
unset CARGO_BUILD_TARGET 2>/dev/null || true
while IFS='=' read -r key _; do
    case "$key" in
        CARGO_TARGET_*_LINKER) unset "$key" ;;
    esac
done < <(env)

# Pin the interpreter explicitly
PYTHON="${VIRTUAL_ENV}/bin/python"
export PYO3_PYTHON="$PYTHON"

# Build wheel with explicit interpreter, then install it
maturin build -m Cargo.toml -i "$PYTHON"
"$PYTHON" -m pip install --force-reinstall --no-deps target/wheels/*.whl
