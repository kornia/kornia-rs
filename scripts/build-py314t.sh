#!/bin/bash
set -euo pipefail

# Build script for Python 3.14t (free-threaded) environment.
#
# maturin 1.10.x cannot detect the platform for uv-provided Python 3.14t
# when running inside a pixi/conda environment. The conda-forge rust
# package and pixi set variables that cause maturin to enter its
# cross-compilation code path and reject the interpreter.
#
# We strip all conda/cargo cross-compile state, then use maturin build
# with an explicit interpreter flag to bypass environment discovery.

PYTHON="${VIRTUAL_ENV}/bin/python"

# Clear ALL environment state that could confuse maturin
unset CARGO_BUILD_TARGET 2>/dev/null || true
unset CONDA_PREFIX 2>/dev/null || true
unset VIRTUAL_ENV 2>/dev/null || true
while IFS='=' read -r key _; do
    case "$key" in
        CARGO_TARGET_*_LINKER) unset "$key" ;;
        _CONDA_SET_*) unset "$key" ;;
        CONDA_*) unset "$key" ;;
    esac
done < <(env)

export PYO3_PYTHON="$PYTHON"

# Use maturin build with explicit interpreter (not develop, which lacks
# --interpreter support) and install the resulting wheel
maturin build -m Cargo.toml -i "$PYTHON"
uv pip install --python "$PYTHON" --force-reinstall --no-deps target/wheels/*.whl
