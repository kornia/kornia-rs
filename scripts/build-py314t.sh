#!/bin/bash
set -euo pipefail

# Build kornia-py in the pixi py314t environment.
#
# Even on maturin >= 1.9 (which supports free-threaded CPython natively),
# two problems appear inside pixi's py314t env:
#
# 1. conda-forge's rust package sets CARGO_BUILD_TARGET and
#    CARGO_TARGET_*_LINKER, pushing maturin into its cross-compile path
#    and rejecting the uv-provided interpreter.
# 2. `maturin develop` can't discover the uv-venv's Python platform
#    (prints "Failed to determine python platform") and falls back to
#    cross-compile for the same reason as (1).
#
# Workaround: strip the env vars and use `maturin build -i $PYTHON` to
# pass the interpreter explicitly, then install the resulting wheel with
# uv pip. Both bypass maturin's environment-driven discovery.

PYTHON="${VIRTUAL_ENV}/bin/python"

unset CARGO_BUILD_TARGET
while IFS='=' read -r key _; do
    case "$key" in
        CARGO_TARGET_*_LINKER) unset "$key" ;;
    esac
done < <(env)

export PYO3_PYTHON="$PYTHON"

maturin build -m Cargo.toml -i "$PYTHON"
uv pip install --python "$PYTHON" --force-reinstall --no-deps target/wheels/*.whl
