#!/bin/bash
set -euo pipefail

# Build kornia-py in the pixi py314t environment.
#
# The py314t env mixes conda-forge (maturin, rust) with a uv-provided
# Python 3.14t venv. Two issues make the conda-forge maturin unusable:
#
# 1. conda-forge's rust package sets CARGO_BUILD_TARGET and
#    CARGO_TARGET_*_LINKER during activation, pushing maturin into its
#    cross-compile code path.
# 2. The conda-forge maturin can't introspect the uv-provided 3.14t
#    interpreter (prints "Failed to determine python platform"), which
#    also triggers its cross-compile code path. That path then rejects
#    the interpreter with "Unsupported Python interpreter for
#    cross-compilation".
#
# Fix: install maturin *inside* the uv venv and invoke it as
# `$PYTHON -m maturin`. It then runs under the same interpreter it is
# building for, so platform introspection is trivial. Also strip the
# stale rust cross-compile env vars for belt-and-braces.

PYTHON="${VIRTUAL_ENV}/bin/python"

unset CARGO_BUILD_TARGET
while IFS='=' read -r key _; do
    case "$key" in
        CARGO_TARGET_*_LINKER) unset "$key" ;;
    esac
done < <(env)

uv pip install --python "$PYTHON" "maturin>=1.9"
exec "$PYTHON" -m maturin develop --uv -m Cargo.toml --extras dev
