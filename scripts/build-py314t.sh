#!/bin/bash
set -euo pipefail

# Build kornia-py in the pixi py314t environment.
#
# The py314t env mixes conda-forge (rust, uv) with a uv-provided
# Python 3.14t venv (conda-forge doesn't ship 3.14t yet). Two issues:
#
# 1. conda-forge's rust package sets CARGO_BUILD_TARGET and
#    CARGO_TARGET_*_LINKER during activation, pushing maturin into its
#    cross-compile code path.
# 2. A conda-forge maturin can't introspect a uv-provided 3.14t
#    interpreter (prints "Failed to determine python platform"), which
#    also triggers its cross-compile code path. That path then rejects
#    the interpreter with "Unsupported Python interpreter for
#    cross-compilation".
#
# Fix: don't rely on the pixi activation hook at all (it can be skipped
# when the env is restored from cache). Create the venv here if missing,
# install maturin *inside* that venv, then invoke it as
# `$PYTHON -m maturin`. It runs under the same interpreter it is
# building for, so platform introspection is trivial. Also strip the
# stale rust cross-compile env vars for belt-and-braces.

VENV_DIR="${VIRTUAL_ENV:-${CONDA_PREFIX}/../py314t-venv}"

if [ ! -x "${VENV_DIR}/bin/python" ]; then
    echo "Creating Python 3.14t virtual environment at ${VENV_DIR}..."
    rm -rf "${VENV_DIR}"
    uv venv -p 3.14t "${VENV_DIR}"
fi

PYTHON="${VENV_DIR}/bin/python"

unset CARGO_BUILD_TARGET
while IFS='=' read -r key _; do
    case "$key" in
        CARGO_TARGET_*_LINKER) unset "$key" ;;
    esac
done < <(env)

uv pip install --python "$PYTHON" "maturin>=1.9"
exec "$PYTHON" -m maturin develop --uv -m Cargo.toml --extras dev
