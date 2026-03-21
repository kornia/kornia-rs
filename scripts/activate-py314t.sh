#!/bin/bash
# Activation script for Python 3.14t (free-threaded) environment
# Uses uv to provide Python 3.14t since conda-forge doesn't have it yet

VENV_DIR="${CONDA_PREFIX}/../py314t-venv"

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python 3.14t virtual environment..."
    uv venv -p 3.14t "$VENV_DIR"
fi

# Override PATH to use the 3.14t Python
export PATH="$VENV_DIR/bin:$PATH"
export VIRTUAL_ENV="$VENV_DIR"

# Unset CONDA_PREFIX so maturin uses the uv venv
unset CONDA_PREFIX

# Prevent maturin from falsely detecting cross-compilation.
# The conda-forge rust package sets CARGO_BUILD_TARGET and
# CARGO_TARGET_*_LINKER vars during activation. When these remain
# set after unsetting CONDA_PREFIX, maturin enters its cross-compile
# code path and rejects the uv-provided python3.14t interpreter.
unset CARGO_BUILD_TARGET
for var in $(env | grep '^CARGO_TARGET_.*_LINKER=' | cut -d= -f1); do
    unset "$var"
done

# Pin the interpreter so maturin doesn't rely on environment discovery
export PYO3_PYTHON="$VENV_DIR/bin/python"
