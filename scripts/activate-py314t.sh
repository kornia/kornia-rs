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
