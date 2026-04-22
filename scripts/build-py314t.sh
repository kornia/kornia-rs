#!/bin/bash
set -euo pipefail

# Build kornia-py in the pixi py314t environment.
#
# Even with maturin >= 1.9 (which supports free-threaded CPython 3.14t
# natively), conda-forge's rust package sets CARGO_BUILD_TARGET and
# CARGO_TARGET_*_LINKER during pixi activation. Those env vars push
# maturin into its cross-compile code path and it then rejects the
# uv-provided 3.14t interpreter with:
#   "Unsupported Python interpreter for cross-compilation"
#
# Unsetting them in the activation script is not sufficient because
# pixi may re-apply rust-package activation after our hook runs.
# Strip them in the same process that invokes maturin so they stay gone.

unset CARGO_BUILD_TARGET
while IFS='=' read -r key _; do
    case "$key" in
        CARGO_TARGET_*_LINKER) unset "$key" ;;
    esac
done < <(env)

exec maturin develop --uv -m Cargo.toml --extras dev
