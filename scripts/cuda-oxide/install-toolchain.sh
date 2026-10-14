#!/usr/bin/env bash
# Install what kornia-imgproc's `cuda` feature needs to compile its Rust CUDA
# kernels (kernels-oxide/) with cuda-oxide, into PREFIX:
#   - the nightly pinned in kernels-oxide/rust-toolchain.toml (via rustup);
#   - cuda.h + the libcuda link stub (NVIDIA redist), unless $CUDA_HOME has them;
#   - LLVM 21 `llc` + libclang from conda-forge (glibc 2.17, so it also runs in
#     manylinux containers), unless llc-21/22 is on PATH;
#   - the codegen backend built at the rev pinned in kernels-oxide/Cargo.toml.in.
# Pieces already in PREFIX at the right version are reused (cache PREFIX in CI).
#
#   scripts/cuda-oxide/install-toolchain.sh [PREFIX]   # default ~/.cargo/cuda-oxide
#   . PREFIX/env.sh                                    # then cargo build --features cuda
#
# OXIDE_USE_SYSTEM=0 ignores system cuda.h/llc (CI: same recipe everywhere).
# With $GITHUB_ENV writable, the variables are also exported to later CI steps.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
KERNELS="$ROOT/crates/kornia-imgproc/kernels-oxide"
TC="$(sed -n 's/^channel *= *"\(.*\)"/\1/p' "$KERNELS/rust-toolchain.toml")"
REV="$(sed -n 's/.*rev = "\([0-9a-f]*\)".*/\1/p' "$KERNELS/Cargo.toml.in" | head -1)"
LLVM_VERSION="${LLVM_VERSION:-21.1.8}"
CUDART_VERSION="${CUDART_VERSION:-12.6.77}"
PREFIX="${1:-${CARGO_HOME:-$HOME/.cargo}/cuda-oxide}"
JOBS="${CARGO_BUILD_JOBS:-2}"
USE_SYSTEM="${OXIDE_USE_SYSTEM:-1}"

die() { echo "install-toolchain: $*" >&2; exit 1; }
log() { echo "install-toolchain: $*"; }
[[ -n "$TC" && -n "$REV" ]] || die "cannot read the pins under $KERNELS"
case "$(uname -m)" in
  x86_64) CONDA_ARCH=linux-64 CUDA_ARCH=linux-x86_64 ;;
  aarch64 | arm64) CONDA_ARCH=linux-aarch64 CUDA_ARCH=linux-sbsa ;;
  *) die "unsupported host $(uname -m)" ;;
esac
mkdir -p "$PREFIX"
PREFIX="$(cd "$PREFIX" && pwd)"

fetch() { curl -fsSL --retry 3 -o "$2" "$1"; }

# -- rust: rustup + the pinned nightly -----------------------------------------
if ! command -v rustup >/dev/null; then
  log "installing rustup"
  fetch https://sh.rustup.rs "$PREFIX/rustup-init.sh"
  sh "$PREFIX/rustup-init.sh" -y --profile minimal --default-toolchain none
  rm -f "$PREFIX/rustup-init.sh"
  # shellcheck disable=SC1091
  . "${CARGO_HOME:-$HOME/.cargo}/env"
fi
rustup toolchain install "$TC" --profile minimal -c rust-src,rustc-dev
SYSROOT="$(rustup run "$TC" rustc --print sysroot)"

# -- cuda.h + libcuda stub ------------------------------------------------------
SYS_CUDA="${CUDA_HOME:-/usr/local/cuda}"
CUDA_ENV=""
if [[ "$USE_SYSTEM" == 1 && -f "$SYS_CUDA/include/cuda.h" ]]; then
  log "cuda.h: $SYS_CUDA"
  CUDA_ENV="export CUDA_HOME=\"$SYS_CUDA\""
else
  if [[ ! -f "$PREFIX/cuda/include/cuda.h" || "$(cat "$PREFIX/cuda/.version" 2>/dev/null)" != "$CUDART_VERSION" ]]; then
    log "fetching cuda_cudart $CUDART_VERSION ($CUDA_ARCH)"
    rm -rf "$PREFIX/cuda" && mkdir -p "$PREFIX/cuda"
    fetch "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/$CUDA_ARCH/cuda_cudart-$CUDA_ARCH-$CUDART_VERSION-archive.tar.xz" "$PREFIX/cudart.tar.xz"
    tar -xJf "$PREFIX/cudart.tar.xz" -C "$PREFIX/cuda" --strip-components=1
    rm -f "$PREFIX/cudart.tar.xz"
    echo "$CUDART_VERSION" >"$PREFIX/cuda/.version"
  fi
  CUDA_ENV="export CUDA_HOME=\"\$P/cuda\"
export LIBRARY_PATH=\"\$P/cuda/lib/stubs\${LIBRARY_PATH:+:\$LIBRARY_PATH}\""
fi

# -- LLVM 21: llc (PTX codegen) + libclang (bindgen of cuda.h) -------------------
LLVM_ENV=""
if [[ "$USE_SYSTEM" == 1 ]] && { [[ -n "${CUDA_OXIDE_LLC:-}" ]] || command -v llc-22 >/dev/null || command -v llc-21 >/dev/null; }; then
  log "llc: system"
else
  if [[ ! -x "$PREFIX/llvm/bin/llc" || "$(cat "$PREFIX/llvm/.version" 2>/dev/null)" != "$LLVM_VERSION" ]]; then
    # LLVM's own release tarballs need glibc 2.34; conda-forge builds need 2.17.
    log "installing LLVM $LLVM_VERSION llc + libclang from conda-forge"
    rm -rf "$PREFIX/llvm" "$PREFIX/.mamba"
    mkdir -p "$PREFIX/.mamba"
    fetch "https://micro.mamba.pm/api/micromamba/$CONDA_ARCH/latest" "$PREFIX/.mamba/mm.tar.bz2"
    tar -xjf "$PREFIX/.mamba/mm.tar.bz2" -C "$PREFIX/.mamba" bin/micromamba
    # clang-21 brings the resource headers (stddef.h) bindgen needs.
    MAMBA_ROOT_PREFIX="$PREFIX/.mamba" "$PREFIX/.mamba/bin/micromamba" create -y -q \
      -p "$PREFIX/llvm" -c conda-forge --override-channels \
      "llvm-tools=$LLVM_VERSION" "libclang=$LLVM_VERSION" "clang-21=$LLVM_VERSION"
    rm -rf "$PREFIX/.mamba"
    "$PREFIX/llvm/bin/llc" --version >/dev/null || die "llc $LLVM_VERSION does not run on this host"
    echo "$LLVM_VERSION" >"$PREFIX/llvm/.version"
  fi
  LLVM_ENV="export CUDA_OXIDE_LLC=\"\$P/llvm/bin/llc\"
export LIBCLANG_PATH=\"\$P/llvm/lib\""
fi

# -- codegen backend at the pin --------------------------------------------------
SO="$PREFIX/librustc_codegen_cuda.so"
STAMP="$REV $TC"
src_head="$(git -C "$PREFIX/src" rev-parse HEAD 2>/dev/null || true)"
if [[ -f "$SO" && ("$(cat "$PREFIX/.backend" 2>/dev/null)" == "$STAMP" || "$src_head" == "$REV") ]]; then
  log "backend: cached ($REV)"
else
  log "building the cuda-oxide backend at $REV (a few minutes)"
  # A tarball, not git: manylinux2014's git is too old to fetch by sha.
  rm -rf "$PREFIX/src" && mkdir -p "$PREFIX/src"
  fetch "https://github.com/NVlabs/cuda-oxide/archive/$REV.tar.gz" "$PREFIX/src.tar.gz"
  tar -xzf "$PREFIX/src.tar.gz" -C "$PREFIX/src" --strip-components=1
  rm -f "$PREFIX/src.tar.gz"
  (
    cd "$PREFIX/src/crates/rustc-codegen-cuda"
    LIBRARY_PATH="$SYSROOT/lib" LD_LIBRARY_PATH="$SYSROOT/lib" \
      rustup run "$TC" cargo build --jobs "$JOBS" --target-dir "$PREFIX/backend-target"
  )
  cp "$PREFIX/backend-target/debug/librustc_codegen_cuda.so" "$SO"
  rm -rf "$PREFIX/backend-target" "$PREFIX/src"
  echo "$STAMP" >"$PREFIX/.backend"
fi

# -- env.sh: relocatable, so a cached PREFIX works at any path -----------------
cat >"$PREFIX/env.sh" <<EOF
# Generated by scripts/cuda-oxide/install-toolchain.sh; source before cargo.
P="\$(cd "\$(dirname "\${BASH_SOURCE[0]:-\$0}")" && pwd)"
export CUDA_OXIDE_BACKEND="\$P/librustc_codegen_cuda.so"
$CUDA_ENV
$LLVM_ENV
EOF
if [[ -n "${GITHUB_ENV:-}" && -w "${GITHUB_ENV:-/nonexistent}" ]]; then
  # shellcheck disable=SC1091
  (. "$PREFIX/env.sh" && env | grep -E '^(CUDA_OXIDE_BACKEND|CUDA_OXIDE_LLC|CUDA_HOME|LIBRARY_PATH|LIBCLANG_PATH)=') >>"$GITHUB_ENV"
fi
log "done; run: . $PREFIX/env.sh"
