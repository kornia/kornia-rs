#!/usr/bin/env bash
# Install what kornia-imgproc's `cuda` feature needs to compile its Rust CUDA
# kernels (kernels-oxide/) with cuda-oxide, into PREFIX:
#   - the nightly pinned in kernels-oxide/rust-toolchain.toml (via rustup);
#   - cuda.h + the libcuda link stub (NVIDIA redist), unless $CUDA_HOME has them;
#   - LLVM 21 `llc` + libclang from conda-forge, locked in llvm-<arch>.txt
#     (glibc 2.17, so it also runs in manylinux containers), unless llc-21/22
#     is on PATH;
#   - the codegen backend built at the rev pinned in kernels-oxide/Cargo.toml.in.
# Pieces already in PREFIX at the right version are reused (cache PREFIX in CI).
# Every download is pinned and sha256-checked, except rustup and the nightly.
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
# sha256 of github.com/NVlabs/cuda-oxide/archive/$REV.tar.gz; bump with the rev.
BACKEND_SHA=e0575a5072ae4bc7c471b2177460f701544de0edf76057eaec0b8dbc1b20f257
CUDART_VERSION=12.6.77
MICROMAMBA_VERSION=2.9.0
PREFIX="${1:-${CARGO_HOME:-$HOME/.cargo}/cuda-oxide}"
JOBS="${CARGO_BUILD_JOBS:-2}"
USE_SYSTEM="${OXIDE_USE_SYSTEM:-1}"

die() { echo "install-toolchain: $*" >&2; exit 1; }
log() { echo "install-toolchain: $*"; }
[[ -n "$TC" && -n "$REV" ]] || die "cannot read the pins under $KERNELS"
# sha256s: NVIDIA's redistrib_12.6.2.json and conda-forge's micromamba files.
case "$(uname -m)" in
  x86_64)
    CONDA_ARCH=linux-64 CUDA_ARCH=linux-x86_64
    CUDART_SHA=f74689258a60fd9c5bdfa7679458527a55e22442691ba678dcfaeffbf4391ef9
    MICROMAMBA_SHA=8761c382127e6363bd9e0a2451aa3ef90d071a79133f736e2f759a3bf13040dd
    ;;
  aarch64 | arm64)
    CONDA_ARCH=linux-aarch64 CUDA_ARCH=linux-sbsa
    CUDART_SHA=c73c8e5bfe8fcd7468d012c9eebff15063005a3bba44423d541d573dc058de58
    MICROMAMBA_SHA=e705ffeed90ce0659eb546e4b1e1028c9eaf0bc9cc854867b19ac5ce0ba5852f
    ;;
  *) die "unsupported host $(uname -m)" ;;
esac
LLVM_LOCK="$ROOT/scripts/cuda-oxide/llvm-$CONDA_ARCH.txt"
LLVM_STAMP="$(sha256sum "$LLVM_LOCK" | cut -d' ' -f1)"
mkdir -p "$PREFIX"
PREFIX="$(cd "$PREFIX" && pwd)"

fetch() { curl -fsSL --retry 3 -o "$2" "$1"; }
# fetch_sha URL FILE SHA256
fetch_sha() {
  fetch "$1" "$2"
  echo "$3  $2" | sha256sum -c --quiet - || die "sha256 mismatch for $1"
}

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
    fetch_sha "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/$CUDA_ARCH/cuda_cudart-$CUDA_ARCH-$CUDART_VERSION-archive.tar.xz" \
      "$PREFIX/cudart.tar.xz" "$CUDART_SHA"
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
  if [[ ! -x "$PREFIX/llvm/bin/llc" || "$(cat "$PREFIX/llvm/.version" 2>/dev/null)" != "$LLVM_STAMP" ]]; then
    # LLVM's own release tarballs need glibc 2.34; conda-forge builds need 2.17.
    # The lock (every package sha256-pinned) was generated with:
    #   micromamba create --dry-run --json --platform <arch> -c conda-forge \
    #     llvm-tools=21.1.8 libclang=21.1.8 clang-21=21.1.8
    # (clang-21 brings the resource headers bindgen needs), url#sha256 per line.
    log "installing LLVM llc + libclang from conda-forge ($LLVM_LOCK)"
    rm -rf "$PREFIX/llvm" "$PREFIX/.mamba"
    mkdir -p "$PREFIX/.mamba"
    fetch_sha "https://conda.anaconda.org/conda-forge/$CONDA_ARCH/micromamba-$MICROMAMBA_VERSION-0.tar.bz2" \
      "$PREFIX/.mamba/mm.tar.bz2" "$MICROMAMBA_SHA"
    tar -xjf "$PREFIX/.mamba/mm.tar.bz2" -C "$PREFIX/.mamba" bin/micromamba
    MAMBA_ROOT_PREFIX="$PREFIX/.mamba" "$PREFIX/.mamba/bin/micromamba" create -y -q \
      -p "$PREFIX/llvm" --file "$LLVM_LOCK"
    rm -rf "$PREFIX/.mamba"
    "$PREFIX/llvm/bin/llc" --version >/dev/null || die "the locked llc does not run on this host"
    echo "$LLVM_STAMP" >"$PREFIX/llvm/.version"
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
  # src/ as a git checkout is a hand-built backend: never delete it.
  [[ -d "$PREFIX/src/.git" ]] &&
    die "$PREFIX/src is a cuda-oxide checkout at ${src_head:-?}; check out $REV and rebuild it, or pick another PREFIX"
  log "building the cuda-oxide backend at $REV (a few minutes)"
  # A tarball, not git: manylinux2014's git is too old to fetch by sha.
  rm -rf "$PREFIX/src" && mkdir -p "$PREFIX/src"
  fetch_sha "https://github.com/NVlabs/cuda-oxide/archive/$REV.tar.gz" "$PREFIX/src.tar.gz" "$BACKEND_SHA"
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
