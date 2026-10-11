#!/usr/bin/env bash
# Regenerate crates/kornia-imgproc/src/cuda/oxide/kernels.ptx from kernels-oxide/
# and run the PTX gate. Idempotent: the build is deterministic, so a clean tree
# stays clean.
#
#   scripts/cuda-oxide/build-ptx.sh            # rebuild, copy, gate
#   scripts/cuda-oxide/build-ptx.sh --check    # rebuild, fail on drift vs committed PTX
#   scripts/cuda-oxide/build-ptx.sh --no-test  # skip the cargo PTX-gate tests
#
# Needs the toolchain in docs/cuda-oxide.md. CI cannot run this (no backend).
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
CRATE="$ROOT/kernels-oxide"
COMMITTED="$ROOT/crates/kornia-imgproc/src/cuda/oxide/kernels.ptx"
EMITTED="$CRATE/kornia_kernels_oxide.ptx"
ARCH="${CUDA_OXIDE_ARCH:-sm_75}"
BACKEND_DIR="${CUDA_OXIDE_HOME:-$HOME/.cargo/cuda-oxide}"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}"
export PATH="/usr/local/cuda/bin:$PATH"

check=0 test=1
for arg in "$@"; do
  case "$arg" in
    --check) check=1 ;;
    --no-test) test=0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

die() { echo "build-ptx: $*" >&2; exit 1; }

# -- toolchain preflight: fail loudly, never fall back -----------------------
command -v cargo-oxide >/dev/null || die "cargo-oxide not on PATH (docs/cuda-oxide.md#toolchain)"
[[ -f "$BACKEND_DIR/librustc_codegen_cuda.so" ]] \
  || die "codegen backend missing: $BACKEND_DIR/librustc_codegen_cuda.so"
command -v llc-22 >/dev/null || command -v llc-21 >/dev/null || [[ -n "${CUDA_OXIDE_LLC:-}" ]] \
  || die "llc-21 or llc-22 required (or set CUDA_OXIDE_LLC)"
TOOLCHAIN="$(sed -n 's/^channel *= *"\(.*\)"/\1/p' "$CRATE/rust-toolchain.toml")"
rustup run "$TOOLCHAIN" rustc --version >/dev/null 2>&1 \
  || die "toolchain $TOOLCHAIN not installed: rustup toolchain install $TOOLCHAIN -c rust-src,rustc-dev"

# The backend must be built from the same cuda-oxide rev the crate depends on.
PINNED="$(sed -n 's/.*rev = "\([0-9a-f]*\)".*/\1/p' "$CRATE/Cargo.toml" | sort -u)"
[[ "$(wc -l <<<"$PINNED")" == 1 ]] || die "cuda-device/cuda-host revs differ in kernels-oxide/Cargo.toml"
if [[ -d "$BACKEND_DIR/src/.git" ]]; then
  HAVE="$(git -C "$BACKEND_DIR/src" rev-parse HEAD)"
  [[ "$HAVE" == "$PINNED" ]] || die "backend built from $HAVE, crate pins $PINNED"
else
  echo "build-ptx: warning: cannot verify backend rev (no $BACKEND_DIR/src/.git)" >&2
fi

# -- build ---------------------------------------------------------------------
rm -f "$EMITTED"
# Own target dir: never mix nightly-backend artifacts into the workspace's.
(cd "$CRATE" && CARGO_TARGET_DIR="${CUDA_OXIDE_TARGET_DIR:-$CRATE/target}" \
  cargo oxide build kornia_kernels_oxide --arch "$ARCH")
[[ -s "$EMITTED" ]] || die "cargo oxide produced no PTX at $EMITTED"

# -- fast textual gate (the cargo tests below are the authoritative copy) -----
grep -qx ".target $ARCH" "$EMITTED" || die "PTX .target is not $ARCH"
! grep -q 'fma\.' "$EMITTED" || die "PTX contains fma (breaks CPU byte parity)"
! grep -Eq '^\s*(@%p[0-9]+\s+)?(ld|st)\.(b|u|s|f|v)[0-9]' "$EMITTED" \
  || die "PTX has generic (non-global) loads/stores"

if (( check )); then
  if ! diff -u "$COMMITTED" "$EMITTED"; then
    die "committed kernels.ptx is stale: run scripts/cuda-oxide/build-ptx.sh and commit"
  fi
  echo "build-ptx: committed PTX matches kernels-oxide/ source"
else
  cp "$EMITTED" "$COMMITTED"
  echo "build-ptx: wrote $COMMITTED"
fi

if (( test )); then
  log="$(mktemp)"
  (cd "$ROOT" && cargo test --release -p kornia-imgproc --features cuda,cuda-oxide --lib -- ptx_gate 2>&1 | tee "$log")
  # cargo skips gated targets silently; prove the gate actually ran.
  ran="$(grep -c '^test cuda::oxide::tests::ptx_gate_.* ok$' "$log" || true)"
  rm -f "$log"
  (( ran >= 4 )) || die "expected >= 4 ptx_gate tests to pass, saw $ran"
  echo "build-ptx: $ran PTX gate tests passed"
fi
