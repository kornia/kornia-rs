#!/usr/bin/env bash
# Jetson Orin: run the cuda-oxide parity tests, then the interleaved A/B bench.
#
#   scripts/cuda-oxide/orin-test-bench.sh           # tests + bench
#   scripts/cuda-oxide/orin-test-bench.sh --no-bench
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}"
export PATH="/usr/local/cuda/bin:$PATH"
FEATURES=cuda,cuda-oxide
DEVFREQ=/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu
bench=1
[[ "${1:-}" == "--no-bench" ]] && bench=0

die() { echo "orin-test-bench: $*" >&2; exit 1; }

echo "== system"
nvpmodel -q 2>/dev/null | head -2 || true
if [[ -r "$DEVFREQ/cur_freq" ]]; then
  cur="$(cat "$DEVFREQ/cur_freq")" min="$(cat "$DEVFREQ/min_freq")" max="$(cat "$DEVFREQ/max_freq")"
  lock="UNLOCKED (DVFS)"; [[ "$min" == "$max" ]] && lock="locked"
  echo "GPU clock: cur=$cur min=$min max=$max -> $lock"
fi
nvcc --version 2>/dev/null | tail -1 || true
head -1 /proc/driver/nvidia/version 2>/dev/null || true

echo "== tests"
log="$(mktemp)"
(cd "$ROOT" && cargo test --release -p kornia-imgproc --features "$FEATURES" --lib \
  -- oxide gray_from_rgb_u8 --test-threads=1 2>&1 | tee "$log")
# A target with unmet features is skipped without a word; check names, not "ok".
# The PTX gate runs in build.rs, so reaching these tests means it passed.
for t in gray_from_rgb_u8_oxide_bit_exact gray_from_rgb_u8_bit_exact_vs_cpu \
         corrupt_ptx_errors_instead_of_panicking; do
  grep -q "^test .*::$t \.\.\. ok$" "$log" || die "test $t did not run and pass"
done
rm -f "$log"
echo "orin-test-bench: all named parity tests ran and passed"

if (( bench )); then
  echo "== bench"
  (cd "$ROOT" && cargo run --release -p kornia-imgproc --example bench_cuda_oxide --features "$FEATURES")
fi
