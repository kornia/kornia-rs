#!/usr/bin/env bash
# Jetson: run the cuda-oxide parity tests, then the bench (oxide vs CPU).
#
#   scripts/cuda-oxide/orin-test-bench.sh                    # tests + bench
#   scripts/cuda-oxide/orin-test-bench.sh --no-bench
#   scripts/cuda-oxide/orin-test-bench.sh --base origin/main # + A/B vs <ref>
#
# --base builds the same bench (benches/bench_cuda_oxide.rs) at <ref> (e.g.
# main's NVRTC kernel) in its own worktree and target dir, then alternates the
# two binaries per cycle.
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}"
export PATH="/usr/local/cuda/bin:$PATH"
# Absolute: cargo resolves a relative CARGO_TARGET_DIR against its cwd, and the
# --base build runs from another worktree.
TARGET="$(cd "$ROOT" && realpath -m "${CARGO_TARGET_DIR:-target}")"
export CARGO_TARGET_DIR="$TARGET"
DEVFREQ=/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu
BENCH=crates/kornia-imgproc/benches/bench_cuda_oxide.rs
bench=1 base="" cycles=3

die() { echo "orin-test-bench: $*" >&2; exit 1; }
# Builds the bench in the cwd's checkout; prints its executable.
build_bench() {
  cargo bench --no-run --message-format=json -p kornia-imgproc --bench bench_cuda_oxide --features cuda |
    sed -n 's/.*"executable":"\([^"]*bench_cuda_oxide[^"]*\)".*/\1/p' | tail -1
}
while (($#)); do
  case "$1" in
    --no-bench) bench=0 ;;
    --base) base="${2:?--base needs a ref}"; shift ;;
    --cycles) cycles="${2:?}"; shift ;;
    *) die "unknown argument $1" ;;
  esac
  shift
done

echo "== system"
nvpmodel -q 2>/dev/null | head -2 || true
if [[ -r "$DEVFREQ/cur_freq" ]]; then
  min="$(cat "$DEVFREQ/min_freq")" max="$(cat "$DEVFREQ/max_freq")"
  lock="UNLOCKED (DVFS)"; [[ "$min" == "$max" ]] && lock="locked"
  echo "GPU clock: cur=$(cat "$DEVFREQ/cur_freq") min=$min max=$max -> $lock"
fi
nvcc --version 2>/dev/null | tail -1 || true
head -1 /proc/driver/nvidia/version 2>/dev/null || true
git -C "$ROOT" log -1 --format='HEAD %H %s'

echo "== tests"
cd "$ROOT"
cargo run -q -p kornia-oxide-kernels --example build_ptx
ptx="$ROOT/crates/kornia-oxide-kernels/kernels.ptx"
# Entry names come from the PTX just built, never a typed list.
mapfile -t entries < <(sed -n 's/.*\.entry \([A-Za-z0-9_]*\).*/\1/p' "$ptx")
((${#entries[@]})) || die "no .entry in $ptx"
echo "entries: ${entries[*]}"
log="$(mktemp)"
cargo test --release -p kornia-imgproc --features cuda --lib -- "${entries[@]}" \
  --test-threads=1 2>&1 | tee "$log"
# The loader lives in kornia-tensor; `ptx` makes its tests load the real PTX.
cargo test --release -p kornia-tensor --features cuda,kornia-oxide-kernels/ptx --lib -- \
  cuda::oxide 2>&1 | tee -a "$log"
# cargo skips targets with unmet features silently: check names, not "ok".
for t in all_entries_load concurrent_calls_share_one_module; do
  grep -q "^test cuda::oxide::tests::$t \.\.\. ok$" "$log" || die "$t did not pass"
done
for e in "${entries[@]}"; do
  grep -Eq "^test .*::${e}[a-z0-9_]*bit_exact \.\.\. ok$" "$log" || die "no passing ${e}*bit_exact test"
done
grep -q "test result: FAILED" "$log" && die "tests failed"
grep -Eq "test result: ok\. [1-9][0-9]* passed; 0 failed" "$log" || die "no tests ran"
rm -f "$log"
echo "orin-test-bench: parity tests passed for ${#entries[@]} entries"

((bench)) || exit 0
echo "== bench"
head_bin="$(build_bench)"
[[ -x "$head_bin" ]] || die "bench build produced no executable"
if [[ -z "$base" ]]; then
  "$head_bin"
  exit 0
fi

wt="$(mktemp -d "$TARGET-base-src.XXXX")"
trap 'git -C "$ROOT" worktree remove --force "$wt" 2>/dev/null || true' EXIT
git -C "$ROOT" worktree add -q --detach "$wt" "$base"
# A base with oxide kernels needs its own PTX (main's NVRTC kernels do not).
if [[ -f "$wt/crates/kornia-oxide-kernels/examples/build_ptx.rs" ]]; then
  (cd "$wt" && CARGO_TARGET_DIR="$TARGET-base" cargo run -q -p kornia-oxide-kernels --example build_ptx)
fi
cp "$ROOT/$BENCH" "$wt/$BENCH" # the bench calls only public launchers
manifest="$wt/crates/kornia-imgproc/Cargo.toml"
grep -A2 '^\[\[bench\]\]' "$manifest" | grep -q '^name = "bench_cuda_oxide"$' ||
  printf '\n[[bench]]\nharness = false\nname = "bench_cuda_oxide"\nrequired-features = ["cuda"]\n' >>"$manifest"
base_bin="$(cd "$wt" && CARGO_TARGET_DIR="$TARGET-base" build_bench)"
[[ -x "$base_bin" ]] || die "bench build at $base produced no executable"
# Provenance (A/A trap): which commit and binary each arm runs.
dirty="$(git -C "$ROOT" status --short --untracked-files=no | wc -l)"
echo "head arm: $(git -C "$ROOT" rev-parse HEAD) (+$dirty modified files) $head_bin"
echo "base arm: $(git -C "$wt" rev-parse HEAD) ($base) $base_bin"
for ((c = 0; c < cycles; c++)); do
  order=(base head)
  ((c % 2)) && order=(head base)
  for arm in "${order[@]}"; do
    bin="$head_bin" ref=HEAD
    [[ "$arm" == base ]] && bin="$base_bin" ref="$base"
    echo "-- cycle $c: $arm ($ref)"
    "$bin" | sed -n "s/^| \([a-z]\)/| $arm | \1/p"
  done
done
