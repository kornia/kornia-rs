#!/usr/bin/env bash
# verify_simd_kernels.sh — assert each NEON/AVX2 kernel reaches the wheel.
#
# Usage: verify_simd_kernels.sh <wheel.whl>
#
# Why per-kernel: a wheel that contains *some* SIMD doesn't tell us each
# named kernel was actually compiled with intrinsics. A `cfg` typo or a
# missing `target_feature` annotation can silently route one kernel to
# scalar while siblings keep their SIMD. This script checks each kernel's
# call path individually.
#
# How it works (per kernel manifest entry "path~asm_pattern"):
#   1. Search every demangled function header for one whose name contains
#      `path` as a substring (matches the public dispatcher, every
#      monomorphization, and rayon closures whose mangled form mentions
#      the kernel).
#   2. Walk the matching function's body (until the next blank line) and
#      assert at least one body matches `asm_pattern`. Pass = at least one
#      such body has SIMD instructions.
#
# Coverage limits: kernels marked `#[inline]` and called from a single
# site get fully inlined into their caller and lose their standalone
# symbol. For those we either pin a sibling symbol (e.g. `hflip_rgb_u8`
# instead of the inner `_neon`) or fall back to the global sanity check
# at the end of the script (any kornia_imgproc-rooted body with SIMD).
#
# Cross-build LLVM (manylinux2014 container) inlines more aggressively
# than native rustc, so symbol-substring search may miss kernels whose
# bodies got hoisted into a rayon closure with a different mangled name.
# `GLOBAL_FLOOR` is the regression-to-zero alarm: a wholesale SIMD strip
# (RUSTFLAGS clobber, `target_feature` annotations dropped) drives the
# count to single digits and trips here.
set -euo pipefail

GLOBAL_FLOOR=50

WHEEL="${1:?usage: $0 <wheel>}"
case "$WHEEL" in
  *manylinux*x86_64*|*linux_x86_64*) ARCH=x86_64 ;;
  *manylinux*aarch64*|*linux_aarch64*) ARCH=aarch64 ;;
  *) echo "::error::cannot infer arch from $WHEEL" >&2; exit 1 ;;
esac

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
unzip -q -o "$WHEEL" '*.so' -d "$WORK"
SO=$(find "$WORK" -name 'kornia_rs*.so' | head -n1)
[ -n "$SO" ] || { echo "::error::no kornia_rs*.so in $WHEEL" >&2; exit 1; }

if [ "$ARCH" = aarch64 ]; then
  OBJDUMP=aarch64-linux-gnu-objdump
  KERNELS=(
    'flip::hflip_rgb_u8~[[:space:]](ld3|rev64)[[:space:]]'
    'color::gray::rgb_to_gray_u8~[[:space:]](ld3|umlal)[[:space:]]'
    'normalize::normalize_rgb_u8~[[:space:]](fmla|ucvtf)[[:space:]]'
    'resize::pyramid::pyrup_~[[:space:]](ld3|urhadd)[[:space:]]'
    'resize::kernels::vertical_row_neon~[[:space:]](smlal|sqshrun)[[:space:]]'
    'features::fast::fast_block_neon~[[:space:]](uqadd|uqsub|cmhs|umaxv)[[:space:]]'
    'features::fast::fast_detect_rows_u~[[:space:]](uqadd|uqsub|cmhs)[[:space:]]'
  )
  GLOBAL_PATTERNS='[[:space:]](ld3|umlal|fmla|urhadd|smlal|sqshrun|uqadd|cmhs|udot)[[:space:]]'
  GLOBAL_LABEL='NEON'
else
  OBJDUMP=objdump
  KERNELS=(
    'flip::hflip_rgb_u8~[[:space:]](vpshufb|vextracti128)[[:space:]]'
    'color::gray::rgb_to_gray_u8~[[:space:]](vpmaddubsw|vpermq)[[:space:]]'
    'normalize::normalize_rgb_u8~[[:space:]](vfmadd|vcvtdq2ps)[[:space:]]'
    'features::fast::fast_block_avx2~[[:space:]](vpcmpeqb|vpsubusb)[[:space:]]'
    'features::fast::fast_detect_rows_u~[[:space:]](vpcmpeqb|vpsubusb)[[:space:]]'
    'features::match::hamming~[[:space:]](vpxor|vpshufb)[[:space:]]'
  )
  GLOBAL_PATTERNS='[[:space:]](vpshufb|vpmaddubsw|vfmadd|vpcmpeqb|vpsubusb|vpaddw|vpmullw|vpermq|vpxor)[[:space:]]'
  GLOBAL_LABEL='AVX2'
fi

DUMP="$WORK/dis.txt"
"$OBJDUMP" -dC --no-show-raw-insn "$SO" > "$DUMP"

# Single pass over the dump: build per-kernel pass/fail, plus the global
# kornia_imgproc SIMD count. `objdump -dC` emits one function header line
# `<addr> <demangled::path::name>:`, body lines `<addr>: <mnemonic> <ops>`,
# then a blank-line terminator. We track which kernel-pattern matched the
# current function's name and mark it `ok` the first time its asm pattern
# fires inside that body; in parallel we count SIMD lines under any
# kornia_imgproc-rooted symbol for the global floor.
RESULTS="$WORK/results.txt"
# Pass kernel manifest as a newline-joined env var; awk splits it in BEGIN.
KERNELS_JOINED=$(printf '%s\n' "${KERNELS[@]}")
export KERNELS_JOINED
awk -v gp="$GLOBAL_PATTERNS" -v out="$RESULTS" '
  BEGIN {
    n = split(ENVIRON["KERNELS_JOINED"], lines, "\n")
    for (i = 1; i <= n; i++) {
      if (lines[i] == "") continue
      split(lines[i], kv, "~")
      paths[i] = kv[1]; pats[i] = kv[2]
      sawpath[i] = 0; matched[i] = 0
      kernel_idx[++kcount] = i
    }
    in_kornia = 0
    global_hits = 0
  }
  /^[0-9a-f]+ </ {
    in_kornia = (index($0, "kornia_imgproc") > 0)
    delete current
    for (j = 1; j <= kcount; j++) {
      i = kernel_idx[j]
      if (index($0, paths[i]) > 0) { current[i] = 1; sawpath[i] = 1 }
    }
    next
  }
  /^$/ { delete current; in_kornia = 0; next }
  {
    if (in_kornia && match($0, gp)) global_hits++
    for (i in current) {
      if (!matched[i] && match($0, pats[i])) matched[i] = 1
    }
  }
  END {
    for (j = 1; j <= kcount; j++) {
      i = kernel_idx[j]
      status = (!sawpath[i]) ? "notfound" : (matched[i] ? "ok" : "noasm")
      printf "%s\t%s\t%s\n", status, paths[i], pats[i] > out
    }
    printf "global\t%d\n", global_hits > out
  }
' "$DUMP"

fail=0
ok=0
notfound=0
GLOBAL_HITS=0
while IFS=$'\t' read -r status path patt; do
  case "$status" in
    ok)       echo "ok   ${ARCH}: $path"; ok=$((ok+1)) ;;
    notfound) echo "::warning::${ARCH}: no function path '$path' present in wheel (renamed? heavily inlined?)"; notfound=$((notfound+1)) ;;
    noasm)    echo "::error::${ARCH}: '$path' present but no body matches SIMD pattern '$patt'"; fail=$((fail+1)) ;;
    global)   GLOBAL_HITS="$path" ;;
  esac
done < "$RESULTS"

if [ "$GLOBAL_HITS" -lt "$GLOBAL_FLOOR" ]; then
  echo "::error::${ARCH}: only $GLOBAL_HITS ${GLOBAL_LABEL} instructions in kornia_imgproc bodies (expected >${GLOBAL_FLOOR}). SIMD pipeline likely broken globally."
  fail=$((fail+1))
else
  echo "ok   ${ARCH}: ${GLOBAL_HITS} ${GLOBAL_LABEL} instructions inside kornia_imgproc bodies (sanity floor ${GLOBAL_FLOOR})"
fi

echo "---"
echo "${ARCH}: ${ok} per-kernel ok, ${notfound} not-found (warn), ${fail} failed, ${GLOBAL_HITS} global ${GLOBAL_LABEL} hits"
exit "$fail"
