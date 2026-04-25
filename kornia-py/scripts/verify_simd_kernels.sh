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
set -euo pipefail

WHEEL="${1:?usage: $0 <wheel>}"
case "$WHEEL" in
  *manylinux*x86_64*|*linux_x86_64*) ARCH=x86_64 ;;
  *manylinux*aarch64*|*linux_aarch64*) ARCH=aarch64 ;;
  *) echo "::error::cannot infer arch from $WHEEL" >&2; exit 1 ;;
esac

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
unzip -q -o "$WHEEL" -d "$WORK"
SO=$(find "$WORK" -name 'kornia_rs*.so' | head -n1)
[ -n "$SO" ] || { echo "::error::no kornia_rs*.so in $WHEEL" >&2; exit 1; }

if [ "$ARCH" = aarch64 ]; then
  OBJDUMP=aarch64-linux-gnu-objdump
  KERNELS=(
    'flip::hflip_rgb_u8~[[:space:]](ld3|rev64)[[:space:]]'
    'color::gray::rgb_to_gray_u8~[[:space:]](ld3|umlal)[[:space:]]'
    'normalize::normalize_rgb_u8~[[:space:]](fmla|ucvtf)[[:space:]]'
    'resize::pyramid::pyrup_~[[:space:]](ld3|urhadd)[[:space:]]'
    'resize::kernels::horizontal_row_c~[[:space:]](smlal|umlal|ld3)[[:space:]]'
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

# Per-kernel check. `objdump -dC` opens each function with a header line
# of the form `00000000000abcde <demangled::path::name>:` followed by
# `addr: <mnemonic> <ops>` lines and a blank-line terminator.
fail=0
ok=0
notfound=0
for entry in "${KERNELS[@]}"; do
  path="${entry%%~*}"
  patt="${entry#*~}"
  status=$(awk -v path="$path" -v p="$patt" '
    BEGIN { state="seek"; matched=0; sawpath=0 }
    state=="seek" && /^[0-9a-f]+ </ && index($0, path) > 0 {
      state="in"; sawpath=1; next
    }
    state=="seek" { next }
    state=="in" && /^$/ { state="seek"; next }
    state=="in" && match($0, p) { matched=1 }
    END {
      if (!sawpath) { print "notfound"; exit }
      print (matched ? "ok" : "noasm")
    }
  ' "$DUMP")
  case "$status" in
    ok)       echo "ok   ${ARCH}: $path"; ok=$((ok+1)) ;;
    notfound) echo "::warning::${ARCH}: no function path '$path' present in wheel (renamed? heavily inlined?)"; notfound=$((notfound+1)) ;;
    noasm)    echo "::error::${ARCH}: '$path' present but no body matches SIMD pattern '$patt'"; fail=$((fail+1)) ;;
  esac
done

# Global sanity check: regardless of inlining, *some* kornia_imgproc-rooted
# function body should contain SIMD asm. Catches the worst-case regression
# where every NEON/AVX2 path got stripped (e.g. RUSTFLAGS accidentally
# disabled the target feature, or a build-system change dropped the
# `cpu_features()` runtime gate).
GLOBAL_HITS=$(awk -v p="$GLOBAL_PATTERNS" '
  /^[0-9a-f]+ </ { in_kornia = (index($0, "kornia_imgproc") > 0); next }
  in_kornia && match($0, p) { n++; }
  END { print n+0 }
' "$DUMP")
if [ "$GLOBAL_HITS" -lt 100 ]; then
  echo "::error::${ARCH}: only $GLOBAL_HITS ${GLOBAL_LABEL} instructions in kornia_imgproc bodies (expected >100). SIMD pipeline likely broken globally."
  fail=$((fail+1))
else
  echo "ok   ${ARCH}: ${GLOBAL_HITS} ${GLOBAL_LABEL} instructions inside kornia_imgproc bodies (sanity floor 100)"
fi

echo "---"
echo "${ARCH}: ${ok} per-kernel ok, ${notfound} not-found (warn), ${fail} failed, ${GLOBAL_HITS} global ${GLOBAL_LABEL} hits"
exit "$fail"
