#!/usr/bin/env bash
# Regenerate cv2 + kornia contour snapshots for all (fixture, mode, method) combos.
# Output: crates/kornia-imgproc/tests/snapshots/{cv2,kornia}_<fixture>_<mode>_<method>.json
#
# Phase 0 of the bit-exact-cv2-parity work. The snapshots act as:
#  - regression catcher (kornia_*.json — current behaviour, frozen)
#  - parity target  (cv2_*.json — what we're aiming at)
#
# Combos (14 each, 28 total):
#  pic1, pic3, pic4  ×  {external, list}  ×  {simple, none}     = 12
#  pic2              ×  {external}        ×  {simple, none}     = 2
#  pic2 LIST is skipped (5722 holes — output is too large to be useful as a snapshot)
#
# Usage: bash crates/kornia-imgproc/examples/dump_snapshots.sh
set -euo pipefail

SNAP_DIR="crates/kornia-imgproc/tests/snapshots"
DATA_DIR="crates/kornia-imgproc/examples/data"
mkdir -p "$SNAP_DIR"

if [ ! -f "target/release/examples/dump_contours" ]; then
    echo "Building dump_contours..."
    cargo build --release --example dump_contours -p kornia-imgproc
fi

write_kornia() {
    local fixture=$1 mode=$2 method=$3
    local out="$SNAP_DIR/kornia_${fixture}_${mode}_${method}.json"
    ./target/release/examples/dump_contours "$DATA_DIR/${fixture}.png" "$mode" "$method" > "$out"
    printf "  kornia %s %s %s -> %s (%d bytes)\n" \
        "$fixture" "$mode" "$method" "$out" "$(stat -c %s "$out")"
}

write_cv2() {
    local fixture=$1 mode=$2 method=$3
    local out="$SNAP_DIR/cv2_${fixture}_${mode}_${method}.json"
    python3 - "$fixture" "$mode" "$method" "$out" <<'PY'
import json, sys, cv2
fixture, mode_s, method_s, out = sys.argv[1:]
img = cv2.imread(f"crates/kornia-imgproc/examples/data/{fixture}.png", cv2.IMREAD_GRAYSCALE)
_, bw = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
mode = {"external": cv2.RETR_EXTERNAL, "list": cv2.RETR_LIST}[mode_s]
method = {"simple": cv2.CHAIN_APPROX_SIMPLE, "none": cv2.CHAIN_APPROX_NONE}[method_s]
contours, _ = cv2.findContours(bw, mode, method)
data = {"contours": [c.reshape(-1, 2).tolist() for c in contours]}
with open(out, "w") as f:
    json.dump(data, f, separators=(",", ":"))
PY
    printf "  cv2    %s %s %s -> %s (%d bytes)\n" \
        "$fixture" "$mode" "$method" "$out" "$(stat -c %s "$out")"
}

echo "== generating snapshots =="
for fixture in pic1 pic3 pic4; do
    for mode in external list; do
        for method in simple none; do
            write_kornia "$fixture" "$mode" "$method"
            write_cv2    "$fixture" "$mode" "$method"
        done
    done
done

# pic2: EXT only — LIST has 5722 holes, snapshot would be ~MB and useless to diff.
for method in simple none; do
    write_kornia pic2 external "$method"
    write_cv2    pic2 external "$method"
done

echo
echo "== summary =="
ls -la "$SNAP_DIR"/*.json | awk '{s+=$5} END {printf "  %d files, %.1f KB total\n", NR, s/1024}'
