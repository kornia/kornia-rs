#!/usr/bin/env bash
# Fetch real-image fixtures from OpenCV's sample data into examples/data/.
#
# These images (pic1-pic6) are part of OpenCV's tutorial set, used for
# bench/correctness checks against cv2.findContours. They live in the
# git history of this branch (412 KB total) but are .gitignored going
# forward — re-fetch them here whenever needed for offline work.
#
# Usage: bash crates/kornia-imgproc/examples/fetch_fixtures.sh
set -e

DIR="$(dirname "$0")/data"
mkdir -p "$DIR"

base="https://github.com/opencv/opencv/raw/4.x/samples/data"
for n in 1 2 3 4 5 6; do
    f="$DIR/pic${n}.png"
    if [ -f "$f" ]; then
        echo "  pic${n}.png already present"
    else
        echo "  fetching pic${n}.png..."
        curl -sLo "$f" "$base/pic${n}.png" || {
            echo "    (skipping — not available)"
            rm -f "$f"
        }
    fi
done

echo
echo "fixtures available:"
ls -la "$DIR"/*.png 2>/dev/null || echo "  (none)"
