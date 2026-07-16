#!/usr/bin/env bash
#
# Publish the workspace's Rust crates to crates.io in topological order.
#
# Default: plan mode — prints what would be published and runs `cargo
# publish --dry-run` against the leaf crates only (tier 0, no kornia-* deps).
# `cargo publish --dry-run` for the dependent crates always fails for
# workspaces because the not-yet-published deps aren't resolvable from the
# crates.io index — that's a Cargo limitation, not a script bug.
#
# Pass --execute to actually publish. Requires CARGO_REGISTRY_TOKEN env var.
#
# Crates are published deps-first. crates.io's index has a few-seconds
# propagation lag after each publish, so we sleep between dependent steps.
#
# Idempotent: each step queries crates.io for the current published version
# and skips if it already matches the local workspace version. Lets us resume
# from where a previous run failed without re-publishing.
#
# Crates intentionally not published from this script:
#   - kornia            (umbrella — re-enable once we decide how to gate
#                        optional/feature-heavy deps)
#   - kornia-vlm        (pulls in candle, tokio, hf-hub — heavy)
#   - kornia-apriltag   (large git submodule of test images)
#   - kornia-calib      (depends on kornia-apriltag, so excluded transitively)
# Add them back to PUBLISH_ORDER below once those constraints are sorted.

set -euo pipefail

MODE="plan"        # plan | execute
SLEEP_SECS=20

for arg in "$@"; do
  case "$arg" in
    --execute|--no-dry-run)
      MODE="execute"
      ;;
    --help|-h)
      sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "Unknown arg: $arg" >&2
      exit 1
      ;;
  esac
done

# Topological publish order. Every crate's kornia-* dependencies must
# already be on crates.io (or appear earlier in this list).
#
# Tier 0 (no kornia-* deps):       kornia-algebra, kornia-bow, kornia-tensor
# Tier 1 (-> tensor):              kornia-tensor-ops, kornia-image
# Tier 2 (-> image, tensor):       kornia-io
# Tier 3 (-> algebra/image/tensor/io): kornia-imgproc
# Tier 4 (-> everything above):    kornia-3d
PUBLISH_ORDER=(
  kornia-algebra
  kornia-bow
  kornia-tensor
  kornia-tensor-ops
  kornia-image
  kornia-io
  kornia-imgproc
  kornia-3d
)

# Tier-0 leaves (no kornia-* deps) — can be `cargo publish --dry-run`'d
# locally for verification because there's nothing to resolve from crates.io.
TIER_0=(kornia-algebra kornia-bow kornia-tensor)

# Per-crate extra flags. We deliberately do NOT pass --all-features:
# kornia-io's gstreamer feature requires glib/gobject dev headers, and
# the published Cargo.toml already declares features so users can opt in.
declare -A CRATE_FEATURES=(
  # Empty by design — see comment above.
  # Add per-crate flags here if a crate needs them.
  [_placeholder]=""
)

local_version() {
  cargo metadata --format-version 1 --no-deps 2>/dev/null \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(next(p['version'] for p in d['packages'] if p['name'] == '$1'))"
}

remote_version() {
  # crates.io requires a User-Agent header (returns 403 without one).
  # Retry up to 3 times to ride out transient rate-limit / network blips —
  # mis-reading "NOT-PUBLISHED" for an already-published crate causes the
  # next cargo publish to error with "crate already exists".
  local crate="$1"
  local attempt
  for attempt in 1 2 3; do
    local response
    response=$(curl -fsS \
      -A "kornia-rs release script (https://github.com/kornia/kornia-rs)" \
      "https://crates.io/api/v1/crates/$crate" 2>/dev/null) || {
      sleep 2
      continue
    }
    local v
    v=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin)['crate']['max_version'])" 2>/dev/null) || {
      sleep 2
      continue
    }
    if [[ -n "$v" ]]; then
      echo "$v"
      return 0
    fi
  done
  # After 3 retries, report 404 vs network-failure honestly.
  local http_code
  http_code=$(curl -s -o /dev/null -w "%{http_code}" \
    -A "kornia-rs release script (https://github.com/kornia/kornia-rs)" \
    "https://crates.io/api/v1/crates/$crate" 2>/dev/null || echo "000")
  if [[ "$http_code" == "404" ]]; then
    echo "NOT-PUBLISHED"
  else
    echo "ERROR-HTTP-$http_code"  # forces an obvious failure, won't false-match local version
  fi
}

if [[ "$MODE" == "plan" ]]; then
  echo "==> PLAN — would publish ${#PUBLISH_ORDER[@]} crates in this order:"
  for crate in "${PUBLISH_ORDER[@]}"; do
    features="${CRATE_FEATURES[$crate]:-}"
    local_v=$(local_version "$crate")
    remote_v=$(remote_version "$crate")
    if [[ "$local_v" == "$remote_v" ]]; then
      echo "      $crate: $remote_v already published, would SKIP"
    else
      echo "      $crate: $remote_v -> $local_v   cargo publish -p $crate $features"
    fi
  done
  echo
  echo "==> Dry-running tier-0 leaves (no kornia-* deps) to validate packaging..."
  for crate in "${TIER_0[@]}"; do
    local_v=$(local_version "$crate")
    remote_v=$(remote_version "$crate")
    if [[ "$local_v" == "$remote_v" ]]; then
      echo "==> $crate: $remote_v already published, skipping dry-run"
      continue
    fi
    features="${CRATE_FEATURES[$crate]:-}"
    echo "==> cargo publish -p $crate $features --dry-run"
    cargo publish -p "$crate" $features --dry-run
  done
  echo
  echo "==> Plan validation complete. Re-run with --execute to publish for real."
  echo "    (Dependent crates can only be 'dry-run' after their deps are on"
  echo "     crates.io — that's a Cargo workspace publish limitation.)"
  exit 0
fi

# --- execute mode ---

if [[ -z "${CARGO_REGISTRY_TOKEN:-}" ]]; then
  echo "ERROR: CARGO_REGISTRY_TOKEN is not set." >&2
  echo "       Either export it, or run 'cargo login' first." >&2
  exit 1
fi

echo "==> EXECUTE — publishing up to ${#PUBLISH_ORDER[@]} crates"
echo

published_count=0
skipped_count=0

for crate in "${PUBLISH_ORDER[@]}"; do
  local_v=$(local_version "$crate")
  remote_v=$(remote_version "$crate")

  if [[ "$local_v" == "$remote_v" ]]; then
    echo "==> $crate: $remote_v already on crates.io, skipping"
    skipped_count=$((skipped_count + 1))
    continue
  fi

  features="${CRATE_FEATURES[$crate]:-}"
  echo "==> $crate: $remote_v -> $local_v"
  echo "    cargo publish -p $crate $features"
  cargo publish -p "$crate" $features
  published_count=$((published_count + 1))

  # Only sleep if there's more to publish.
  if [[ "$crate" != "${PUBLISH_ORDER[-1]}" ]]; then
    echo "    sleeping ${SLEEP_SECS}s for crates.io index propagation..."
    sleep "$SLEEP_SECS"
  fi
done

echo
echo "==> Done — published $published_count crate(s), skipped $skipped_count (already on crates.io)"
