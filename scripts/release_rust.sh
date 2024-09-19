#!/bin/bash -e

# Set dry-run as default
DRY_RUN="--dry-run"

# Check if --no-dry-run argument is passed
if [[ "$1" == "--no-dry-run" ]]; then
  DRY_RUN=""
fi

# Publish crates
cross publish -p kornia-core $DRY_RUN
cross publish -p kornia-image $DRY_RUN
cross publish -p kornia-io --all-features $DRY_RUN
cross publish -p kornia-imgproc $DRY_RUN
cross publish -p kornia --all-features $DRY_RUN
