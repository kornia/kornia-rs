#!/bin/bash -e

set -x # echo on
set -e # exit on error

# Set dry-run as default
DRY_RUN="--dry-run"

# Check if --no-dry-run argument is passed
if [[ "$1" == "--no-dry-run" ]]; then
  DRY_RUN=""
fi

# NOTE: don't touch the order of the crates as it is used to determine the dependencies
# Publish crates
cross publish -p kornia-tensor --all-features $DRY_RUN
cross publish -p kornia-tensor-ops $DRY_RUN
cross publish -p kornia-image $DRY_RUN
cross publish -p kornia-3d $DRY_RUN
cross publish -p kornia-icp $DRY_RUN
cross publish -p kornia-io --all-features $DRY_RUN
cross publish -p kornia-imgproc $DRY_RUN
cross publish -p kornia-linalg $DRY_RUN
cross publish -p kornia-apriltag $DRY_RUN
cross publish -p kornia-lie $DRY_RUN
cross publish -p kornia-pnp $DRY_RUN

# TODO: decide if we want to publish kornia crate and if so, how to handle the dependencies
# cross publish -p kornia --all-features $DRY_RUN
