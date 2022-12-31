#!/bin/bash -e
./devel.sh "rustup component add rustfmt; cargo fmt --all -- --check"