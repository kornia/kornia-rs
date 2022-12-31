#!/bin/bash -e
./devel.sh "rustup component add clippy; cargo clippy -- -D warnings"