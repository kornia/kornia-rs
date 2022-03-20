#!/bin/bash -e
./devel.sh "python3 -m venv .venv; source .venv/bin/activate; maturin develop --extras dev --cargo-extra-args="--all-features"; pytest test/"