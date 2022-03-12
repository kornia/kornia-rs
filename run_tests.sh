#!/bin/bash -e
./devel.sh "python3 -m venv .venv; source .venv/bin/activate; maturin develop --extras dev; pytest test/"