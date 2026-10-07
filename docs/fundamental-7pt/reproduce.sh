#!/usr/bin/env bash
# Build only the implementation under test and run the focused F7/F8 evaluation.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
data_root="${RANSAC_DATA_ROOT:-/Users/oldufo/dev/pydegensac/benchmarks/data}"
task_dir="$(mktemp -d /tmp/fundamental-solvers.XXXXXX)"
output_dir="${RANSAC_OUTPUT_DIR:-$task_dir/results}"
uv venv --python 3.11 "$task_dir/venv"
python_bin="$task_dir/venv/bin/python"
uv pip install --python "$python_bin" numpy==2.4.6 h5py==3.16.0 \
    opencv-python==4.13.0.92 maturin==1.15.0 matplotlib==3.11.2 pytest==9.1.1
export CARGO_TARGET_DIR="$task_dir/target"
"$task_dir/venv/bin/maturin" build --release --no-default-features \
    -m "$repo_root/kornia-py/Cargo.toml" -i "$python_bin" --out "$task_dir/wheels"
uv pip install --python "$python_bin" "$task_dir"/wheels/*.whl
cd "$repo_root"
cargo test -p kornia-3d --lib fundamental
cargo test -p kornia-3d --lib ransac::driver::tests
cargo test -p kornia-3d --doc fundamental
"$python_bin" -m pytest -q kornia-py/tests/test_fundamental_solvers.py
cargo bench -p kornia-3d --bench bench_fundamental -- --quick
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=4
"$python_bin" kornia-py/benchmarks/bench_fundamental_solvers.py \
    --data-root "$data_root" --json "$output_dir/results.json"
"$python_bin" kornia-py/benchmarks/plot_fundamental_solvers.py \
    --json "$output_dir/results.json" --out-dir "$output_dir"
printf 'Results saved in %s\n' "$output_dir"
