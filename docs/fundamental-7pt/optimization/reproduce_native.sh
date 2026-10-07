#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
task_dir="$(mktemp -d /tmp/fundamental-native.XXXXXX)"
kornia_root="${KORNIA_ROOT:-$HOME/dev/kornia}"
pydegensac_root="${PYDEGENSAC_ROOT:-$HOME/dev/pydegensac}"
torch_python="${TORCH_PYTHON:-$kornia_root/.venv/bin/python}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$task_dir/target}"
mkdir -p "$task_dir/probe/src"
cp "$repo_root/docs/fundamental-7pt/optimization/solver_probe.rs" "$task_dir/probe/src/main.rs"
cp "$repo_root/Cargo.lock" "$task_dir/probe/Cargo.lock"
python3 - "$repo_root" "$task_dir/probe" <<'PY'
import json, sys
from pathlib import Path
repo, probe = map(Path, sys.argv[1:])
manifest = '''[package]
name = "fundamental-solver-probe"
version = "0.1.0"
edition = "2021"
[dependencies]
'''
for crate in ("kornia-3d", "kornia-algebra"):
    manifest += crate + ' = { path = ' + json.dumps(str(repo / "crates" / crate)) + ' }\n'
manifest += 'serde_json = "1"\n[profile.release]\nlto = "thin"\ncodegen-units = 1\n'
(probe / "Cargo.toml").write_text(manifest)
PY
cargo build --release --manifest-path "$task_dir/probe/Cargo.toml"
python3 "$repo_root/kornia-py/benchmarks/build_pydegensac_minimal.py" \
  --pydegensac-root "$pydegensac_root" --out "$task_dir/c-reference"
clang -O3 -fPIC -shared "$task_dir/c-reference/pydegensac_minimal.c" \
  -o "$task_dir/c-reference/libpydegensac_minimal.so"
probe="$CARGO_TARGET_DIR/release/fundamental-solver-probe"
samples="$repo_root/docs/fundamental-7pt/optimization/real-minimal-samples.npy"
"$torch_python" "$repo_root/kornia-py/benchmarks/check_fundamental_references.py" \
  --rust-probe "$probe" --pydegensac-library "$task_dir/c-reference/libpydegensac_minimal.so" \
  --kornia-root "$kornia_root" --samples-npy "$samples" --json "$task_dir/references.json"
"$torch_python" "$repo_root/kornia-py/benchmarks/bench_fundamental_minimal_references.py" \
  --rust-probe "$probe" --pydegensac-library "$task_dir/c-reference/libpydegensac_minimal.so" \
  --kornia-root "$kornia_root" --samples-npy "$samples" --json "$task_dir/minimal-times.json"
printf 'Results: %s\n' "$task_dir"
