#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
task_dir="$(mktemp -d /tmp/fundamental-confidence.XXXXXX)"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$task_dir/target}"
mkdir -p "$task_dir/src"
cp "$repo_root/docs/fundamental-7pt/optimization/confidence_probe.rs" "$task_dir/src/main.rs"
cp "$repo_root/Cargo.lock" "$task_dir/Cargo.lock"
python3 - "$repo_root" "$task_dir" <<'PY'
import json, sys
from pathlib import Path
repo, probe = map(Path, sys.argv[1:])
manifest = '''[package]
name = "fundamental-confidence-probe"
version = "0.1.0"
edition = "2021"
[dependencies]
'''
for crate in ("kornia-3d", "kornia-algebra"):
    manifest += crate + ' = { path = ' + json.dumps(str(repo / "crates" / crate)) + ' }\n'
manifest += 'serde_json = "1"\nrand = "0.10"\n[profile.release]\nlto = "thin"\ncodegen-units = 1\n'
(probe / "Cargo.toml").write_text(manifest)
PY
cargo build --release --manifest-path "$task_dir/Cargo.toml"
"$CARGO_TARGET_DIR/release/fundamental-confidence-probe" > "$task_dir/results.json"
printf 'Results: %s\n' "$task_dir/results.json"
