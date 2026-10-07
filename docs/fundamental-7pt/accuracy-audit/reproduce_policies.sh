#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
task_dir="$(mktemp -d /tmp/fundamental-policies.XXXXXX)"
python_exec="${PYTHON_EXEC:-python3}"
data_root="${DATA_ROOT:-$HOME/dev/pydegensac/benchmarks/data}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$task_dir/target}"
mkdir -p "$task_dir/src"
cp "$repo_root/docs/fundamental-7pt/accuracy-audit/quality_policies.rs" "$task_dir/src/main.rs"
cp "$repo_root/Cargo.lock" "$task_dir/Cargo.lock"
"$python_exec" - "$repo_root" "$task_dir" <<'PY'
import json, sys
from pathlib import Path
repo, probe = map(Path, sys.argv[1:])
manifest = '[package]\nname="fundamental-quality-policies"\nversion="0.1.0"\nedition="2021"\n[dependencies]\n'
for crate in ('kornia-3d', 'kornia-algebra'):
    manifest += crate + '={path=' + json.dumps(str(repo / 'crates' / crate)) + '}\n'
manifest += 'serde_json="1"\nrand="0.10"\n[profile.release]\nlto="thin"\ncodegen-units=1\n'
(probe / 'Cargo.toml').write_text(manifest)
PY
cargo build --release --manifest-path "$task_dir/Cargo.toml"
for experiment in lo controls; do
    options=(--modes count_lo --budgets 64,256,1000,4096,16384 --rounds 3)
    if [[ "$experiment" == controls ]]; then
        options=(--modes count_adaptive,count_fixed,msac_adaptive,msac_fixed,count_lo,msac_lo --budgets 4096 --rounds 1)
    fi
    "$python_exec" "$repo_root/kornia-py/benchmarks/prepare_fundamental_policy_inputs.py" \
        --data-root "$data_root" --output "$task_dir/$experiment-input.jsonl" \
        --confidence 0.999999 "${options[@]}"
    "$CARGO_TARGET_DIR/release/fundamental-quality-policies" \
        < "$task_dir/$experiment-input.jsonl" > "$task_dir/$experiment-raw.jsonl"
    "$python_exec" "$repo_root/kornia-py/benchmarks/evaluate_fundamental_policies.py" \
        --raw "$task_dir/$experiment-raw.jsonl" --data-root "$data_root" \
        --out-dir "$task_dir/$experiment" --probe "$CARGO_TARGET_DIR/release/fundamental-quality-policies" \
        --confidence 0.999999
done
"$python_exec" "$repo_root/kornia-py/benchmarks/plot_fundamental_pareto.py" \
    --json "$task_dir/lo/count_lo.json" --out-dir "$task_dir/lo"
printf 'Results: %s\n' "$task_dir"
