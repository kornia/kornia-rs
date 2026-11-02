#!/usr/bin/env bash
# Shared scaffold for `claude plugin eval` (see README.md). Called by each
# case's scaffold.sh with the case directory; runs in the empty workspace.
# Builds the reviewed tree: repo copy + the case's patch, all eval files removed.
set -euo pipefail
case_dir="$1"
repo="$case_dir"
until [[ -f "$repo/Cargo.toml" && -d "$repo/.claude/skills" ]]; do
  [[ "$repo" == / ]] && { echo "no repo root above $case_dir" >&2; exit 1; }
  repo="$(dirname "$repo")"
done
rsync -a --exclude target --exclude .git --exclude .pixi --exclude evals/results "$repo/" ./
cp "$case_dir/pr.md" PR.md
# A control case's change is already merged (merged.diff); the reviewer sees
# it plus patch.diff as one PR.
if [[ -f "$case_dir/merged.diff" ]]; then
  cat "$case_dir/merged.diff" "$case_dir/patch.diff" > pr.diff
else
  cp "$case_dir/patch.diff" pr.diff
fi
if [[ -f "$case_dir/issue.md" ]]; then cp "$case_dir/issue.md" ISSUE.md; fi
patch -p1 -s < "$case_dir/patch.diff"
# The README names the cases and what they test, so the reviewer gets a stub.
rm -r .claude/skills/kornia-code-review/evals
mkdir .claude/skills/kornia-code-review/evals
echo "Eval cases are hidden during an eval run." > .claude/skills/kornia-code-review/evals/README.md
