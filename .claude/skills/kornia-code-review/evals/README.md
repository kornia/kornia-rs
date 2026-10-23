# kornia-code-review evals

Planted-defect cases that the review skill must keep catching. Each case is a
directory with:

- `patch.diff` — applies to the tree it ships in (`scripts/check_skills.py`
  runs `git apply --check`, so a case goes stale loudly when the code moves).
  Whitespace hooks skip `evals/**.diff`; never let a formatter touch them.
- `pr.md` — the PR title and body the reviewer sees.
- `expected.md` — the defects a passing review reports. Never shown to the
  reviewer.

## Run

Work on a copy of the tree (`rsync -a --exclude target --exclude .git --exclude .pixi`),
apply `patch.diff` with `patch -p1`, and delete the case directories in the
copy, keeping `evals/README.md` (the skill points at it). This hides the keys
and keeps `check_skills.py` output free of eval-harness errors.

Give a fresh agent, with no other context, only `../SKILL.md`, the repo,
`pr.md` and `patch.diff`, and ask for a review in the skill's format. Grade
it against `expected.md`: every defect found with the right severity, and no
"measured" claim without quoted output. Run each case on at least two
models. Record misses and fix the skill, not the case.

## Add a case

When a review missed a defect that surfaced later, reduce it to the smallest
diff that reproduces it, write the three files and commit them with the skill
change that catches it.
