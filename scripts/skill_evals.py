#!/usr/bin/env python3
"""Tools around the kornia-code-review eval cases (`pixi run skills-eval`).

  sync       Regenerate each case's graders/precision.md from its key, PR,
             issue and diff. check_skills.py fails while one is stale.
  report     RESULTS.json... Per case, arm and grader: passes over runs,
             pass^k (caught on every one of k runs), Wilson 95% interval of
             the case pass rate, WITH-minus-W/OUT delta. --dump DIR also
             writes every review and a labels file for `calibrate`.
  calibrate  LABELS.json  Agreement and Cohen's kappa between the LLM judge
             and reference labels ("human": a person, or a second grader
             blind to the judge; "labeller" says which). --rejudge MODEL
             re-asks the judge with the current graders instead of using
             the stored verdicts.
  mine       SZZ: fix commits on a ref, the commits that introduced the lines
             they changed, and whether reverting the fix still applies.
             --make SHA NAME writes an unvetted case to evals/heldout/NAME.

See .claude/skills/kornia-code-review/evals/README.md.
"""
import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVALS = ROOT / ".claude" / "skills" / "kornia-code-review" / "evals"
HELDOUT = EVALS / "heldout"
SYNC_MARK = "<!-- skill_evals.py sync: {} -->"
TEST_PATH = re.compile(r"(^|/)(tests?|benches)/|_test\.rs$|/tests\.rs$")


def cases() -> list:
    return sorted(p for p in EVALS.iterdir() if p.is_dir() and p.name not in ("results", "heldout", "calibration"))


def git(*args, check=True) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=check, capture_output=True, text=True).stdout


# --- sync -------------------------------------------------------------------

# An indicator (not scored) until a judge is calibrated on it: on 2026-10-24 a
# Sonnet judge gave 0/3 to 3/3 on the same review and criterion.
PRECISION = """---
type: llm
arm: with-only
---
Judge only the review's blocking and major findings, against the PR below.
FAIL if any of them is false: its conclusion is wrong because it says the
diff does something it does not, cites a function or line in the diff that
does not exist, calls correct code wrong, or states the opposite of a fact
in the key. Severity is never a contradiction: rating a key item higher or
lower than the key is graded elsewhere. A wrong
detail in a finding whose conclusion still holds is not false. Line numbers
refer to the whole file, which the diff shows only in part: never fail a finding
for its line number. A claim about code
outside the diff is taken as true unless the key contradicts it. Findings that
are not in the key are fine when true; missing findings, severity choices and
style are graded elsewhere. PASS when no blocking or major finding is false,
including when the review has none.
When failing, quote the false finding.

## Key (expected.md)

{key}

## PR description

{pr}

## Linked issue

{issue}

## Diff

```diff
{diff}```

{mark}
"""


def precision_text(case: Path) -> str:
    parts = [case / n for n in ("expected.md", "pr.md", "issue.md", "merged.diff", "patch.diff")]
    digest = hashlib.sha256(b"".join(p.read_bytes() for p in parts if p.is_file())).hexdigest()[:16]
    read = lambda n: (case / n).read_text(encoding="utf-8").strip() if (case / n).is_file() else "(none: not readable in this review)"
    # The key's items, without the dated run notes below them.
    key = "\n".join(l for l in read("expected.md").splitlines() if re.match(r"\d+\. |A passing review|Items .* pass bar", l))
    return PRECISION.format(key=key, pr=read("pr.md"), issue=read("issue.md"),
                            diff="".join((case / n).read_text(encoding="utf-8") for n in ("merged.diff", "patch.diff")
                                         if (case / n).is_file()), mark=SYNC_MARK.format(digest))


def stale_precision() -> list:
    return [c.name for c in cases() if (c / "patch.diff").is_file()
            and ((c / "graders/precision.md").read_text(encoding="utf-8") if (c / "graders/precision.md").is_file() else "")
            != precision_text(c)]


def sync(_args) -> int:
    for c in cases():
        (c / "graders").mkdir(exist_ok=True)
        (c / "graders/precision.md").write_text(precision_text(c), encoding="utf-8")
    print(f"synced graders/precision.md for {len(cases())} cases")
    return 0


# --- report -----------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def pass_hat_k(c: int, n: int, k: int) -> float:
    return math.comb(c, k) / math.comb(n, k) if 0 < k <= n else float("nan")


def review_of(run: dict) -> str:
    for g in run.get("graders", []):
        if g.get("evidence") and "judge votes" in str(g.get("explanation", "")):
            return g["evidence"]
    return ""


def indicators(case: str) -> set:
    """Graders that are reported but not scored (`arm: with-only`), per the current files."""
    d = EVALS / case if (EVALS / case).is_dir() else HELDOUT / case
    return {g.stem for g in (d / "graders").glob("*.md") if "\narm: with-only\n" in g.read_text(encoding="utf-8")}


def report(args) -> int:
    rows, labels = [], []
    for path in args.results:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        model = doc.get("suite", {}).get("modelOverride") or "default"
        judge = doc.get("suite", {}).get("judgeModel") or "default"
        print(f"\n## {path}: reviewer {model}, judge {judge}\n")
        print("| case | arm | runs | case passes | 95% CI | pass^k | failing graders (passes/runs) | run errors |")
        print("|---|---|---|---|---|---|---|---|")
        for case in doc.get("cases", []):
            scores = {}
            for arm, all_runs in case.get("arms", {}).items():
                # A run that errored (sandbox, timeout, API) was never reviewed: report it, don't score it.
                runs = [r for r in all_runs if not r.get("error")]
                errors = len(all_runs) - len(runs)
                n = len(runs)
                ind = indicators(case["name"])
                # Re-score with the current grader files, so older results stay comparable.
                ok = sum(1 for r in runs if all(g.get("passed") for g in r.get("graders", []) if g["name"] not in ind))
                k = min(args.k or n, n)
                per = Counter()
                for i, r in enumerate(runs):
                    for g in r.get("graders", []):
                        per[g["name"]] += bool(g.get("passed"))
                    if args.dump and review_of(r):
                        name = f"{case['name']}-{arm}-{model}-{i}.md"
                        out = Path(args.dump)
                        out.mkdir(parents=True, exist_ok=True)
                        (out / name).write_text(review_of(r), encoding="utf-8")
                        for g in r.get("graders", []):
                            if g.get("judgeVotes"):
                                labels.append({"case": case["name"], "grader": g["name"], "review": name,
                                               "judge": bool(g.get("passed")), "human": None, "labeller": None})
                lo, hi = wilson(ok, n)
                weak = ", ".join(f"{g}{' (indicator)' if g in ind else ''} {c}/{n}"
                                 for g, c in sorted(per.items()) if c < n) or "-"
                print(f"| {case['name']} | {arm} | {n} | {ok}/{n} | {lo:.2f}-{hi:.2f} | {pass_hat_k(ok, n, k):.2f} (k={k}) | {weak} | {errors} |")
                if n:
                    scored = [[bool(g.get("passed")) for g in r.get("graders", []) if g["name"] not in ind] for r in runs]
                    scores[arm] = sum(sum(s) / len(s) for s in scored if s) / n
            if "with" in scores and "without" in scores:
                rows.append((case["name"], model, scores["with"] - scores["without"]))
    if rows:
        print("\n| case | reviewer | score delta WITH - W/OUT |\n|---|---|---|")
        for name, model, d in rows:
            print(f"| {name} | {model} | {d:+.2f} |")
    if args.dump and labels:
        (Path(args.dump) / "labels.json").write_text(json.dumps(labels, indent=1) + "\n", encoding="utf-8")
        print(f"\nwrote {len(labels)} labels to {args.dump}/labels.json; fill in each \"human\" and \"labeller\"")
    return 0


# --- calibrate --------------------------------------------------------------

def grader_body(case: str, grader: str) -> str:
    text = (EVALS / case / "graders" / f"{grader}.md").read_text(encoding="utf-8")
    return text.split("---\n", 2)[2] if text.startswith("---\n") else text


def rejudge(model: str, criteria: str, review: str) -> bool:
    prompt = ("You grade a code review against one criterion. Reply with PASS or FAIL on the first line, "
              f"then one sentence of reason.\n\n## Criterion\n\n{criteria}\n\n## Review\n\n{review}\n")
    # Keep personal settings, hooks and the repo's CLAUDE.md out of the judge: no
    # setting sources, a neutral cwd, and --bare when an API key allows it.
    bare = ["--bare"] if os.environ.get("ANTHROPIC_API_KEY") else []
    out = subprocess.run(["claude", "-p", *bare, "--model", model, "--tools", "", "--setting-sources", "",
                          "--no-session-persistence"], input=prompt, capture_output=True, text=True, check=True,
                         cwd=tempfile.gettempdir()).stdout
    return out.strip().upper().startswith("PASS")


def kappa(pairs: list) -> float:
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    pa, pb = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def calibrate(args) -> int:
    path = Path(args.labels)
    labels = [x for x in json.loads(path.read_text(encoding="utf-8")) if x.get("human") is not None]
    if not labels:
        print("no human labels yet", file=sys.stderr)
        return 1
    pairs, by_grader, misses = [], {}, []
    for x in labels:
        judge = x["judge"]
        if args.rejudge:
            review = (path.parent / x["review"]).read_text(encoding="utf-8")
            judge = rejudge(args.rejudge, grader_body(x["case"], x["grader"]), review)
        pairs.append((judge, x["human"]))
        by_grader.setdefault(x["grader"], []).append((judge, x["human"]))
        if judge != x["human"]:
            misses.append(f"{x['review']} {x['grader']}: judge {judge}, human {x['human']}")
    agree = sum(a == b for a, b in pairs) / len(pairs)
    who = Counter(x.get("labeller") or "unknown" for x in labels)
    print(f"{len(pairs)} labels ({', '.join(f'{k}: {v}' for k, v in who.items())}): "
          f"agreement {agree:.0%}, Cohen's kappa {kappa(pairs):.2f}")
    for g, ps in sorted(by_grader.items()):
        print(f"  {g}: {sum(a == b for a, b in ps)}/{len(ps)}")
    for m in misses:
        print(f"  disagree: {m}")
    return 0 if agree >= args.min_agreement else 1


# --- mine -------------------------------------------------------------------

def changed_code(sha: str) -> list:
    files = git("diff", "--name-only", f"{sha}^", sha).split()
    return [f for f in files if f.endswith(".rs") and f.startswith("crates/") and not TEST_PATH.search(f)]


def introducers(sha: str, files: list) -> Counter:
    found = Counter()
    for f in files:
        diff = git("diff", "-U0", f"{sha}^", sha, "--", f)
        for m in re.finditer(r"^@@ -(\d+)(?:,(\d+))? ", diff, re.M):
            start, count = int(m.group(1)), int(m.group(2) or 1)
            if count == 0:
                continue
            blame = git("blame", "--porcelain", "-L", f"{start},{start + count - 1}", f"{sha}^", "--", f, check=False)
            found.update(line.split()[0] for line in blame.splitlines() if re.match(r"^[0-9a-f]{40} ", line))
    return found


def revert_patch(sha: str, files: list) -> str:
    return git("diff", sha, f"{sha}^", "--", *files)


def applies(patch: str) -> bool:
    return subprocess.run(["git", "apply", "--check", "-"], cwd=ROOT, input=patch, capture_output=True, text=True).returncode == 0


def mine(args) -> int:
    if args.make:
        return make_case(*args.make)
    log = git("log", args.ref, f"--since={args.since}", "-i", "-E", "--grep=^(fix|revert)|\\bbug\\b|regression",
              "--format=%H %s")
    print("| fix | subject | code lines | introduced by | revert applies |\n|---|---|---|---|---|")
    shown = 0
    for line in log.splitlines():
        sha, subject = line.split(" ", 1)
        files = changed_code(sha)
        if not files:
            continue
        size = sum(int(a) + int(b) for a, b, *_ in (l.split("\t") for l in git("diff", "--numstat", f"{sha}^", sha, "--", *files).splitlines()) if a != "-")
        if size > args.max_lines:
            continue
        intro = introducers(sha, files).most_common(1)
        who = git("log", "-1", "--format=%h %s", intro[0][0]).strip() if intro else "-"
        print(f"| {sha[:9]} | {subject[:60]} | {size} | {who[:70]} | {'yes' if applies(revert_patch(sha, files)) else 'no'} |")
        shown += 1
        if shown >= args.limit:
            break
    return 0


def make_case(sha: str, name: str) -> int:
    files = changed_code(sha)
    patch = revert_patch(sha, files)
    if not files or not applies(patch):
        print(f"{sha}: reverting its code changes no longer applies", file=sys.stderr)
        return 1
    intro = introducers(sha, files).most_common(1)
    case = HELDOUT / name
    (case / "graders").mkdir(parents=True, exist_ok=False)
    src = EVALS / "gray-copy-scope"
    (case / "patch.diff").write_text(patch, encoding="utf-8")
    pr = git("log", "-1", "--format=Title: %s%n%n%b", intro[0][0]) if intro else "Title: refactor\n"
    (case / "pr.md").write_text(pr, encoding="utf-8")
    fix = git("log", "-1", "--format=%s%n%n%b", sha)
    (case / "expected.md").write_text(
        f"UNVETTED (mined from fix {sha[:9]}): a passing review reports the defect this later fix corrected.\n\n"
        f"{fix}\n\nVet it before trusting a score: the defect must be visible from the diff alone.\n", encoding="utf-8")
    # No issue.md yet: the reviewer is told nothing about ISSUE.md. Copy another
    # case's prompt.md when you add one while vetting.
    prompt = (src / "prompt.md").read_text(encoding="utf-8")
    prompt = prompt.replace(prompt.split("\n")[1], f'description: "Mined (SZZ) from fix {sha[:9]}; unvetted."')
    (case / "prompt.md").write_text(prompt
                                    .replace("There is no network and no `gh`; `ISSUE.md` is what\n`gh issue view` prints for the linked issue.",
                                             "There is no network and no `gh`."), encoding="utf-8")
    (case / "case.yaml").write_text(f'schema_version: "1.1"\nname: {name}\ncontext:\n  scaffold_script: scaffold.sh\n', encoding="utf-8")
    wrap = case / "scaffold.sh"
    wrap.write_text('#!/usr/bin/env bash\nexec bash "$(dirname "$(realpath "$0")")/../../scaffold.sh" "$(dirname "$(realpath "$0")")"\n', encoding="utf-8")
    wrap.chmod(0o755)
    (case / "graders/defect.md").write_text(
        "---\ntype: llm\n---\nPASS if the review reports, as major or blocking, the defect that this later fix "
        f"corrected (same code, same problem):\n\n{fix}\n```diff\n{git('diff', f'{sha}^', sha, '--', *files)}```\n"
        "FAIL if the review does not identify that defect.\n", encoding="utf-8")
    print(f"wrote {case.relative_to(ROOT)}; vet expected.md, then run it with --case {name}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sync")
    r = sub.add_parser("report")
    r.add_argument("results", nargs="+")
    r.add_argument("--k", type=int, help="k for pass^k (default: all runs)")
    r.add_argument("--dump", help="write reviews and a labels file here")
    c = sub.add_parser("calibrate")
    c.add_argument("labels")
    c.add_argument("--rejudge", metavar="MODEL")
    c.add_argument("--min-agreement", type=float, default=0.9)
    m = sub.add_parser("mine")
    m.add_argument("--ref", default="origin/main")
    m.add_argument("--since", default="12 months ago")
    m.add_argument("--max-lines", type=int, default=60, help="max changed non-test lines in the fix")
    m.add_argument("--limit", type=int, default=20)
    m.add_argument("--make", nargs=2, metavar=("SHA", "NAME"))
    args = ap.parse_args()
    return {"sync": sync, "report": report, "calibrate": calibrate, "mine": mine}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
