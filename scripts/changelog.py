#!/usr/bin/env python3
"""Changelog fragments: one file per PR in changelog.d/, assembled at release.

  check   --base-ref origin/main --pr-number N [--skip-presence]
          CI: fragment names/content are valid; the PR adds changelog.d/N.<type>.md
          and does not edit CHANGELOG.md. --skip-presence (the `no-changelog`
          label) waives those two rules, never the validity checks.
  preview Print the pending notes; changes nothing.
  build   --version X.Y.Z --date YYYY-MM-DD
          Move [Unreleased] (legacy prose + rendered fragments) into a dated
          section, reset [Unreleased], delete the consumed fragments.

See changelog.d/README.md.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRAGMENTS = ROOT / "changelog.d"
CHANGELOG = ROOT / "CHANGELOG.md"
TYPES = {"breaking": "Breaking changes", "added": "Added", "perf": "Performance", "fixed": "Bug fixes"}
NAME = re.compile(rf"([0-9]+)\.({'|'.join(TYPES)})\.md")
PR_URL = "https://github.com/kornia/kornia-rs/pull/{}"
UNRELEASED = "## [Unreleased]"


def fragments() -> list:
    if not FRAGMENTS.is_dir():
        return []
    return sorted(p for p in FRAGMENTS.iterdir() if p.name != "README.md")


def validate() -> list:
    errors = []
    for p in fragments():
        if not p.is_file() or not NAME.fullmatch(p.name):
            errors.append(f"changelog.d/{p.name}: name must be <PR>.({'|'.join(TYPES)}).md (rename +draft names to the PR number)")
            continue
        text = p.read_text(encoding="utf-8").strip()
        if not text:
            errors.append(f"changelog.d/{p.name}: empty")
        elif text.startswith(("#", "- ", "* ")):
            errors.append(f"changelog.d/{p.name}: no heading or outer bullet; the build adds both")
    return errors


def render() -> str:
    # Identical text in several PRs' fragments becomes one bullet crediting all of them.
    by_type = {t: {} for t in TYPES}
    for p in fragments():
        m = NAME.fullmatch(p.name)
        if m:
            text = p.read_text(encoding="utf-8").strip()
            by_type[m.group(2)].setdefault(text, []).append(int(m.group(1)))
    out = []
    for t, heading in TYPES.items():
        if by_type[t]:
            out.append(f"### {heading}\n")
            for text, prs in sorted(by_type[t].items(), key=lambda kv: min(kv[1])):
                links = ", ".join(f"[#{n}]({PR_URL.format(n)})" for n in sorted(prs))
                body = text.replace("\n", "\n  ")
                out.append(f"- {body} ({links})")
            out.append("")
    return "\n".join(out).strip()


def changed(base_ref: str) -> set:
    base = subprocess.run(["git", "merge-base", base_ref, "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    diff = subprocess.run(["git", "diff", "-M", "--name-only", f"{base}..HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout
    return set(diff.split())


def check(args) -> int:
    errors = validate()
    if not args.skip_presence:
        paths = changed(args.base_ref)
        mine = [p for p in paths if p.startswith("changelog.d/") and NAME.fullmatch(Path(p).name)
                and Path(p).name.startswith(f"{args.pr_number}.") and (ROOT / p).is_file()]
        if not mine:
            errors.append(f"no changelog.d/{args.pr_number}.<type>.md in this PR; add one for a user-visible change, "
                          "or ask a maintainer for the no-changelog label (internal, CI-only or release PRs)")
        if "CHANGELOG.md" in paths:
            errors.append("ordinary PRs must not edit CHANGELOG.md; only labeled release PRs assemble it")
    for e in errors:
        print(f"::error::{e}")
    if not errors:
        print("changelog fragments OK")
    return 1 if errors else 0


def build(args) -> int:
    errors = validate()
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    text = CHANGELOG.read_text(encoding="utf-8")
    start = text.index(UNRELEASED) + len(UNRELEASED)
    end = text.index("\n## [", start)
    legacy = text[start:end].strip()
    body = "\n\n".join(s for s in (legacy, render()) if s)
    if not body:
        print("nothing to release: [Unreleased] and changelog.d/ are empty", file=sys.stderr)
        return 1
    section = f"{UNRELEASED}\n\n## [{args.version}] — {args.date}\n\n{body}\n"
    CHANGELOG.write_text(text[: text.index(UNRELEASED)] + section + text[end:], encoding="utf-8")
    for p in fragments():
        p.unlink()
    print(f"CHANGELOG.md: [{args.version}] written; fragments removed. Add the compare link (see release skill).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--base-ref", required=True)
    c.add_argument("--pr-number", required=True, type=int)
    c.add_argument("--skip-presence", action="store_true")
    sub.add_parser("preview")
    b = sub.add_parser("build")
    b.add_argument("--version", required=True)
    b.add_argument("--date", required=True, help="release date, YYYY-MM-DD; never invented")
    args = ap.parse_args()
    if args.cmd == "check":
        return check(args)
    if args.cmd == "preview":
        errors = validate()
        print("\n".join(errors) if errors else (render() or "(no pending fragments)"))
        return 1 if errors else 0
    return build(args)


if __name__ == "__main__":
    sys.exit(main())
