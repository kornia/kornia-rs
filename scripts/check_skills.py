#!/usr/bin/env python3
"""Validate agent skills under .claude/skills (agentskills.io format).

Structure: every skill dir has a SKILL.md whose frontmatter `name` matches the
directory and whose description starts with "Use "; required skills exist and
AGENTS.md references them; cited repo paths exist.

Facts (restated facts must match their source of truth, see kornia-curation):
MSRV (skills, AGENTS.md, CONTRIBUTING.md, copilot instructions) == root
Cargo.toml `rust-version`; every `pixi run <task>` exists in
pixi.toml; the kornia-arch crate table matches crates/*; no skill names a crate
from PUBLISH_ORDER in scripts/release_rust.sh as excluded from publishing;
review eval cases (`<skill>/evals/<case>/`) are complete, runnable by
`claude plugin eval`, and their patch applies; no eval key names a
backtick term the skill under test also names.
The publish check first runs on PUBLISH_PROBES; a change that stops it rejecting
one fails here, not silently.

This cannot check that guidance is complete or that it was applied.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKILLS = ROOT / ".claude" / "skills"
REQUIRED = [
    "kornia-arch",
    "kornia-workflow",
    "kornia-coding-style",
    "kornia-api-flavor",
    "kornia-code-review",
    "kornia-curation",
    "kornia-simd",
    "kornia-bindings",
    "release",
]
PATH_REF = r"`((?:crates|scripts|kornia-py|kornia-cpp|docs|\.github)/[A-Za-z0-9_./-]+)`"


def frontmatter(text: str) -> dict:
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        return {}
    fields = {}
    for line in m.group(1).splitlines():
        k, sep, v = line.partition(":")
        if sep:
            fields[k.strip()] = v.strip()
    return fields


def pixi_tasks() -> set:
    tasks, section = set(), ""
    for line in (ROOT / "pixi.toml").read_text(encoding="utf-8").splitlines():
        if m := re.match(r"^\[(.+)\]\s*$", line):
            section = m.group(1)
        elif re.fullmatch(r"tasks|feature\.[^.]+\.tasks", section) and (m := re.match(r"^([A-Za-z0-9_-]+)\s*=", line)):
            tasks.add(m.group(1))
    return tasks


def publish_order() -> set:
    m = re.search(r"^PUBLISH_ORDER=\((.*?)\)", (ROOT / "scripts" / "release_rust.sh").read_text(encoding="utf-8"), re.S | re.M)
    return set(m.group(1).split()) if m else set()


# Short names that are ordinary words ("image io") are only matched as `kornia-x`.
# Skill vocabulary that eval keys may name without leaking an answer.
KEY_TERMS_ALLOWED = {"Reuse searches", "kornia-arch", "pub(crate)", "width", "cuda"}
COMMON_SHORT = {"3d", "algebra", "bow", "image", "io", "tensor"}
EXCLUDED = r"exclud|skipp?ed|unpublished|not (?:be )?published|never published|(?:not|n't) publish|not on crates\.io|not released"


def publish_claims(text: str, published: set) -> set:
    """Crates from `published` that `text` calls excluded from publishing."""
    # Table rows stand alone. In prose, the exclusion word and the crate share a
    # sentence, but "publish" may sit elsewhere in the paragraph ("…skips
    # already-published versions. Excludes `kornia-apriltag`.").
    units = []
    for block in re.split(r"\n\s*\n", text):
        rows = [r for r in block.splitlines() if r.lstrip().startswith("|")]
        units += [(r, r) for r in rows]
        prose = " ".join(r for r in block.splitlines() if not r.lstrip().startswith("|"))
        units += [(sent, prose) for sent in re.split(r"(?<=[.!?])\s+", prose)]
    found = set()
    for unit, context in units:
        if re.search(EXCLUDED, unit, re.I) and re.search(r"publish|crates|release", context, re.I):
            for crate in published:
                short = crate.removeprefix("kornia-")
                names = [crate] if short in COMMON_SHORT else [crate, short]
                if any(re.search(rf"\b{re.escape(n)}\b", unit, re.I) for n in names):
                    found.add(crate)
    return found


# (text, flagged?) against the real PUBLISH_ORDER. Add the phrasing of every
# stale claim that slipped through; never delete one to pass.
PUBLISH_PROBES = [
    ("Only `PUBLISH_ORDER` crates are\npublished; the umbrella, VLM and AprilTag crates are intentionally excluded.", True),
    ("Publish script: `scripts/release_rust.sh`, topological, idempotent, skips\n  already-published versions. Excludes `kornia`, `kornia-vlm`, `kornia-apriltag`.", True),
    ("The `kornia-apriltag` crate is excluded from publishing.", True),
    ("The `kornia-apriltag` crate is excluded\nfrom publishing.", True),
    ("Crates excluded from crates.io:\n`kornia-apriltag`.", True),
    ("kornia-apriltag is not published to crates.io.", True),
    ("AprilTag is never published.", True),
    ("Excluded from Publishing: `kornia-apriltag`.", True),
    ("The apriltag crate is skipped when publishing.", True),
    ("| `kornia-apriltag` | detection, not published |", True),
    ("kornia-apriltag is unpublished.", True),
    ("We do not publish kornia-apriltag.", True),
    ("`kornia-apriltag` is not on crates.io.", True),
    ("`kornia-apriltag` is excluded from the release.", True),
    ("The `kornia-vlm` crate is not published.", False),
    ("The `kornia-vlm` crate is not published; it does no image io.", False),
    ("The publish set is `PUBLISH_ORDER` in `scripts/release_rust.sh`. Read it there.", False),
    ("`kornia-apriltag` detects tags. Crates with `publish = false` are skipped.", False),
]


def main() -> int:
    errors = []
    published = publish_order()
    for probe, flagged in PUBLISH_PROBES:
        if bool(publish_claims(probe, published)) != flagged:
            errors.append(f"publish check self-test: {'missed' if flagged else 'false positive on'} probe {probe!r}")
    m = re.search(r'^rust-version\s*=\s*"([^"]+)"', (ROOT / "Cargo.toml").read_text(encoding="utf-8"), re.M)
    msrv = m.group(1) if m else None
    if not msrv:
        errors.append("could not read rust-version from Cargo.toml")
    tasks = pixi_tasks()
    crates = {p.name for p in (ROOT / "crates").iterdir() if (p / "Cargo.toml").is_file()}
    if not published:
        errors.append("could not parse PUBLISH_ORDER from scripts/release_rust.sh")

    for name in REQUIRED:
        if not (SKILLS / name / "SKILL.md").is_file():
            errors.append(f"missing required skill: {name}")
    texts = {}
    for skill in sorted(p for p in SKILLS.iterdir() if p.is_dir()):
        md = skill / "SKILL.md"
        if not md.is_file():
            errors.append(f"{skill.name}: missing SKILL.md")
            continue
        text = texts[skill.name] = md.read_text(encoding="utf-8")
        fm = frontmatter(text)
        if fm.get("name") != skill.name:
            errors.append(f"{skill.name}: frontmatter name {fm.get('name')!r} != directory")
        if not fm.get("description", "").startswith("Use "):
            errors.append(f"{skill.name}: description should start with 'Use ' (trigger wording)")
        for ref in re.findall(PATH_REF, text):
            if not (ROOT / ref.rstrip("/")).exists():
                errors.append(f"{skill.name}: referenced path does not exist: {ref}")
        for task in sorted(set(re.findall(r"pixi run (?:(?:-e|--environment) \S+ )?([A-Za-z0-9_-]+)", text))):
            if task not in tasks:
                errors.append(f"{skill.name}: unknown pixi task: {task}")
        for crate in sorted(publish_claims(text, published)):
            errors.append(f"{skill.name}: says {crate} is excluded from publishing, but it is in PUBLISH_ORDER")

    # Crate table rows look like "| `kornia-x`, `kornia-y` | purpose |".
    table = set()
    for row in texts.get("kornia-arch", "").splitlines():
        if row.startswith("| `"):
            table |= set(re.findall(r"`(kornia(?:-[a-z0-9-]+)?)`", row.split("|")[1]))
    for c in sorted(table - crates):
        errors.append(f"kornia-arch: crate table lists {c}, not in crates/")
    for c in sorted(crates - table):
        errors.append(f"kornia-arch: crates/{c} missing from crate table")

    docs = {f"{n} skill": s for n, s in texts.items()}
    for doc in ("AGENTS.md", "CONTRIBUTING.md", ".github/copilot-instructions.md"):
        docs[doc] = (ROOT / doc).read_text(encoding="utf-8")
    for where, text in docs.items():
        for v in re.findall(r"MSRV\W{0,6}v?(\d+\.\d+(?:\.\d+)?)", text):
            if v != msrv:
                errors.append(f"{where}: MSRV {v} != Cargo.toml rust-version {msrv}")

    # Review eval cases must be complete and still apply to the tree.
    # `claude plugin eval` needs prompt.md, case.yaml, an executable scaffold.sh
    # and at least one grader; results/ is its local output.
    for case in sorted(p for p in SKILLS.glob("*/evals/*") if p.is_dir() and p.name not in ("results", "heldout", "calibration")):
        for f in ("patch.diff", "pr.md", "expected.md", "prompt.md", "case.yaml", "scaffold.sh"):
            if not (case / f).is_file():
                errors.append(f"eval {case.name}: missing {f}")
        if (case / "scaffold.sh").is_file() and not os.access(case / "scaffold.sh", os.X_OK):
            errors.append(f"eval {case.name}: scaffold.sh is not executable")
        if not list(case.glob("graders/*.md")):
            errors.append(f"eval {case.name}: no graders/*.md")
        patch = case / "patch.diff"
        if patch.is_file():
            r = subprocess.run(["git", "apply", "--check", str(patch)], cwd=ROOT, capture_output=True, text=True)
            if r.returncode != 0:
                errors.append(f"eval {case.name}: patch.diff no longer applies: {r.stderr.strip()}")
        merged = case / "merged.diff"
        if merged.is_file():
            r = subprocess.run(["git", "apply", "-R", "--check", str(merged)], cwd=ROOT, capture_output=True, text=True)
            if r.returncode != 0:
                errors.append(f"eval {case.name}: merged.diff is no longer in the tree: {r.stderr.strip()}")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT / "scripts"))
    from skill_evals import stale_precision
    for name in stale_precision():
        errors.append(f"eval {name}: graders/precision.md is stale; run `python3 scripts/skill_evals.py sync`")

    # An eval key must not leak into the skill it grades: a name backticked in an
    # expected.md that the skill also uses (backticked, or bare for identifiers
    # such as `check_size`) turns a case into a recall test of the skill text.
    for skill_md in SKILLS.glob("*/SKILL.md"):
        text = skill_md.read_text(encoding="utf-8")
        for key in sorted(skill_md.parent.glob("evals/*/expected.md")):
            for tok in sorted(set(re.findall(r"`([^`\n]{4,})`", key.read_text(encoding="utf-8"))) - KEY_TERMS_ALLOWED):
                ident = re.fullmatch(r"[A-Za-z_][\w:]*_[\w:]*", tok)
                if f"`{tok}`" in text or (ident and re.search(rf"(?<![\w:]){re.escape(tok)}(?![\w:])", text)):
                    errors.append(f"{skill_md.parent.name}/SKILL.md names `{tok}` from eval key {key.parent.name}/expected.md")

    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    for name in REQUIRED:
        if f"`{name}`" not in agents:
            errors.append(f"AGENTS.md does not reference skill `{name}`")
    link = ROOT / ".agents" / "skills"
    if not link.exists() or link.resolve() != SKILLS.resolve():
        errors.append(".agents/skills must point to .claude/skills")
    for doc in ("CONTRIBUTING.md", ".github/copilot-instructions.md", ".github/pull_request_template.md"):
        if ".claude/skills" not in (ROOT / doc).read_text(encoding="utf-8"):
            errors.append(f"{doc} does not reference .claude/skills")

    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    if not errors:
        print("skills OK")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
