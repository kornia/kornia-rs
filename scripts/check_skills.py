#!/usr/bin/env python3
"""Validate agent skills under .claude/skills (agentskills.io format).

Structure: every skill dir has a SKILL.md whose frontmatter `name` matches the
directory and whose description starts with "Use "; required skills exist and
AGENTS.md references them; cited repo paths exist.

Facts (restated facts must match their source of truth, see kornia-curation):
MSRV (skills, AGENTS.md, CONTRIBUTING.md, copilot instructions) == root
Cargo.toml `rust-version`; every `pixi run <task>` exists in
pixi.toml; the kornia-arch crate table matches crates/*; no skill names a crate
from PUBLISH_ORDER in scripts/release_rust.sh as excluded from publishing.

This cannot check that guidance is complete or that it was applied.
"""
import re
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


def publish_order() -> set:
    m = re.search(r"^PUBLISH_ORDER=\((.*?)\)", (ROOT / "scripts" / "release_rust.sh").read_text(), re.S | re.M)
    return set(m.group(1).split()) if m else set()


def main() -> int:
    errors = []
    m = re.search(r'^rust-version\s*=\s*"([^"]+)"', (ROOT / "Cargo.toml").read_text(), re.M)
    msrv = m.group(1) if m else None
    if not msrv:
        errors.append("could not read rust-version from Cargo.toml")
    pixi = (ROOT / "pixi.toml").read_text()
    crates = {p.name for p in (ROOT / "crates").iterdir() if (p / "Cargo.toml").is_file()}
    published = publish_order()
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
        text = texts[skill.name] = md.read_text()
        fm = frontmatter(text)
        if fm.get("name") != skill.name:
            errors.append(f"{skill.name}: frontmatter name {fm.get('name')!r} != directory")
        if not fm.get("description", "").startswith("Use "):
            errors.append(f"{skill.name}: description should start with 'Use ' (trigger wording)")
        for ref in re.findall(PATH_REF, text):
            if not (ROOT / ref.rstrip("/")).exists():
                errors.append(f"{skill.name}: referenced path does not exist: {ref}")
        for task in sorted(set(re.findall(r"pixi run ([a-z0-9-]+)", text))):
            if not re.search(rf"^{re.escape(task)}\s*=", pixi, re.M):
                errors.append(f"{skill.name}: unknown pixi task: {task}")
        for line in text.splitlines():
            if re.search(r"exclud", line, re.I) and re.search(r"publish", line, re.I):
                for crate in sorted(published & set(re.findall(r"kornia-[a-z0-9-]+", line))):
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
        docs[doc] = (ROOT / doc).read_text()
    for where, text in docs.items():
        for v in re.findall(r"MSRV\W{0,6}v?(\d+\.\d+(?:\.\d+)?)", text):
            if v != msrv:
                errors.append(f"{where}: MSRV {v} != Cargo.toml rust-version {msrv}")

    agents = (ROOT / "AGENTS.md").read_text()
    for name in REQUIRED:
        if f"`{name}`" not in agents:
            errors.append(f"AGENTS.md does not reference skill `{name}`")
    link = ROOT / ".agents" / "skills"
    if not link.exists() or link.resolve() != SKILLS.resolve():
        errors.append(".agents/skills must point to .claude/skills")
    for doc in ("CONTRIBUTING.md", ".github/copilot-instructions.md", ".github/pull_request_template.md"):
        if ".claude/skills" not in (ROOT / doc).read_text():
            errors.append(f"{doc} does not reference .claude/skills")

    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    if not errors:
        print("skills OK")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
