#!/usr/bin/env python3
"""Validate agent skills under .claude/skills (agentskills.io format).

Checks: every skill dir has a SKILL.md whose frontmatter `name` matches the
directory and has a non-empty `description`; every required skill exists;
AGENTS.md references each skill; pixi tasks referenced by the workflow skill exist.
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
    "kornia-simd",
    "kornia-bindings",
    "release",
]


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


def main() -> int:
    errors = []
    for name in REQUIRED:
        if not (SKILLS / name / "SKILL.md").is_file():
            errors.append(f"missing required skill: {name}")
    for skill in sorted(p for p in SKILLS.iterdir() if p.is_dir()):
        md = skill / "SKILL.md"
        if not md.is_file():
            errors.append(f"{skill.name}: missing SKILL.md")
            continue
        fm = frontmatter(md.read_text())
        if fm.get("name") != skill.name:
            errors.append(f"{skill.name}: frontmatter name {fm.get('name')!r} != directory")
        if not fm.get("description"):
            errors.append(f"{skill.name}: empty description")
    agents = (ROOT / "AGENTS.md").read_text()
    for name in REQUIRED:
        if f"`{name}`" not in agents:
            errors.append(f"AGENTS.md does not reference skill `{name}`")
    pixi = (ROOT / "pixi.toml").read_text()
    wf = (SKILLS / "kornia-workflow" / "SKILL.md")
    if wf.is_file():
        for task in sorted(set(re.findall(r"pixi run ([a-z0-9-]+)", wf.read_text()))):
            if not re.search(rf"^{re.escape(task)}\s*=", pixi, re.M):
                errors.append(f"kornia-workflow references unknown pixi task: {task}")
    link = ROOT / ".agents" / "skills"
    if not link.exists() or link.resolve() != SKILLS.resolve():
        errors.append(".agents/skills must point to .claude/skills")
    for skill in sorted(p for p in SKILLS.iterdir() if p.is_dir()):
        md = skill / "SKILL.md"
        if not md.is_file():
            continue
        text = md.read_text()
        if not frontmatter(text).get("description", "").startswith("Use "):
            errors.append(f"{skill.name}: description should start with 'Use ' (trigger wording)")
        for ref in re.findall(r"`((?:crates|scripts|kornia-py|kornia-cpp|docs|\.github)/[A-Za-z0-9_./-]+)`", text):
            if not (ROOT / ref.rstrip("/")).exists():
                errors.append(f"{skill.name}: referenced path does not exist: {ref}")
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
