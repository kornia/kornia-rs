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
    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    if not errors:
        print("skills OK")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
