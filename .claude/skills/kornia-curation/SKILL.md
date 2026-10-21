---
name: kornia-curation
description: Use when a change touches a fact the skills, AGENTS.md or CONTRIBUTING.md describe (MSRV, publish set, pixi tasks, CI commands, features, crate layout), when you hit a repo trap worth recording, or when editing .claude/skills — keeps the guidance true and current. Trigger on "update the skills", "docs are stale", "add a skill", or any edit to Cargo.toml, pixi.toml, scripts/ or .github/workflows/.
---

# Keeping kornia-rs guidance current

Stale guidance is worse than none: agents follow it. Guidance points at the
source of truth; it restates a fact only when `scripts/check_skills.py`
cross-checks it.

## Sources of truth

| Fact | Source | Restated in skills? |
|---|---|---|
| MSRV | root `Cargo.toml` `rust-version` | yes, in skills, AGENTS.md, CONTRIBUTING.md, Copilot instructions |
| Publish set / order | `scripts/release_rust.sh` (`PUBLISH_ORDER`, `NOT_PUBLISHED`) | no — link only; a sentence calling a `PUBLISH_ORDER` crate (`kornia-x` or short name) excluded / not published fails the check |
| pixi tasks | `pixi.toml` `[tasks]` / `[feature.*.tasks]` | yes, every `pixi run [-e env] <task>` is checked |
| CI commands | `.github/workflows/*.yml` | summarised; workflow file wins |
| Crates | `crates/*` | yes, names in `kornia-arch` are checked |
| Paths cited in skills | the tree | yes, backticked paths under `crates/`, `scripts/`, `kornia-py/`, `kornia-cpp/`, `docs/`, `.github/` |
| Review eval cases | `kornia-code-review/evals/` | yes, files present and `patch.diff` applies |
| Features | each crate's `Cargo.toml` | summarised |
| SIMD rules | `crates/kornia-imgproc/SIMD.md` | link only |

## Rules

1. **Same-PR updates.** A PR that changes a source above updates every skill
   and doc that describes it. `pixi run skills-check` (and the pre-commit hook,
   which also fires on `Cargo.toml`, `scripts/release_rust.sh` and workflow
   edits) catches the checked facts; the rest is on the author and reviewer.
2. **Link, don't copy.** New guidance names the file that owns the fact. If you
   must restate a fact, add its check to `scripts/check_skills.py` in the same PR.
3. **Record traps.** A non-obvious failure hit during a task (silent skip,
   wrong default, misleading error) becomes one line in the owning skill: the
   invariant and how to detect it. No narratives, no dates.
4. **Reviews feed skills.** Every `kornia-code-review` run ends with a
   "Skill gaps" section. Each gap becomes a one-line rule in the owning skill;
   each defect a review missed becomes a case in
   `.claude/skills/kornia-code-review/evals/`. A skill edit is done only when
   all eval cases still pass.
5. **Delete what is wrong.** Remove or fix stale lines; never add a
   contradicting line next to them.
6. **New skill**: directory name == frontmatter `name`, description starts with
   "Use " and says when to trigger, row added to the AGENTS.md table, and
   `REQUIRED` in `scripts/check_skills.py` if contributors must read it.
7. **Short.** A skill is a checklist, not a tutorial. Cut before you add.
