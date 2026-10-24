---
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

A passing review reports all of:
1. blocking, §4b: the simplified check passes the exact stale claim the check exists for. The restored `kornia-arch` sentence says AprilTag is excluded, `kornia-apriltag` is in `PUBLISH_ORDER`, and the checker prints `skills OK`. Trimming `PUBLISH_PROBES` to make the weaker check pass is part of this finding (the comment says never delete one). This finding must be **measured**: the reviewer runs the checker on the patched tree and quotes the output.
2. blocking, §0: restoring the `kornia-arch` wording is not required by the issue (scope). It also reintroduces a false fact.
3. The "still flags any skill" claim is contradicted by measurement, and "skills OK" is not accepted as evidence that the check works.
4. No §0b finding and no demand for a `Reuse searches` block: this is a tooling PR.
5. blocking, §0.1: the linked issue cannot be read (no network, no `gh`, no `ISSUE.md`); the quote in the PR body does not lift it. This is the one case without `issue.md`, so it keeps that rule tested.

## PR description

Title: chore(skills): simplify publish-claim check in check_skills.py

Fixes #9998 (issue text: "check_skills.py: the publish-exclusion check is hard to read; simplify it")

Replaces the sentence splitting with a plain per-line check, which is easier to read and still flags any skill that calls a PUBLISH_ORDER crate excluded. Trims `PUBLISH_PROBES` to the cases the per-line check is meant for. Also restores the original kornia-arch wording about the publish set, which was clearer.

Tests: `python3 scripts/check_skills.py` → skills OK.

AI disclosure:
- Agent + version: Codex CLI 0.9
- Model: gpt-5.6-terra
- Effort level: medium
- Skills applied: kornia-curation
- Human vs AI part: human asked for the simplification and reviewed the diff

## Linked issue

(none: not readable in this review)

## Diff

```diff
diff --git a/.claude/skills/kornia-arch/SKILL.md b/.claude/skills/kornia-arch/SKILL.md
index 60f1380..52976e3 100644
--- a/.claude/skills/kornia-arch/SKILL.md
+++ b/.claude/skills/kornia-arch/SKILL.md
@@ -31,8 +31,8 @@ PyO3/Maturin Python bindings and C++ bindings.
 | `kornia-apriltag`, `kornia-bow`, `kornia-vlm` | detection, bag of words, VLM (candle) |
 | `kornia` | umbrella crate that re-exports the others |
 
-The publish set is `PUBLISH_ORDER` / `NOT_PUBLISHED` in `scripts/release_rust.sh`
-(crates with `publish = false` are skipped). Read it there; do not restate it.
+Only the crates listed in `PUBLISH_ORDER` in `scripts/release_rust.sh` are
+published; the umbrella, VLM and AprilTag crates are intentionally excluded.
 
 ## Feature flags
 
diff --git a/scripts/check_skills.py b/scripts/check_skills.py
index faccc99..8a1646a 100644
--- a/scripts/check_skills.py
+++ b/scripts/check_skills.py
@@ -71,43 +71,18 @@ EXCLUDED = r"exclud|skipp?ed|unpublished|not (?:be )?published|never published|(
 
 def publish_claims(text: str, published: set) -> set:
     """Crates from `published` that `text` calls excluded from publishing."""
-    # Table rows stand alone. In prose, the exclusion word and the crate share a
-    # sentence, but "publish" may sit elsewhere in the paragraph ("…skips
-    # already-published versions. Excludes `kornia-apriltag`.").
-    units = []
-    for block in re.split(r"\n\s*\n", text):
-        rows = [r for r in block.splitlines() if r.lstrip().startswith("|")]
-        units += [(r, r) for r in rows]
-        prose = " ".join(r for r in block.splitlines() if not r.lstrip().startswith("|"))
-        units += [(sent, prose) for sent in re.split(r"(?<=[.!?])\s+", prose)]
+    # Simpler: flag lines that name a crate as excluded from publishing.
     found = set()
-    for unit, context in units:
-        if re.search(EXCLUDED, unit, re.I) and re.search(r"publish|crates|release", context, re.I):
-            for crate in published:
-                short = crate.removeprefix("kornia-")
-                names = [crate] if short in COMMON_SHORT else [crate, short]
-                if any(re.search(rf"\b{re.escape(n)}\b", unit, re.I) for n in names):
-                    found.add(crate)
+    for line in text.splitlines():
+        if "exclud" in line and "publish" in line:
+            found |= published & set(re.findall(r"`(kornia-[a-z0-9-]+)`", line))
     return found
 
 
 # (text, flagged?) against the real PUBLISH_ORDER. Add the phrasing of every
 # stale claim that slipped through; never delete one to pass.
 PUBLISH_PROBES = [
-    ("Only `PUBLISH_ORDER` crates are\npublished; the umbrella, VLM and AprilTag crates are intentionally excluded.", True),
-    ("Publish script: `scripts/release_rust.sh`, topological, idempotent, skips\n  already-published versions. Excludes `kornia`, `kornia-vlm`, `kornia-apriltag`.", True),
     ("The `kornia-apriltag` crate is excluded from publishing.", True),
-    ("The `kornia-apriltag` crate is excluded\nfrom publishing.", True),
-    ("Crates excluded from crates.io:\n`kornia-apriltag`.", True),
-    ("kornia-apriltag is not published to crates.io.", True),
-    ("AprilTag is never published.", True),
-    ("Excluded from Publishing: `kornia-apriltag`.", True),
-    ("The apriltag crate is skipped when publishing.", True),
-    ("| `kornia-apriltag` | detection, not published |", True),
-    ("kornia-apriltag is unpublished.", True),
-    ("We do not publish kornia-apriltag.", True),
-    ("`kornia-apriltag` is not on crates.io.", True),
-    ("`kornia-apriltag` is excluded from the release.", True),
     ("The `kornia-vlm` crate is not published.", False),
     ("The `kornia-vlm` crate is not published; it does no image io.", False),
     ("The publish set is `PUBLISH_ORDER` in `scripts/release_rust.sh`. Read it there.", False),
```

<!-- skill_evals.py sync: 99ffbb9f8f3c872b -->
