## 📝 Description

**⚠️ Issue Link Required**: This PR must be linked to an approved and assigned issue. See [Contributing Guide](CONTRIBUTING.md#pull-request) for details.

**Fixes/Relates to:** # (issue number)

<!-- User-visible change: add changelog.d/<PR>.<type>.md (breaking, added, perf, fixed); see changelog.d/README.md.
Internal-only: say so here and ask for the no-changelog label. Do not edit CHANGELOG.md. -->

**Important**:
- Ensure you are assigned to the linked issue before submitting this PR
- This PR should strictly implement what the linked issue describes
- Do not include changes beyond the scope of the linked issue

---

## 🛠️ Changes Made
- [ ] Item 1
- [ ] Item 2

---

## 🧪 How Was This Tested?
- [ ] **Unit Tests:** (List new/updated tests)
- [ ] **Manual Verification:** (Describe the steps you took)
- [ ] **Performance/Edge Cases:** (How does this handle nulls, large data, etc.?)

<!-- Paste the commands you ran and their output. Say what failed or was not run.
Perf-sensitive change: before/after `cargo bench` numbers (see kornia-code-review). -->
```text
$ pixi run pr-check
```

---

## 🔍 Self-review (required)

<!-- Run the kornia-code-review skill on `git diff main...HEAD` before opening the PR
(e.g. tell your agent: "self-review my diff with .claude/skills/kornia-code-review"),
fix what it finds, and paste its final output here. The PR check fails while this
section has no verdict. -->
```text
Verdict:

Skill gaps:
```

---

## 🧠 Author walkthrough (required, written by you, not by AI)

<!-- In your own words: for each changed file, what the change does and why the issue
needs it. Then the riskiest part and how you checked it. Reviewers will ask about
specific lines; if you cannot explain a line, remove it. The PR check fails while
these lines are empty. -->
- **Per file:**
- **Riskiest part and how I checked it:**

---

## 🕵️ AI Usage Disclosure

<!-- Exact names, e.g. `Claude Code 2.x`, `claude-opus-5-5`, `high`. No AI used: write
`none` for Agent + version and leave Model, Effort level and Skills applied empty. -->
- **Agent + version:**
- **Model:**
- **Effort level:**
- **Skills applied:** (names from `.claude/skills/`)
- **Human vs AI part:** (which part was done by a human, if any; e.g. "human diagnosed the bug and reviewed the diff, agent wrote the fix and tests")

---

## 🚦 Checklist
- [ ] I followed the relevant skills in `.claude/skills/` (see AGENTS.md), pasted the `kornia-code-review` self-review above, and `pixi run pr-check` passes
- [ ] Every change is required by the linked issue; nothing out of scope
- [ ] I reviewed every line myself and can explain each change without AI (see AI_POLICY.md: you are the sole responsible author)
- [ ] I am assigned to the linked issue (required before PR submission)
- [ ] The linked issue has been approved by a maintainer
- [ ] This PR strictly implements what the linked issue describes (no scope creep)
- [ ] I have performed a **self-review** of my code (no "ghost" variables or hallucinations).
- [ ] My code follows the existing style guidelines of this project.
- [ ] I have commented my code, particularly in hard-to-understand areas.
- [ ] I have added tests that prove my fix is effective or that my feature works.
- [ ] (Optional) I have attached screenshots/recordings for UI changes.

---

## 💭 Additional Context
Add any other context or screenshots about the pull request here.
