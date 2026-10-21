# 🤖 Kornia-rs AI & Authorship Policy

**Version:** 1.0
**Enforcement:** Strict
**Applicability:** All Pull Requests (Human & Bot)

## 1. Core Philosophy

Kornia-rs accepts AI-assisted code (e.g., using Copilot, Cursors-AI, etc.), but strictly rejects AI-generated contributions where the submitter acts merely as a proxy. The submitter is the **Sole Responsible Author** for every line of code, comment, and design decision.

## 2. The 3 Laws of Contribution

### Law 1: Proof of Verification

AI tools frequently write code that looks correct but fails execution. Therefore, "vibe checks" are insufficient.

**Requirement:** Every PR introducing functional changes must include a pasted snippet of the local test logs (e.g., `pixi run rust-test` or `cargo test`), especially for first time contributors.

**Failure Condition:** If a PR lacks execution proof and contains complex logic, it will be flagged as **Unverified**.

**Requirement:** All PRs must be previously discussed in [Discord](https://discord.gg/HfnywwpBnD) or via a [GitHub issue](https://github.com/kornia/kornia-rs/issues) before implementation. The PR must reference the discussion or issue.

**Requirement:** Implementations must be based on an existing library reference (e.g., Rust crates from crates.io, OpenCV, existing Rust CV libraries, etc.) that must be provided in the PR description for verification. This reference serves as proof that the implementation follows established algorithms and is not hallucinated.

### Law 2: The Hallucination & Redundancy Ban

AI models often hallucinate comments or reinvent existing utilities.

**Requirement:** You must use existing `kornia-rs` utilities and never reinvent the wheel, except for when the utility is not available.

**Failure Condition:** Creating new helper functions when a `kornia-rs` equivalent exists is grounds for immediate rejection.

**Failure Condition:** "Ghost Comments" (comments explaining logic that was deleted or doesn't exist) will result in a request for a full manual rewrite. Redundant comments are not allowed. Example: "This function returns the input image".

**Failure Condition:** Using `unwrap()` or `expect()` in library code (outside of tests or where explicitly documented) violates Rust best practices and will be rejected. Use `Result<T, E>` with proper error handling.

### Law 3: The "Explain It" Standard

**Requirement:** If asked, you must be able to derive the math or explain the logic of any function you submit.

**Failure Condition:** Answering a review question with "That's what the AI outputted" or "I don't know, it works" leads to immediate closure.

**Requirement:** The PR's **Author walkthrough** is written by you, in your own words: each changed file, why the issue needs it, the riskiest part and how you checked it.

## 3. Instructions for Reviewers (human or AI)

Review with the [`kornia-code-review`](.claude/skills/kornia-code-review/SKILL.md) skill: scope against the linked issue first, then copies, latency, benchmarks, correctness and the coding-style rules, each finding tagged measured or inferred.

**Do not speculate about whether or how AI was used.** Disclosure belongs to the author, in the PR template. Judge the change: a hallucinated comment, a reinvented utility or an `unwrap()` in library code is a defect whoever wrote it, and poor code or a long, structured description is not evidence of AI use. Only demonstrable deception (verification that never ran, fabricated output) is grounds for closure.

To check that the author understands the change, use the Author walkthrough and ask about specific lines (Law 3), rather than guessing at provenance.

## 4. Additional Resources

For comprehensive guidance on contributing to Kornia-rs, including development workflows, code quality standards, testing practices, and AI-assisted development best practices, see the [Best Practices section](CONTRIBUTING.md#best-practices) in `CONTRIBUTING.md`.
