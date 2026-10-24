Title: refactor(vlm): clearer BF16 fallback warning, drop dead helper

Fixes #9995

## Changes Made
- `device.rs`: the FP16 fallback warning now says why FP16 can be worse (attention logits may overflow).
- `device.rs`: removes `cuda_supports_bf16`, which is dead code.

## How Was This Tested?
```text
$ cargo clippy -p kornia-vlm -- -D warnings
Finished `dev` profile
$ cargo test -p kornia-vlm
test result: ok.
```

## Self-review
```text
Verdict: approve
Skill gaps: none
```

## Author walkthrough
- **Per file:** `device.rs`: reworded the warning; removed an unused function.
- **Riskiest part and how I checked it:** none, the helper had no callers and clippy is clean.

## AI Usage Disclosure
- **Agent + version:** Claude Code 2.1
- **Model:** claude-sonnet-5-5
- **Effort level:** low
- **Skills applied:** kornia-coding-style
- **Human vs AI part:** human asked for the wording change; agent wrote the diff
