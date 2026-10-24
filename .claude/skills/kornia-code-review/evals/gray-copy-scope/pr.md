Title: refactor(imgproc): simplify gray_from_rgb u8 validation

Fixes #9999 (issue text: "gray_from_rgb u8: inline the size check into the impl for readability")

Inlines the size check into the u8 impl so the validation is simpler and local to the kernel call, and keeps the source in a local buffer for better cache locality. Should be slightly faster.

Tests: cargo test -p kornia-imgproc passes.

AI disclosure:
- Agent + version: Cursor 1.4
- Model: gpt-5
- Effort level: medium
- Skills applied: kornia-coding-style
- Human vs AI part: human reviewed
