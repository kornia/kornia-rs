title:	[Bug]: Missing PartialEq derive on Lie group structs breaks compilation with --all-features
state:	OPEN
author:	namanguptagit
labels:	bug, help wanted
assignees:	namanguptagit
projects:	
milestone:	
number:	728
--
### 🐛 Describe the bug

The kornia-algebra crate currently fails to compile when the --all-features flag is passed. This is because the optional approx crate dependency is enabled, and the approx trait AbsDiffEq requires PartialEq. However, PartialEq is missing from several Lie group structs in crates/kornia-algebra/src/lie/*.rs.

### 🔄 Steps to Reproduce

```bash
1. Clone the repository
2. Navigate to crates/kornia-algebra or the workspace root.
3. Run the following command:
cargo check --all-targets --all-features -p kornia-algebra
```

### 💻 Minimal Code Example

```rust
// N/A - This is a compilation error in the library itself.
```

### ✅ Expected behavior

The crate kornia-algebra should compile successfully with --all-features enabled.

### ❌ Actual behavior

Compilation fails with errors similar to the following:

```
error[E0277]: can't compare `SO3F32` with `SO3F32`
   --> crates/kornia-algebra/src/lie/so3.rs:267:29
    |
267 | impl approx::AbsDiffEq for SO3F32 {
    |                            ^^^^^^ no implementation for `SO3F32 == SO3F32`
    |
    = help: the trait `PartialEq` is not implemented for `SO3F32`
```

### 🔧 Environment

```shell
- kornia-rs version:main (v0.1.10-rc.5-121-gc317f04 approx)
- Rust version (`rustc -V`):rustc 1.93.1 
- Cargo version (`cargo -V`):cargo 1.93.1 
- OS (e.g., Linux, macOS, Windows): macOS
- Target architecture (if cross-compiling):N/A
- Python version (if using Python bindings):N/A
```

### 📝 Additional context

The proposed fix is to simply add PartialEq to the #[derive(Debug, Clone, Copy)] list for the following structs in crates/kornia-algebra/src/lie/:

SO2F32 (so2.rs)
SO3F32 (so3.rs)
RxSO3F32 (rxso3.rs)
Sim3F32 (sim3.rs)
SE2F32 (se2.rs)
SE3F32 (se3.rs)

### 🤝 Contribution Intent

- [x] I plan to submit a PR to fix this bug
- [ ] I'm reporting this bug but not planning to fix it
