# CUDA kernels in Rust (cuda-oxide)

The CUDA kernels in `crates/kornia-oxide-kernels/device/` are written in Rust
and compiled to PTX with NVIDIA's [cuda-oxide](https://github.com/NVIDIA/cuda-rust)
(the `cuda-oxide/` tree of NVIDIA/cuda-rust) ahead of time: the published
`kornia-oxide-kernels` crate ships the PTX, so building kornia with `cuda` from
crates.io needs only stable Rust. In a checkout the PTX is built by
`cargo run -p kornia-oxide-kernels --example build_ptx` (the cuda-oxide
toolchain below; no CUDA toolkit, GPU or driver). PTX is never committed. This
is the pattern cuda-oxide suggests for library crates
([NVIDIA/cuda-rust#1464](https://github.com/NVIDIA/cuda-rust/issues/1464)). Kernels not yet ported stay NVRTC-compiled at runtime
and keep working under the same feature. Roadmap:
[#1135](https://github.com/kornia/kornia-rs/issues/1135).

`kornia-oxide-kernels` owns the kernels and the PTX build; it is host-only data
with no dependencies. kornia-tensor's `cuda` feature owns the loader
(`kornia_tensor::cuda::oxide`), so every crate with kernels, kornia-tensor
included, reaches it without a cycle. The PTX is real only when a crate in the
build turns on `kornia-oxide-kernels/ptx`. Rule: a crate's `cuda` feature
turns on `ptx` exactly when that crate ships oxide kernels (kornia-imgproc
today; kornia-tensor's `cuda` joins when its `ops.rs` kernels migrate). Users
only ever set `cuda`; crates using kornia-tensor for device memory alone stay
on stable without the toolchain.

| Path | Role |
|---|---|
| `crates/kornia-oxide-kernels/device/src/<family>.rs` | Kernel source, one module per family (`color`, `resize`, ...). Own `[workspace]`, pinned nightly, manifest shipped as `Cargo.toml.in` so `cargo package` keeps the directory. |
| `crates/kornia-oxide-kernels/examples/build_ptx.rs` | Builds `device/` with cuda-oxide, gates the PTX and writes the stamped `kernels.ptx` (gitignored; the published crate ships it). |
| `crates/kornia-oxide-kernels/build.rs` | Scans `device/src` for `#[kernel]` items; with `ptx`, checks the stamp of `kernels.ptx` against `device/` and gates it. Writes `$OUT_DIR/kernels.ptx` + `$OUT_DIR/entries.rs`. Needs no toolchain. |
| `crates/kornia-oxide-kernels/src/lib.rs` | `PTX`, `IS_STUB`, `ENTRIES` and `entry::<NAME>` (the PTX entry name). |
| `crates/kornia-tensor/src/cuda/oxide.rs` | Loader: one module per device, `kernel(stream, entry::<NAME>) -> &'static CudaKernel`, lock-free after the first load. |
| `crates/kornia-imgproc/src/cuda/<family>/` | The family's existing public launchers; only their kernel fetch changes. |
| `crates/kornia-imgproc/benches/bench_cuda_oxide.rs` | Every oxide kernel vs its CPU path; one `CASES` row per op and size. |
| `scripts/cuda-oxide/install-toolchain.sh` | Installs the toolchain below (local and CI share it). |
| `.github/actions/cuda-oxide-ptx` | CI: cached toolchain + `build_ptx`; used by the lint, wheel and Rust release workflows. |
| `scripts/cuda-oxide/orin-test-bench.sh` | Jetson: parity tests (names derived from the PTX) + bench; `--base <ref>` A/B. |

## How it builds

- `examples/build_ptx.rs` (kernel authors, CI, the release):
  1. checks the toolchain: the pinned nightly via rustup, the backend `.so`
     and its `.backend` stamp (`<rev> <nightly>`, written by the install
     script) against `device/`'s pins;
  2. stages `device/` under `target/kornia-oxide-kernels/` (as `Cargo.toml`;
     `device/` itself never holds one, or `cargo package` would drop it) and
     runs `cargo build --release --locked` there under the pinned nightly and
     the backend, with the caller's `CARGO_*`/`RUSTC*`/`RUSTFLAGS` removed;
  3. gates the PTX (below) and writes `crates/kornia-oxide-kernels/kernels.ptx`,
     whose first line is a stamp: an FNV-1a hash of `device/src`,
     `Cargo.toml.in`, `Cargo.lock`, `rust-toolchain.toml`, `sm_75` and the
     nested build's flags. Cold on an Orin at `-j2`: ~70 s; after a kernel
     edit: ~2 s.
- `build.rs` (every build): writes `entries.rs` from the `#[kernel]` scan
  (sorted; a duplicate name is an error). Without `ptx` the PTX is empty
  (`IS_STUB`). With `ptx` it reads `kernels.ptx` (or `$KORNIA_OXIDE_PTX`, e.g.
  for a git dependency), fails if it is missing or its stamp does not match
  `device/` (an edited kernel without a rebuilt PTX), and gates it again.
- Release: the Rust release workflow builds the PTX, and
  `scripts/release_rust.sh` publishes the crate with `--features ptx
  --allow-dirty` (cargo counts the gitignored-but-included file as dirty), so
  the verify build checks the stamp inside the packaged copy.
- The PTX targets `sm_75` (PTX ISA 6.3): the driver JITs it for any
  Turing-or-newer GPU and any host CPU, so one file serves every wheel and
  platform. Maxwell/Pascal (Jetson Nano/TX2) cannot run ported ops.
- Nothing in a plain workspace build enables `cuda`: the CUDA examples
  (`cuda-fusion`, `cuda-camera-sift`, `cuda-camera-preprocess`) have their own
  `cuda` feature and build only with `--features cuda`.

### PTX gate (`kornia-oxide-kernels/ptx_gate.rs`, run by `build_ptx` and `build.rs`)

- module: `.target sm_75`; no `.extern` symbols; the `.entry` set equals the
  scanned `#[kernel]` set;
- per `.entry` and per non-inlined `.func`: every `ld`/`st` names a state
  space (a generic access means address-space inference was lost); no
  `fma`/float `mad` unless it is an entry in `FMA_ALLOWED` (empty); no
  libdevice (`__nv_*`) calls. A helper that trips this should be
  `#[inline(always)]`.

Errors read `gate: <function>: <reason>`. `kornia-oxide-kernels/tests/ptx_gate.rs`
covers it in the plain (non-CUDA) test suite.

## Toolchain

`scripts/cuda-oxide/install-toolchain.sh [PREFIX]` installs everything into
`PREFIX` (default `~/.cargo/cuda-oxide`) and writes `PREFIX/env.sh`; source it
before cargo. Pieces already in `PREFIX` at the right version are reused, so CI
caches `PREFIX`. It needs rustup; the backend archive is pinned and
sha256-checked. What it installs:

1. the nightly pinned in `device/rust-toolchain.toml`
   (`nightly-2026-10-11`, with `rust-src`, `rustc-dev`, `llvm-tools`). The
   backend runs the nightly's own `llc`/`opt` (LLVM 23), so no system LLVM;
2. the codegen backend `librustc_codegen_cuda.so`, built at the rev pinned in
   `device/Cargo.toml.in` (NVIDIA/cuda-rust `6d47ae4`; the script pins
   the archive's sha256 too). `build_ptx` refuses a backend whose `.backend`
   stamp names another rev or nightly.

The kernel crate is **device-only**: it depends on `cuda-device` and on
`cuda-macros` with `default-features = false`, so `#[kernel]` emits no host
launch helpers and cuda-host / cuda-core / cuda-bindings (bindgen of `cuda.h`)
drop out of its lock (8 packages). That is why neither `cuda.h` nor libclang is
needed. Bump the pin only together with a re-run of the parity tests and the
bench.

**`CUDA_OXIDE_NO_FMA=1`.** This cuda-oxide contracts `fmul`+`fadd` into `fma`
by default (like `nvcc --fmad=true`); the CPU twins round each op, so
`build_ptx` always sets it on the nested build (an outer value cannot change
it) and the gate still rejects any `fma`.

## Adding a kernel / migrating an NVRTC family

A second kernel PR (e.g. bicubic resize) does exactly this; nothing else:

1. **Kernel.** Add `#[kernel] pub unsafe fn <name>` to
   `crates/kornia-oxide-kernels/device/src/<family>.rs` (new family: add the
   file and `mod <family>;` to `main.rs`); `entry::<NAME>` appears by itself. `<name>` is the NVRTC entry it replaces, exactly; same
   parameters in the same order, so the existing launch code drives it.
   Rebuild the PTX with `cargo run -p kornia-oxide-kernels --example
   build_ptx`; until then any `ptx` build fails with a stale-stamp error.
   - raw pointers, not slices (slices lose the address space and add a bounds
     branch: 12-13% on resize upscale);
   - **all loads before the first store**: cuda-oxide emits `noalias` only for
     slice parameters, so raw-pointer loads/stores still serialize when
     interleaved (2x on resize upscale);
   - mirror each constant with a comment naming the CPU constant it copies;
   - backend gaps: `Ord::min` and `f32::min`/`max` work; `to_int_unchecked`
     does not compile (use `as`); no `__ldg` (a no-op on sm_87 anyway).
2. **Launcher.** In the family's public launcher replace the NVRTC kernel fetch
   with `oxide::kernel(stream, oxide::entry::<NAME>)?` (`use
   kornia_tensor::cuda::oxide`; add `From<OxideError>` for the family error if
   it lacks one; a crate other than kornia-imgproc also enables
   `kornia-oxide-kernels/ptx` in its `cuda` feature) and **delete the NVRTC source** for that op. Validation and
   launch geometry stay; no new public API, no `_nvrtc` twin, no fallback.
3. **Tests.** A family test named `<name>_bit_exact` on the public launcher,
   with the CPU path as the reference (never a test-written one): every tail
   length of the per-thread vector width, 1xN / Nx1 / odd sizes, at least one HD
   size, a sentinel-filled oversized `dst`, and the degenerate inputs (zero
   size, overflowing size -> `Err`). `orin-test-bench.sh` fails if an entry has
   no passing `<name>*bit_exact` test, and kornia-tensor's `oxide` loader tests
   must pass with real PTX.
4. **Bench.** One `CASES` row per size in
   `crates/kornia-imgproc/benches/bench_cuda_oxide.rs`.
5. **Numbers.** One row and a short section in `docs/benchmark-cuda-oxide.md`,
   including `orin-test-bench.sh --base origin/main` (main's NVRTC kernel vs
   the branch). Merge bar: at 4K on an Orin with clocks locked, no regression
   beyond ~2%.
6. **FMA.** Add the entry to `FMA_ALLOWED` only if its CPU twin uses an
   explicit `mul_add`/`fmaf`.
7. **Never commit PTX** (`*.ptx`/`*.cubin`); CI fails if one is tracked.

The old spike layout (root-level `oxide-kernels/`, committed `.ptx`, a
`OnceLock` per module) does not apply: one crate, one module, generated entry
names.

### End state

When the last NVRTC family is ported: delete `CudaKernel::compile*`,
`KernelCell`/`KernelSuiteCell` and `CUDA_COLOR_COMMON`, and drop cudarc's
`nvrtc` feature.

## Benchmark protocol

`benches/bench_cuda_oxide.rs` (`cargo bench -p kornia-imgproc --bench
bench_cuda_oxide --features cuda`, or `orin-test-bench.sh`; `harness = false`,
like the other CUDA benches): owned stream, GPU
output checked against the CPU output, warmup, then 5 cycles alternating arm
order; the GPU arm is 5 rounds x 100 launches between CUDA events (kernel time,
no transfers), the CPU arm the same count on the wall clock; each cycle keeps
its fastest round. Reported: min, median and spread of the cycle minima, GB/s,
and the GPU clock seen after each round. `--base <ref>` builds the same bench
at `<ref>` in a separate worktree and target dir and alternates the binaries.
Record power mode, CUDA/driver versions and whether clocks were locked
(`sudo jetson_clocks`); with unlocked clocks say so.

## PR checklist

- [ ] `scripts/cuda-oxide/orin-test-bench.sh --base origin/main` passes; paste the test lines and tables.
- [ ] Row + section in `docs/benchmark-cuda-oxide.md`.
- [ ] `cargo clippy --all-targets -p kornia-oxide-kernels --features ptx -- -D warnings`, the same for `-p kornia-imgproc --features cuda`, `cargo fmt --all --check`.
- [ ] CI `Check - CUDA features` green (it builds and gates the PTX).
