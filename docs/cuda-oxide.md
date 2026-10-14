# CUDA kernels in Rust (cuda-oxide)

kornia-imgproc's `cuda` feature compiles the CUDA kernels in
`crates/kornia-imgproc/kernels-oxide/` from Rust to PTX at build time with
NVlabs' [cuda-oxide](https://github.com/NVlabs/cuda-oxide). No PTX is committed.
kornia itself stays on stable Rust: the kernels live in a standalone crate with
its own pinned nightly. Kernels not yet ported stay NVRTC-compiled at runtime
and keep working under the same feature. Roadmap:
[#1135](https://github.com/kornia/kornia-rs/issues/1135).

| Path | Role |
|---|---|
| `kernels-oxide/src/<family>.rs` | Kernel source, one module per family (`color`, `resize`, ...). Own `[workspace]`, pinned nightly, manifest shipped as `Cargo.toml.in` so `cargo package` keeps the directory. |
| `build.rs` | Scans `kernels-oxide/src` for `#[kernel]` items, builds the PTX, gates it, writes `$OUT_DIR/kernels.ptx` + `$OUT_DIR/oxide_entries.rs`. |
| `src/cuda/oxide/mod.rs` | Loader only (crate-private): one module, `kernel(stream, entry::<NAME>)`. |
| `src/cuda/<family>/` | The family's existing public launchers; only their kernel fetch changes. |
| `examples/bench_cuda_oxide.rs` | Every oxide kernel vs its CPU path; one `CASES` row per op and size. |
| `scripts/cuda-oxide/install-toolchain.sh` | Installs the toolchain below (local, CI and wheels share it). |
| `scripts/cuda-oxide/orin-test-bench.sh` | Jetson: parity tests (names derived from the PTX) + bench; `--base <ref>` A/B. |

## How it builds

- With `cuda` on, `build.rs`:
  1. scans `kernels-oxide/src/**/*.rs` for `#[kernel] pub unsafe fn <name>`
     (sorted; a duplicate name is an error) and writes `oxide_entries.rs`:
     `ENTRIES` plus `entry::<NAME>` index constants;
  2. copies the crate to `$OUT_DIR/kernels-oxide-src` (the source may be a
     read-only registry checkout) and runs `cargo build --release --locked`
     there under the pinned nightly and the codegen backend, in its own target
     dir, with the outer cargo's environment scrubbed (`RUSTC*`, `RUSTFLAGS`,
     the jobserver, every `CARGO_*` except `CARGO_HOME`, `CARGO_BUILD_JOBS` and
     network settings), jobs from `CARGO_BUILD_JOBS` or the outer `NUM_JOBS`;
  3. gates the PTX (below) and writes `kernels.ptx`.
- It reruns when `kernels-oxide/{src,Cargo.toml.in,Cargo.lock,rust-toolchain.toml}`,
  the backend `.so`, or one of `CUDA_OXIDE_LLC`, `CUDA_OXIDE_BACKEND`,
  `CARGO_HOME`, `HOME`, `CUDA_HOME`, `CUDA_TOOLKIT_PATH`, `DOCS_RS`,
  `KORNIA_OXIDE_STUB` changes. Cold nested build on an Orin at `-j2`: ~190 s;
  after a kernel edit: ~4 s.
- **Stub:** under docs.rs (`DOCS_RS`) or with `KORNIA_OXIDE_STUB=1` the PTX is
  empty (a cargo warning says so); launchers compile, and fetching a kernel
  returns an error at runtime.
- The PTX targets `sm_75` (PTX ISA 6.3): the driver JITs it for any
  Turing-or-newer GPU. Maxwell/Pascal (Jetson Nano/TX2) cannot run ported ops.
- A missing toolchain fails the build with one message naming the missing
  piece; it never falls back to an empty PTX.
- Nothing in a plain workspace build enables `cuda`: the CUDA examples
  (`cuda-fusion`, `cuda-camera-sift`, `cuda-camera-preprocess`) have their own
  `cuda` feature and build only with `--features cuda`.

### PTX gate (`gate()` in `build.rs`)

- module: `.target sm_75`; no `.extern` symbols; the `.entry` set equals the
  scanned `#[kernel]` set;
- per entry: every `ld`/`st` names a state space (a generic access means
  address-space inference was lost); no `fma`/float `mad` unless the entry is
  in `FMA_ALLOWED` (empty); no libdevice (`__nv_*`) calls.

Errors read `gate: <entry>: <reason>`.

## Toolchain

`scripts/cuda-oxide/install-toolchain.sh [PREFIX]` installs everything into
`PREFIX` (default `~/.cargo/cuda-oxide`) and writes `PREFIX/env.sh`; source it
before cargo. Pieces already in `PREFIX` at the right version are reused, so CI
caches `PREFIX`. What it installs:

1. the nightly pinned in `kernels-oxide/rust-toolchain.toml`
   (`nightly-2026-04-03`, with `rust-src`, `rustc-dev`);
2. `cuda.h` and the libcuda link stub (NVIDIA's `cuda_cudart` redist archive),
   unless `$CUDA_HOME/include/cuda.h` exists. No GPU or driver is needed;
3. LLVM 21 `llc` + libclang (for bindgen of `cuda.h`) from conda-forge, unless
   `llc-21`/`llc-22` is on `PATH`. conda-forge builds need glibc 2.17; LLVM's
   own release tarballs need 2.34 and do not run in manylinux containers;
4. the codegen backend `librustc_codegen_cuda.so`, built at the rev pinned in
   `kernels-oxide/Cargo.toml.in` (`f819f23`). `build.rs` refuses a backend
   whose `src/` checkout is at another rev.

`OXIDE_USE_SYSTEM=0` ignores a system `cuda.h`/`llc` (CI uses it so every job
runs the same recipe). cuda-oxide `main` needs CUDA 13 / R580+, which JetPack 6
lacks; the pinned snapshot works on CUDA 12.6 and never contracts FP. Bump the
pin only together with a re-run of the parity tests and the bench.

## Adding a kernel / migrating an NVRTC family

A second kernel PR (e.g. bicubic resize) does exactly this; nothing else:

1. **Kernel.** Add `#[kernel] pub unsafe fn <name>` to
   `kernels-oxide/src/<family>.rs` (new family: add the file and `mod <family>;`
   to `main.rs`). `<name>` is the NVRTC entry it replaces, exactly; same
   parameters in the same order, so the existing launch code drives it.
   - raw pointers, not slices (slices lose the address space and add a bounds
     branch: 12-13% on resize upscale);
   - **all loads before the first store**: raw pointers carry no `noalias`, so
     interleaved loads/stores serialize (2x on resize upscale);
   - mirror each constant with a comment naming the CPU constant it copies;
   - backend gaps: no `Ord::min` (use `if`), `f32::min`/`max` become compares,
     no `__ldg` (a no-op on sm_87 anyway).
2. **Launcher.** In the family's public launcher replace the NVRTC kernel fetch
   with `oxide::kernel(stream, oxide::entry::<NAME>)?` (add
   `From<OxideError>` for the family error if it lacks one) and **delete the NVRTC source** for that op. Validation and
   launch geometry stay; no new public API, no `_nvrtc` twin, no fallback.
3. **Tests.** A family test named `<name>_bit_exact` on the public launcher,
   with the CPU path as the reference (never a test-written one): every tail
   length of the per-thread vector width, 1xN / Nx1 / odd sizes, at least one HD
   size, a sentinel-filled oversized `dst`, and the degenerate inputs (zero
   size, overflowing size -> `Err`). `orin-test-bench.sh` fails if an entry has
   no passing `<name>*bit_exact` test, and `all_entries_load` must pass.
4. **Bench.** One `CASES` row per size in `examples/bench_cuda_oxide.rs`.
5. **Numbers.** One row and a short section in `docs/benchmark-cuda-oxide.md`,
   including `orin-test-bench.sh --base origin/main` (main's NVRTC kernel vs
   the branch). Merge bar: at 4K on an Orin with clocks locked, no regression
   beyond ~2%.
6. **FMA.** Add the entry to `FMA_ALLOWED` only if its CPU twin uses an
   explicit `mul_add`/`fmaf`.
7. **Never commit PTX** (`*.ptx`/`*.cubin`); CI fails if one is tracked.

The old spike layout (root-level `oxide-kernels/`, committed `.ptx`, a
`OnceLock` per module) does not apply: one crate, one module, generated entry
indices.

### End state

When the last NVRTC family is ported: delete `CudaKernel::compile*`,
`KernelCell`/`KernelSuiteCell` and `CUDA_COLOR_COMMON`, and drop cudarc's
`nvrtc` feature.

## Benchmark protocol

`examples/bench_cuda_oxide.rs` (run by `orin-test-bench.sh`): owned stream, GPU
output checked against the CPU output, warmup, then 5 cycles alternating arm
order; the GPU arm is 5 rounds x 100 launches between CUDA events (kernel time,
no transfers), the CPU arm the same count on the wall clock; each cycle keeps
its fastest round. Reported: min, median and spread of the cycle minima, GB/s,
and the GPU clock seen after each round. `--base <ref>` builds the same example
at `<ref>` in a separate worktree and target dir and alternates the binaries.
Record power mode, CUDA/driver versions and whether clocks were locked
(`sudo jetson_clocks`); with unlocked clocks say so.

## PR checklist

- [ ] `scripts/cuda-oxide/orin-test-bench.sh --base origin/main` passes; paste the test lines and tables.
- [ ] Row + section in `docs/benchmark-cuda-oxide.md`.
- [ ] `cargo clippy --all-targets -p kornia-imgproc --features cuda -- -D warnings`, `cargo fmt --all --check`.
- [ ] CI `Check - CUDA features` green (it builds and gates the PTX, and builds the docs.rs stub).
