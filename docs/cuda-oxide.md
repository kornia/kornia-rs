# CUDA kernels in Rust with cuda-oxide

kornia-imgproc can ship CUDA kernels written in Rust and compiled to PTX by
NVlabs' [cuda-oxide](https://github.com/NVlabs/cuda-oxide). kornia stays on
stable Rust: the kernels live in a standalone nightly crate, and kornia-imgproc's
`build.rs` compiles them to PTX at build time when the `cuda-oxide` feature is
on. No PTX is committed. Tracked in
[#1135](https://github.com/kornia/kornia-rs/issues/1135) (Phase 2).

| Path | Role |
|---|---|
| `crates/kornia-imgproc/kernels-oxide/` | Kernel source. Manifest is `Cargo.toml.in` (so `cargo package` ships the directory), own `[workspace]`, pinned nightly; never built by the main workspace. |
| `crates/kornia-imgproc/build.rs` | With `cuda-oxide`: stages `kernels-oxide/` in `OUT_DIR`, builds it to `$OUT_DIR/kernels.ptx` and gates it. Otherwise a no-op. |
| `crates/kornia-imgproc/src/cuda/oxide/mod.rs` | Module loader (`include_str!` from `OUT_DIR`), oxide-only launchers. |
| `scripts/cuda-oxide/orin-test-bench.sh` | Parity tests (names checked) + interleaved A/B bench on a Jetson. |
| `docs/benchmark-cuda-oxide.md` | Parity report: every oxide kernel vs its NVRTC twin. |

## How it runs

- Feature `cuda-oxide` (off by default, implies `cuda`). Without it the build
  script returns at once: default builds and the wheels are unchanged.
- With it, `build.rs` copies `kernels-oxide/` to `$OUT_DIR/kernels-oxide-src`
  (renaming `Cargo.toml.in` to `Cargo.toml`; only changed files are rewritten),
  so the source dir is only ever read, as in a read-only `~/.cargo/registry`
  checkout. It then runs `cargo build --release --locked` there under the
  pinned nightly with the cuda-oxide codegen backend:
  - in its own target dir, `$OUT_DIR/kernels-oxide-target`, so it never waits on
    the outer build's lock;
  - with the outer cargo's environment scrubbed: `RUSTC*`, `RUSTFLAGS`,
    `RUSTUP_TOOLCHAIN`, the jobserver, and every `CARGO_*` except
    `CARGO_HOME`, `CARGO_BUILD_JOBS` and the network settings;
  - with jobs set by `CARGO_BUILD_JOBS`, or else the outer build's `NUM_JOBS`.
- It drives cargo directly with the flags `cargo oxide build` would pass.
  `cargo oxide` itself rewrites `src/main.rs` on every run, which would re-trigger
  the build script forever, and it writes output into the source tree.
- It reruns only when one of these changes:
  - `kernels-oxide/{src,Cargo.toml.in,Cargo.lock,rust-toolchain.toml}`;
  - `CUDA_OXIDE_LLC`, `CUDA_OXIDE_BACKEND`, `CARGO_HOME`, `CUDA_HOME` or
    `CUDA_TOOLKIT_PATH`.
- The PTX targets **`sm_75`** (PTX ISA 6.3), so the driver JITs it on any
  Turing-or-newer GPU; CUDA 13 drops pre-Turing targets anyway. The kernel body
  is identical to an `sm_87` build; only the header differs.
- A kernel's existing public launcher (e.g.
  `cuda::color::gray::launch_gray_from_rgb_u8`) runs the oxide kernel with the
  same signature and launch geometry. No public API changes.
- **Runtime fallback:** if the module fails to load (a pre-Turing GPU, or a
  driver that rejects the PTX), the launcher uses its NVRTC kernel. The failure
  is cached, and `cuda::oxide::ensure_loaded(&stream)` returns it.
- For A/B work, `cuda::oxide::launch_<op>` forces oxide and
  `launch_<op>_nvrtc` forces NVRTC. Both exist only under the feature.
- `kernels-oxide/` ships inside the published `kornia-imgproc` package, so the
  feature also builds from crates.io (given the toolchain below).

## Toolchain (what `build.rs` needs)

If anything is missing, the build fails with one message naming it. It never
falls back to an empty PTX. The same setup runs in CI: job `cuda-feature-check`
in `.github/workflows/rust_lint.yml`.

1. `rustup` with the nightly pinned in `kernels-oxide/rust-toolchain.toml`
   (paths here are relative to `crates/kornia-imgproc/`):
   `rustup toolchain install nightly-2026-04-03 -c rust-src,rustc-dev`.
2. LLVM 21+: `llc-21` or `llc-22` on `PATH`, or set `CUDA_OXIDE_LLC`.
3. `cuda.h` and the libcuda stub under `$CUDA_HOME` (default `/usr/local/cuda`).
   Only headers are needed, no GPU or driver: on Ubuntu,
   `cuda-cudart-dev-12-6` (cuda.h) plus `cuda-driver-dev-12-6` (the stub) is
   enough. Point `CUDA_HOME` at the directory holding `include/cuda.h`; put the
   stub's directory on `LIBRARY_PATH` if it is not under `$CUDA_HOME`.
4. The codegen backend, built **at the rev pinned in `kernels-oxide/Cargo.toml.in`**
   (`f819f23fe035ec1c24b8ec06e738b62a9f377140`), at
   `$CARGO_HOME/cuda-oxide/librustc_codegen_cuda.so` or `$CUDA_OXIDE_BACKEND`:
   ```bash
   git clone https://github.com/NVlabs/cuda-oxide ~/.cargo/cuda-oxide/src
   git -C ~/.cargo/cuda-oxide/src checkout f819f23fe035ec1c24b8ec06e738b62a9f377140
   cd ~/.cargo/cuda-oxide/src/crates/rustc-codegen-cuda
   SYSROOT=$(rustup run nightly-2026-04-03 rustc --print sysroot)
   LIBRARY_PATH=$SYSROOT/lib LD_LIBRARY_PATH=$SYSROOT/lib rustup run nightly-2026-04-03 cargo build
   cp target/debug/librustc_codegen_cuda.so ~/.cargo/cuda-oxide/
   ```
   If `~/.cargo/cuda-oxide/src` is a git checkout, `build.rs` refuses to build
   when its `HEAD` differs from the pin. `cargo-oxide` itself is optional:
   `cargo oxide doctor` is a handy check, but the build never invokes it.

cuda-oxide `main` requires CUDA 13 / driver R580+, which JetPack 6 does not
have; the pinned snapshot generates PTX fine on CUDA 12.6. Bump the pin only
together with a re-run of the parity report. `main` also enables FP contraction
by default. The pinned snapshot does not contract, which is the contract kornia
needs (see below).

## Workflow

```bash
cargo test --release -p kornia-imgproc --features cuda,cuda-oxide --lib -- oxide gray_from_rgb_u8
scripts/cuda-oxide/orin-test-bench.sh      # the same tests (names checked) + A/B bench, on the Orin
```

Editing a file under `kernels-oxide/src` rebuilds the PTX on the next cargo
invocation. Nested-build cost, measured on the Orin at `-j2`:

| Case | Nested build |
|---|---|
| cold (clean `OUT_DIR`; compiles cuda-oxide's host crates) | about 190 s |
| after a kernel edit (kernel crate only) | about 4 s |
| nothing changed | not invoked |

After any build-script rerun, cargo also recompiles kornia-imgproc itself. On
small boards keep `CARGO_BUILD_JOBS=2`, because the nested build runs alongside
the outer one.

## Porting checklist for the next kernel

1. Add `src/<family>_<op>.rs` to `kernels-oxide/` with a `#[kernel] pub unsafe
   fn <same name as the NVRTC entry>`. It must take **the same parameters in the
   same order**, so `launch_map` or the existing launch builder can drive either
   kernel.
2. Transcribe the CUDA C expression for expression. Mirror each constant, with a
   comment naming the `#define` or CPU constant it copies.
3. Use **raw pointers**, not slices. On resize upscale, `&[T]` and
   `DisjointSlice`:
   - lost address-space inference (generic `ld.b32`);
   - added a bounds branch per access;
   - cost 12-13%.
4. **Load everything before the first store.** Raw pointers carry no `noalias`,
   so the compiler will not move a later load above an earlier store. On resize
   upscale, interleaved channel stores cost 2x (0.384 vs 0.197 ms). CUDA C gets
   this ordering for free from `__restrict__`.
5. Do not chase `__ldg`. This snapshot cannot express it, and it measured as a
   no-op on sm_87: resize timings were identical with and without it.
6. FP contraction: Rust never contracts implicitly, so the PTX has no `fma`
   without any flag. That *is* kornia's NVRTC `fmad=false` contract.
   - Explicit `f32::mul_add` would emit `fma` and trip the gate.
   - Change the gate only for a kernel that mirrors an explicit `fmaf` on the
     CPU side.
7. Backend gaps in this snapshot, each worked around by a value-preserving
   rewrite:
   - integer `Ord::min` becomes `if a < b`;
   - `f32::min`/`max` become explicit compares;
   - `to_int_unchecked` becomes `as`, which adds a saturating select.

   `for i in a..b` works.
8. Wire the kernel into kornia-imgproc:
   - append the entry name to `ENTRIES` in `cuda/oxide/mod.rs` and add an index
     constant;
   - dispatch in the existing launcher before the NVRTC path:
     `if let Ok(k) = oxide::kernel(stream, IDX) { return launch(k, ..) }`;
   - expose `launch_<op>_nvrtc` under the feature.
9. Build, then read the PTX at `$OUT_DIR/kernels.ptx`
   (`target/<profile>/build/kornia-imgproc-*/out/`).

## Tests

- **PTX gate, at build time** (`gate()` in `build.rs`). It requires:
  - `.target sm_75`;
  - at least one `.entry`;
  - zero `fma.`;
  - a state space on every `ld`/`st` (no generic memory ops).

  A failing gate fails the build, so every build with the feature enforces it,
  including CI's clippy.
- **Byte-exact parity**, one test per kernel, named `<op>_oxide_bit_exact`:
  - Call `ensure_loaded` first. Otherwise the test silently compares NVRTC
    against itself, and a missing entry would go unnoticed.
  - Use sizes that cover every tail length (`n mod 4` for 4 px/thread kernels),
    plus a 1080p-sized case.
  - Assert `oxide == CPU`, `NVRTC == CPU` and `default launcher == oxide` with
    `assert_eq!`.
  - Pre-fill an oversized `dst` with a sentinel, and assert the bytes past
    `npixels` survive.
- cargo skips targets with unmet `required-features` without a word. Check the
  test **names** in the output, as `orin-test-bench.sh` does.

## Benchmark protocol

`examples/bench_cuda_oxide.rs` (run by `orin-test-bench.sh`):

- owned stream (the legacy default stream serializes and inflates timings);
- arms verified equal before timing;
- warmup, then 5 cycles alternating which arm goes first;
- each arm runs 5 rounds of 100 back-to-back launches, bracketed by CUDA events
  and synced per round, and each cycle keeps its fastest round;
- report the min over cycles, the median, the spread of the cycle minima, GB/s
  at the min, and the GPU clock range observed after each round;
- sizes: small (320x240), 1080p and 4K.

Record the power mode, whether clocks were locked (`sudo jetson_clocks`), and
the CUDA and driver versions. With unlocked clocks, the interleaving and min-of-N
are what keep the comparison fair; say so in the report.

## PR checklist

- [ ] `scripts/cuda-oxide/orin-test-bench.sh` passes; paste the test names and the bench table.
- [ ] New section in `docs/benchmark-cuda-oxide.md` with system table and A/B numbers.
- [ ] `cargo clippy --all-targets -p kornia-imgproc --features cuda,cuda-oxide -- -D warnings` and `cargo fmt --all --check`.
- [ ] CI's `Check - CUDA features` job is green (it builds and gates the PTX).
