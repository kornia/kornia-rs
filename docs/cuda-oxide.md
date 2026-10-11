# CUDA kernels in Rust with cuda-oxide

kornia-imgproc can ship CUDA kernels written in Rust and compiled to PTX by
NVlabs' [cuda-oxide](https://github.com/NVlabs/cuda-oxide). kornia stays on
stable Rust: the kernels live in a standalone nightly crate, and only their
committed PTX enters the build, through `include_str!` and cudarc. Tracked in
[#1135](https://github.com/kornia/kornia-rs/issues/1135) (Phase 2).

| Path | Role |
|---|---|
| `kernels-oxide/` | Kernel source. Own `[workspace]`, pinned nightly, never built by the main workspace. |
| `crates/kornia-imgproc/src/cuda/oxide/kernels.ptx` | The one committed PTX copy. Generated, never hand-edited. |
| `crates/kornia-imgproc/src/cuda/oxide/mod.rs` | Module loader, oxide-only launchers, PTX gate tests. |
| `scripts/cuda-oxide/build-ptx.sh` | Regenerate the PTX and gate it, or `--check` for drift. |
| `scripts/cuda-oxide/orin-test-bench.sh` | Parity tests + interleaved A/B bench on a Jetson. |
| `docs/benchmark-cuda-oxide.md` | Parity report: every oxide kernel vs its NVRTC twin. |

## How it runs

- Feature `cuda-oxide` (off by default, implies `cuda`). Without it nothing
  changes.
- With it, a kernel's existing public launcher (e.g.
  `cuda::color::gray::launch_gray_from_rgb_u8`) runs the oxide kernel with the
  same signature and launch geometry. No public API changes.
- The PTX targets **`sm_75`** (PTX ISA 6.3), so the driver JITs it on any
  Turing-or-newer GPU; CUDA 13 drops pre-Turing targets anyway. The kernel body
  is identical to an `sm_87` build, only the header differs.
- **Fallback:** if the module fails to load (pre-Turing GPU, a driver that
  rejects the PTX), the launcher uses its NVRTC kernel. The failure is cached;
  `cuda::oxide::ensure_loaded(&stream)` returns it, so a caller (or test) can
  tell which path is live.
- For A/B work, `cuda::oxide::launch_<op>` forces oxide and
  `launch_<op>_nvrtc` forces NVRTC (both only exist under the feature).

## Toolchain (Jetson Orin, JetPack 6 / CUDA 12.6)

Verified on Orin Nano Super, L4T R36.4.3, CUDA 12.6.68, driver 540.4.0.

1. CUDA on `PATH`: `export PATH=/usr/local/cuda/bin:$PATH` (`cargo oxide
   doctor` fails on nvcc otherwise).
2. LLVM 21+: `llc-21` (or `llc-22`; override with `CUDA_OXIDE_LLC`).
3. Nightly: `rustup toolchain install nightly-2026-04-03 -c rust-src,rustc-dev`
   (pinned in `kernels-oxide/rust-toolchain.toml`).
4. cuda-oxide **at the rev pinned in `kernels-oxide/Cargo.toml`**
   (`f819f23fe035ec1c24b8ec06e738b62a9f377140`): build `cargo-oxide` and the
   codegen backend from that checkout so that
   `~/.cargo/cuda-oxide/librustc_codegen_cuda.so` and
   `~/.cargo/cuda-oxide/src` (a git checkout at the pinned rev) exist.
   `build-ptx.sh` refuses to run if the checkout's `HEAD` differs from the pin.
   `CUDA_OXIDE_HOME` overrides the location.

cuda-oxide `main` requires CUDA 13 / driver R580+, which JetPack 6 does not
have; this snapshot generates PTX fine on CUDA 12.6. Bump the pin only together
with a regenerated PTX and a re-run of the parity report. Note that `main`
enables FP contraction by default (`--no-fmad` switches it off); the pinned
snapshot does not contract, which is the contract kornia needs (see below).

## Workflow

```bash
scripts/cuda-oxide/build-ptx.sh            # rebuild PTX from Rust, copy it in, run the PTX gate
scripts/cuda-oxide/build-ptx.sh --check    # rebuild and diff against the committed PTX (drift check)
scripts/cuda-oxide/orin-test-bench.sh      # parity tests (names checked) + A/B bench, on the Orin
```

All three cap cargo at `CARGO_BUILD_JOBS=2` by default (Orin Nano RAM). The
build is deterministic: rebuilding unchanged source leaves the tree clean.
CI cannot run `build-ptx.sh` (no backend on GitHub runners); it only compiles
and lints the `cuda-oxide` feature, and runs nothing on a GPU. Run `--check`
before pushing any change under `kernels-oxide/`.

## Porting checklist for the next kernel

1. Add `src/<family>_<op>.rs` to `kernels-oxide/` with a `#[kernel] pub unsafe
   fn <same name as the NVRTC entry>` taking **the same parameters in the same
   order**, so `launch_map` / the existing launch builder drives either.
2. Transcribe the CUDA C expression for expression. Mirror constants with a
   comment naming the `#define` / CPU constant they copy.
3. Use **raw pointers**, not slices. `&[T]` / `DisjointSlice` lost address-space
   inference (generic `ld.b32`), added a bounds branch per access, and cost
   12-13% on resize upscale.
4. **Load everything before the first store.** Raw pointers carry no
   `noalias`, so the compiler will not hoist a later load above an earlier store;
   interleaved channel stores cost 2x on resize upscale (0.384 vs 0.197 ms).
   CUDA C gets this from `__restrict__`.
5. Do not chase `__ldg`: this snapshot cannot express it, and it measured as a
   no-op on sm_87 (identical resize timings with and without).
6. FP contraction: Rust never contracts implicitly, so the PTX has no `fma`
   without any flag. That *is* kornia's NVRTC `fmad=false` contract. Explicit
   `f32::mul_add` would emit `fma` and trip the gate; change the gate only for a
   kernel that mirrors an explicit `fmaf` on the CPU side.
7. Backend gaps in this snapshot (value-preserving rewrites): integer
   `Ord::min` -> `if a < b`; `f32::min`/`max` -> explicit compares;
   `to_int_unchecked` -> `as` (adds a saturating select).
8. Append the entry name to `ENTRIES` in `cuda/oxide/mod.rs`, add an index
   constant, and dispatch in the existing launcher:
   `if let Ok(k) = oxide::kernel(stream, IDX) { return launch(k, ..) }` before
   the NVRTC path. Expose `launch_<op>_nvrtc` under the feature.
9. `scripts/cuda-oxide/build-ptx.sh`, then read the PTX diff.

## Tests

- **PTX gate** (`cuda::oxide::tests::ptx_gate_*`, no GPU needed): `.target
  sm_75`; every `ENTRIES` name present; zero `fma.`; every `ld`/`st` has a state
  space (no generic memory ops). `build-ptx.sh` runs it and checks that at least
  four gate tests actually executed.
- **Byte-exact parity**, one per kernel, pattern
  `<op>_oxide_bit_exact`: call `ensure_loaded` first (otherwise the test silently
  compares NVRTC against itself), then for sizes covering every tail length
  (`n mod 4` for 4 px/thread kernels) plus a 1080p-sized case, assert
  `oxide == CPU`, `NVRTC == CPU` and `default launcher == oxide` with
  `assert_eq!`. Pre-fill an oversized `dst` with a sentinel and assert the bytes
  past `npixels` survive.
- cargo skips targets with unmet `required-features` without a word. Check the
  test **names** in the output, as `orin-test-bench.sh` does.

## Benchmark protocol

`examples/bench_cuda_oxide.rs` (run by `orin-test-bench.sh`):

- owned stream (the legacy default stream serializes and inflates timings);
- arms verified equal before timing;
- warmup, then 5 cycles alternating which arm goes first; each arm runs 5 rounds
  of 100 back-to-back launches bracketed by CUDA events and synced per round;
  each cycle keeps its fastest round;
- report the min over cycles, the median, the spread of the cycle minima, GB/s
  at the min, and the GPU clock range observed after each round;
- small (320x240), 1080p and 4K.

Record power mode, whether clocks were locked (`sudo jetson_clocks`), CUDA and
driver versions. With unlocked clocks the interleaving and min-of-N are what
make the comparison fair; say so in the report.

## PR checklist

- [ ] `scripts/cuda-oxide/build-ptx.sh --check` passes (committed PTX matches source).
- [ ] `scripts/cuda-oxide/orin-test-bench.sh` passes; paste the test names and the bench table.
- [ ] New section in `docs/benchmark-cuda-oxide.md` with system table and A/B numbers.
- [ ] `cargo clippy --tests -p kornia-imgproc --features cuda,cuda-oxide -- -D warnings` and `cargo fmt --all --check`.
- [ ] Only `kernels.ptx` is committed; no `.ll` or second PTX copy.
