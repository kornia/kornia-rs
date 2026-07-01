# SIMD development guide for `kornia-imgproc`

How to add, test, profile, and benchmark SIMD kernels across aarch64 (NEON) and x86_64 (AVX2/AVX-512). Written so an agent (human or AI) picking this up cold can be productive in one session.

---

## Architecture

### Runtime CPU feature probe — `src/simd/mod.rs`

Single source of truth for "what SIMD is available right now." Probed once per process into a `&'static CpuFeatures` via `OnceLock`. All kernels read `crate::simd::cpu_features()` instead of sprinkling `is_x86_feature_detected!` calls — keeps dispatch decisions consistent and avoids per-call atomic loads.

### Dispatch pattern

Every op that ships a SIMD path follows this structure:

```rust
pub fn my_op(src: &[u8], dst: &mut [f32], ...) {
    #[cfg(target_arch = "aarch64")]
    {
        // NEON is architectural on aarch64 — no runtime check needed.
        // SAFETY: length preconditions documented below.
        unsafe { my_op_neon(src, dst, ...) };
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        let cpu = crate::simd::cpu_features();
        if cpu.has_avx2 && cpu.has_fma {
            unsafe { my_op_avx2(src, dst, ...) };
            return;
        }
    }
    #[allow(unreachable_code)]
    my_op_scalar(src, dst, ...);
}
```

The `cfg(target_arch)` gates let the compiler dead-code-eliminate branches per build target. Runtime `cpu.has_*` checks only discriminate feature levels **within** an ISA (AVX2 vs AVX-512, NEON vs SVE).

### Where kernels live

- **Single-op kernels** (e.g. `normalize_rgb_u8`): co-located in the op's module (`normalize.rs`). Public entry + `fn _scalar` + `unsafe fn _neon` / `unsafe fn _avx2`.
- **Shared row-level kernels** (used by multiple algos, e.g. `pyrdown_row_rgb_u8`, `horizontal_row_rgb_u8`): `src/resize/kernels.rs` or `src/warp/kernels.rs`. Dispatcher is `#[inline(always)]` so call-site `cfg` branches collapse.
- **Multi-dtype ops** (same semantic op with different SIMD kernels per pixel type, e.g. RGB→gray for `u8`/`f32`/`f64`): promote the module to a **subdir** — `<op>/mod.rs` + `<op>/kernels.rs` — and wire dtype dispatch through a sealed trait. See `color/gray/` as the canonical example.

#### Submodule layout for multi-dtype SIMD ops

When an op has meaningfully different SIMD kernels per dtype (not just a cast wrapper), split the module into a directory that mirrors `filter/`, `warp/`, `resize/`, and `morphology/`:

```
color/gray/
  mod.rs      — public Image-typed API + sealed trait impls + thin compat wrappers
  kernels.rs  — constants, par_strip_dispatch, slice-level dispatchers,
                NEON/AVX2/scalar kernels (all private except pub slice fns)
```

`color/mod.rs` keeps `mod gray;` unchanged — Rust resolves it to `gray/mod.rs` automatically.

#### Dtype dispatch: sealed trait

Use a **sealed trait** to route a single `my_op<T>` entry point to the right kernel at compile time with zero runtime cost:

```rust
// mod.rs
mod sealed { pub trait Sealed {} }

pub trait GrayFromRgb: sealed::Sealed + Sized {
    #[doc(hidden)]
    fn gray_from_rgb_impl(
        src: &Image<Self, 3>,
        dst: &mut Image<Self, 1>,
    ) -> Result<(), ImageError>;
}

impl sealed::Sealed for u8 {}
impl GrayFromRgb for u8 {
    fn gray_from_rgb_impl(src: &Image<u8, 3>, dst: &mut Image<u8, 1>)
        -> Result<(), ImageError>
    {
        check_size(src, dst)?;
        kernels::rgb_to_gray_u8(src.as_slice(), dst.as_slice_mut(), src.rows() * src.cols());
        Ok(())
    }
}
// impl for f32 → kernels::rgb_to_gray_f32 (NEON/AVX2+FMA)
// impl for f64 → portable scalar

pub fn gray_from_rgb<T: GrayFromRgb>(
    src: &Image<T, 3>,
    dst: &mut Image<T, 1>,
) -> Result<(), ImageError> {
    T::gray_from_rgb_impl(src, dst)
}
```

Rules:
- `mod sealed { pub trait Sealed {} }` prevents external impls — callers can use the trait as a bound but cannot add new implementations.
- Name the hidden method `<fn>_impl` and mark it `#[doc(hidden)]`.
- Explicitly list each supported type (`u8`, `f32`, `f64`). Do **not** use a blanket `impl<T: Float>` — it conflicts with concrete `impl for f32` (no stable specialization in Rust).
- Keep backward-compat wrappers (`gray_from_rgb_u8`, `gray_from_rgb_f32`) as thin one-liners that call `gray_from_rgb`. Don't duplicate the size check or the kernel call.

#### Shared size-check helper

Extract the repeated `src.size() != dst.size()` boilerplate into a private `check_size` that works across const-generic channel counts:

```rust
#[inline]
fn check_size<T, U, const C1: usize, const C2: usize>(
    src: &Image<T, C1>,
    dst: &Image<U, C2>,
) -> Result<(), ImageError> {
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(), src.rows(), dst.cols(), dst.rows(),
        ));
    }
    Ok(())
}
```

Call it at the top of every `*_impl` body and in `rgb_from_gray`. One definition eliminates copy-paste across every dtype impl.

### Naming convention

Public functions follow `<output_type>_from_<input_type>` — the **output comes first**:

```
gray_from_rgb       ✓       rgb_to_gray         ✗
rgb_from_gray       ✓       gray_to_rgb         ✗
bgr_from_rgb        ✓       rgb_to_bgr          ✗
```

This mirrors Rust's `From`/`Into` trait naming and makes the conversion direction unambiguous at every call site. The rule applies at every level — public Image API, slice-level fns, and internal `_kernel`/`_neon`/`_avx2`/`_scalar` leaves.

### Adding a new SIMD path

1. Write the scalar reference first. It is the correctness oracle.
2. Add the ISA kernel as `unsafe fn` with `#[target_feature(enable = "…")]`.
3. Add a correctness test that compares SIMD output to scalar on a PRNG-seeded batch. Max \|diff\| must be ≤ `1e-4` for f32 (FMA rounding) or bit-exact for integer kernels.
4. Update the dispatcher.
5. If it's a new shared kernel, add to `kernels.rs` and document per-arch feature requirements in the module header.
6. If the op has different kernels per dtype, promote to `<op>/mod.rs` + `<op>/kernels.rs` and add a sealed-trait dispatcher (see pattern above).

---

## Correctness: cross-arch bit-equivalence

**Every SIMD kernel must pass bit-equivalence vs its scalar reference on both aarch64 and x86_64.** The scalar path is authoritative; vectorized paths are fast implementations of the same arithmetic.

### The verify example pattern — `examples/verify_normalize_avx2.rs`

Standalone binary that:
1. Prints the runtime CPU feature probe.
2. Reports which dispatcher path got selected.
3. Runs both the dispatched SIMD path AND an inline scalar reference on identical PRNG-seeded input.
4. Reports max \|diff\| and PASS/FAIL on 1e-4 tolerance.

Built as an **example**, not a test, because `cargo test` pulls in `criterion` as a dev-dep, which transitively requires a C toolchain under cross-compilation. Examples skip dev-deps.

Clone this pattern for each new kernel port. ~80 LOC.

### Expected numerical tolerance

- **f32 FMA kernels** (normalize, warp): max \|diff\| ~= 2.4e-7 due to single-rounding-step FMA vs scalar's split multiply+add.
- **i16/Q14 kernels** (resize bicubic/lanczos): mean \|diff\| ≤ 0.04 and max \|diff\| ≤ 20 vs Pillow (Q14 rounding noise). Bit-exact within the same backend.

---

## Cross-arch testing

**Prefer native hardware.** If you have access to both an aarch64 box and an x86_64 box, run the tests and benchmarks natively on each — no emulation layer, no translation overhead, wall-clock numbers are real. qemu is a *fallback* for developers who only have one of the two, and should not be used to produce numbers that go into PRs, changelogs, or memory entries.

### Native path (recommended)

On each target machine, a plain `cargo` invocation is enough — the runtime dispatcher picks the right kernel automatically:

```bash
# On an x86_64 box with AVX2+FMA
cargo run --release -p kornia-imgproc --example verify_normalize_avx2
cargo run --release -p kornia-imgproc --example bench_normalize_cross_arch

# On an aarch64 box (Jetson, Apple Silicon under Linux, Graviton)
cargo run --release -p kornia-imgproc --example verify_normalize_avx2
cargo run --release -p kornia-imgproc --example bench_normalize_cross_arch
```

The examples print the arch tag and which dispatcher path got selected, so you can confirm the intended kernel is running. Publishable perf numbers come from this path.

**Tip:** if you're porting a kernel and want to compare SIMD vs scalar on the same native target, build once with `-Ctarget-cpu=native` and once with `-Ctarget-cpu=x86-64` (or `-Ctarget-cpu=generic` on aarch64). The scalar-baseline build forces the dispatcher's fallback path.

### Fallback — qemu on an aarch64-only host

Useful when you're developing on a Jetson (or any aarch64 box) and want to verify the AVX2 port *compiles, links, dispatches correctly, and is bit-equivalent to scalar* before pushing to CI or a native x86 machine. **Do not trust qemu wall-clock times** — TCG translation is 5–20× slower than native silicon. Trust only:

- Correctness (bit-equivalence against the scalar reference).
- Which dispatcher path gets selected at runtime.
- SIMD-vs-scalar **ratio** on the same target (both paths translate through the same TCG layer, so the ratio cancels the overhead).

#### One-time qemu setup (aarch64 host only)

Ubuntu jammy ships qemu-user-static 6.2, which predates AVX2 TCG support (needs ≥7.2). **AVX2 code SIGILLs under stock qemu.** Register a newer qemu via `tonistiigi/binfmt`:

```bash
# Prereqs: user in docker group; run once
sudo docker run --privileged --network=none --rm \
    tonistiigi/binfmt:latest --uninstall amd64
sudo docker run --privileged --network=none --rm \
    tonistiigi/binfmt:latest --install amd64
```

`--uninstall` first is important: tonistiigi is a no-op if a handler is already registered, and the default jammy handler points at qemu 6.2. After reinstall, `cat /proc/sys/fs/binfmt_misc/qemu-x86_64 | head -3` shows `interpreter /usr/bin/qemu-x86_64` with flags `POCF` (tonistiigi's embedded qemu 8.x). If you still see `/usr/libexec/qemu-binfmt/x86_64-binfmt-P` with flags `PF`, the stock handler is still active — AVX2 will crash.

`--network=none` works around Jetson's missing iptables `raw` table; harmless on other hosts.

#### Cargo config — `.cargo/config.toml`

```toml
[target.x86_64-unknown-linux-gnu]
runner = ["env", "QEMU_LD_PREFIX=/usr/x86_64-linux-gnu"]
linker = "x86_64-linux-gnu-gcc"
rustflags = ["-Ctarget-cpu=x86-64"]
```

- `runner = ["env", "QEMU_LD_PREFIX=…"]`: binfmt_misc picks up the binary transparently; `QEMU_LD_PREFIX` tells qemu where the amd64 dynamic loader + libc live (the `gcc-x86-64-linux-gnu` package ships the sysroot there).
- `linker = "x86_64-linux-gnu-gcc"`: needed because `rust-lld` doesn't know where the amd64 sysroot is.

- `rustflags = ["-Ctarget-cpu=x86-64"]`: keeps the x86_64 build portable for older CPUs. AVX2/FMA kernels are still compiled with `#[target_feature]` and selected through runtime dispatch when available.

#### Running x86 examples under qemu

```bash
# Verify an AVX2 kernel (correctness + dispatch confirmation only)
cargo run --release -p kornia-imgproc --example verify_normalize_avx2 \
    --target x86_64-unknown-linux-gnu

# SIMD-vs-scalar ratio (ratio is valid; absolute times are not)
cargo run --release -p kornia-imgproc --example bench_normalize_cross_arch \
    --target x86_64-unknown-linux-gnu
```

Unit tests under qemu: `cargo test` pulls in `criterion` as a dev-dep, which transitively needs a C toolchain under cross-compilation. Prefer **examples** for cross-arch verification — they skip dev-deps.

#### When to promote from qemu → native

Your port should hit native hardware **before merging** whenever possible:

1. Code compiles + link-cleanly under cross.
2. `verify_*` example passes bit-equivalence under qemu.
3. Spin up a cloud x86 instance (AWS t3/c6i, GCE e2/c3) *or* borrow a local x86 box.
4. Run the same `verify_*` + `bench_*` examples natively.
5. Paste native numbers (not qemu numbers) into the PR.

If native hardware genuinely isn't available, call that out explicitly in the PR — "validated under qemu only, native perf verification pending" — so reviewers know which claims are emulated.

---

## Benchmarking methodology

### Single-shot benches lie

`timeit`-style one-measurement benches have stdev > 10% on this class of hardware due to:
- rayon thread pool warm-up.
- Turbo clock ramp on the Jetson.
- Cache-cold first iteration.
- Neighbor thread (Ollama, Xorg) jitter.

Use **best-of-3 rounds × 30–50 iterations each, with 5–10 iter warmup**, median over rounds. Example harness (Python, what we use for K vs OpenCV comparisons):

```python
def bench(fn, rounds=4, n=30, warmup=5):
    for _ in range(warmup): fn()
    xs = []
    for _ in range(rounds):
        t = time.perf_counter()
        for _ in range(n): fn()
        xs.append((time.perf_counter() - t) * 1000 / n)
    return sorted(xs)[len(xs) // 2]   # median
```

In Rust (`examples/bench_normalize_cross_arch.rs`), same shape: 3 rounds × 50 iters, take min (wall-clock best approximates compute time best).

### Correctness-before-perf

Always run the correctness check first. A kernel that's 5× faster but produces garbage is negative progress. Keep the scalar reference close to the SIMD path in source — a divergence between them is the first thing to check after a perf regression.

### What to bench against

- **OpenCV (cv2)** — primary reference. `cv2.resize`, `cv2.GaussianBlur`, etc.
- **Pillow** — the Pillow reference is what `resize` correctness tests compare against (pymode `P.BICUBIC`, `P.LANCZOS`).
- **numpy scalar** — for ops with no OpenCV equivalent (normalize with mean/std).

### Publishing a perf claim

Whenever updating memory (project_future_roadmap.md) or a PR description with perf numbers, include:
- Image size + channel count
- Best-of-N harness used
- Reference library + version
- Commit hash (`git rev-parse HEAD`)

---

## Profiling tools

### Rust-level

- `cargo flamegraph --release -p kornia-imgproc --example <name>` — needs `perf` (`sudo apt install linux-tools-common`).
- `perf stat -e task-clock,cycles,instructions,cache-misses,L1-dcache-load-misses ./target/release/examples/<name>` — IPC and cache behavior.

### NEON / AVX2 assembly inspection

```bash
cargo rustc --release -p kornia-imgproc --lib -- --emit=asm
# Output under target/release/deps/kornia_imgproc-<hash>.s
```

Look for unexpected spills (stores into `[sp, #…]` in hot loops on aarch64 — means register pressure is too high and the compiler is spilling accumulators). Fix: reduce the number of in-flight accumulators, or split the kernel into two passes.

### Memory behavior

- `perf stat -e LLC-load-misses,dTLB-load-misses ...` — is the kernel L2-bound?
- `likwid-perfctr` if available — direct hardware-counter access.

For 1080p RGB8, the arithmetic intensity of most preprocess kernels (normalize, resize, blur) is low — they are almost always **memory-bound on the Jetson**. Doubling FMA throughput won't help; reducing bytes-touched-per-output-pixel will. See `resize/fused.rs` for the canonical "fuse passes to halve memory traffic" pattern.

---

## Common gotchas

### Avoid `-Ctarget-cpu=x86-64-v3` for portable builds

Because AVX2+FMA+BMI2 are in the v3 baseline, the runtime feature check may fold to an always-true constant. The dispatcher will always pick AVX2 and **never** fall through to scalar. Public Linux x86_64 wheels should use `-Ctarget-cpu=x86-64`; AVX2/FMA should remain isolated in `#[target_feature]` kernels and selected through runtime dispatch.

To exercise the scalar fallback under qemu, build with `-Ctarget-cpu=x86-64` (plain baseline).

### Criterion under qemu

`criterion` pulls in `alloca` which invokes `cc-rs` which wants `x86_64-linux-gnu-gcc` at build time. That's installed on the Jetson, but if tests start failing at link time with `-m64` errors, the aarch64 gcc got picked as the linker — check `.cargo/config.toml`'s `linker = "x86_64-linux-gnu-gcc"`.

### Why the 4-chain rolling accumulator pattern

In `resize/kernels.rs::vertical_row_neon` and `horizontal_row_c3_neon` you'll see 4 (or more) independent accumulators cycled through by `k % 4`. Reason: `vmlal_s16` has ~4-cycle latency on A-class cores. With one accumulator, each MAC waits on the previous. Four independent chains means four MACs in flight per cycle (matching the 2×issue NEON pipe), fully pipelining. If you rewrite one of these kernels, preserve ≥4 accumulators per SIMD width.

### `ROWS_PER_TASK = 16` coarse chunking for rayon

Many ops in this crate use `par_chunks_mut(ROWS_PER_TASK * row_len)` instead of per-row parallelism. Reason: per-row task spawn overhead is 2–5 µs, which dominates on memory-bound ops at 1080p. 16-row chunks amortize this. When adding a new parallel op, follow the existing pattern — see `flip.rs`, `crop.rs`, `padding.rs`, `core.rs`, `resize/nearest.rs` for examples.

### Benchmark jitter from background load

Kill Ollama (`! sudo systemctl stop ollama`) before serious benching on the Jetson — it pins a CPU core when idle-loading a model. Same for any other heavy background service.

---

## Reference — key files

| File | Role |
|---|---|
| `src/simd/mod.rs` | `CpuFeatures` + `cpu_features()` probe |
| `src/color/gray/mod.rs` | **Canonical multi-dtype sealed-trait dispatch** — `GrayFromRgb` + single `gray_from_rgb<T>` entry |
| `src/color/gray/kernels.rs` | **Canonical `mod.rs`/`kernels.rs` split** — `par_strip_dispatch`, u8 NEON/AVX2, f32 NEON/AVX2+FMA, scalar fallbacks |
| `src/resize/kernels.rs` | Shared NEON kernels for resize (H pass, V pass, pyrdown, pyrup) |
| `src/warp/kernels.rs` | Per-arch dispatch for warp_perspective's 4-wide recip |
| `src/normalize.rs` | Reference single-dtype op with both NEON and AVX2 |
| `src/resize/fused.rs` | Canonical "fuse passes to halve memory traffic" pattern |
| `.cargo/config.toml` | Cross-arch build + qemu runner configuration |
| `examples/verify_normalize_avx2.rs` | Cross-arch correctness verification template |
| `examples/bench_normalize_cross_arch.rs` | SIMD-vs-scalar benchmark template |
