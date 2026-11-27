# TSDF cuda-oxide audit (kornia-rs fcd488e9, pin cuda-rust 6d47ae4 vs head 348895e)

Read-only. The shipped `kernels.ptx` is current for the commit: built 20:42:00, after the last `tsdf.rs` (20:40) and `tsdf_core.rs` (20:41) edits, tree clean. The pin checkout (`~/.cargo/git/checkouts/cuda-rust-27a4e94e7e45d2f6/6d47ae4`) is byte-identical to `/mnt/data/kr-tsdf-spikes/cuda-rust-src`.

## ptxas -v -arch=sm_87 (CUDA 12.6) on a copy of the shipped PTX

| entry | regs | stack | smem |
|---|---|---|---|
| tsdf_raycast | 78 | **32 B** | - |
| tsdf_mc_emit | 72 | **240 B** | 34,728 B |
| tsdf_integrate_pinhole / spherical | 66 / 64 | 0 | - |
| tsdf_touch_pinhole / spherical | 62 / 56 | 0 | - |
| tsdf_mc_count | 39 | **176 B** | 4,288 B |
| tsdf_rank_misses | 32 | 0 | 2,048 B |
| tsdf_assign_misses / merge / rehash / mc_dirty | 26 / 16 / 14 / 22 | 0 | - |

- No spill stores or loads in any entry.
- Zero `.maxntid` / `.reqntid` directives in the module.
- No kernel parameter is ever loaded with `ld.global.nc`. The only `.nc` loads (mc_count 13, mc_emit 28) read the `__device_global_*` marching-cubes tables.
- Trap sites per entry: mc_emit 12, integrate_spherical 8, integrate_pinhole 2, mc_count 2, touch 1 each, raycast 1. At the pin these are slice and array index checks only; SharedArray indexing is not checked at the pin.
- Shared-memory addresses are 64-bit (`%rd`), because the pin's llc runs without `--nvptx-short-ptr`.

---

## (A) Upstream practices, and whether the pin has each

| Practice (source) | Pin 6d47ae4 | Head 348895e |
|---|---|---|
| **noalias on params.** `noalias` and `dereferenceable` come only from reference params (`&T`, `&[T]`, `&mut`), via rustc FnAbi. Raw pointers get nothing. `readonly` is deliberately never emitted, so `.nc` loads only appear when the optimizer infers read-only (`mir-importer/src/translator/facts.rs`). | yes | yes |
| **Atomic-mutated buffers** go through `&[DeviceAtomicU32/U64]`. A write through a plain `&[u32]` is UB (`examples/hashmap_v2/src/lib.rs:1162`, upstream #151). | yes | yes |
| **Safe write path.** `DisjointSlice` + `thread::index_1d()`, `launch_contract(requires=...)`, `Uniform<T>` (`gpu-safety/the-safety-model.md`). This assumes `#[cuda_module]` launchers and one-thread-one-output kernels, so it does not fit the TSDF scatter/list kernels. | yes | yes |
| **Struct params by value** travel as one byval `.param`. `#[grid_constant] &T` needs sm_70 (`gpu-programming/kernels-and-device-functions.md` §Parameter constraints / Grid-constant). | yes | yes |
| **`#[kernel(unchecked_indexing)]`** removes `a[i]` checks on slices and arrays (`gpu-safety/bounds-checks.md`). | yes | yes |
| **SharedArray indexing is bounds-checked.** `TILE[i]` traps when `i >= N`, and `unchecked_indexing` removes that check (`advanced/shared-memory-and-synchronization.md`). | **no**: unchecked, as the pin book says | **new**: `memory.rs:735` `emit_shared_array_index` |
| **`#[launch_bounds(max, min)]`** emits `.maxntid`; `launch_contract(block=...)` emits `.reqntid`. | yes | yes; head adds a spill warning, cubin builds only |
| **Warp primitives.** `match_any_i64_sync` (sm_70), `ballot_sync`, `lanemask_lt`, `popc`, `shuffle_*_sync`. `redux_sync_*` needs sm_80, so it is not usable in the single sm_75 PTX (`advanced/warp-level-programming.md`). | yes | yes |
| **Warp dedup of hash inserts**: `match_any` with a sentinel key for inactive lanes, so the collective stays full-warp (`examples/hashmap_v3` `insert_kernel_dedup`). | yes | yes |
| **Shared memory across barriers.** Access through `S[i]` or `SharedArray::as_raw_mut_ptr(&raw mut S)`. Never keep a `&T`/`&[T]` into a shared slot live across `sync_threads()`, because rustc merges the reads. `ptx_asm!(options(register_only))` may be moved or deleted, so use it for register-only snippets. | API yes; rules not documented | rules documented |
| **DynamicSharedArray** sized at launch. | yes | yes, plus a pool-alignment fix |
| **Zero stack frames.** No runtime-indexed local arrays (cuda-oxide `AGENTS.md` §2). The "local could not be promoted" diagnostic runs under `CUDA_OXIDE_VERBOSE=1` (`compiler/compiler-optimizations.md`). | yes (`local_memory_diagnostic.rs`) | yes |
| **`#[unroll]` / `#[unroll(N)]`** on constant-step loops. | yes | yes |
| **`#[nounroll]`**, and `#[unroll(N)]` on block-stride (runtime-step) loops. | no | yes |
| **`#[inline]` / `#[inline(never)]`** carried to LLVM. | no (only `alwaysinline` survives) | yes |
| **`f32::to_int_unchecked`** compiles (#1595). | no | yes |
| **`--nvptx-short-ptr`** (32-bit shared addressing). | no | yes |
| **FMA policy.** Contraction is on by default (`llc -fp-contract=fast`), and `CUDA_OXIDE_NO_FMA=1` turns it off. `f32::mul_add` and `sqrt` lower to libdevice (`__nv_fmaf`, `__nv_sqrtf`), so kornia's inline `fma.rn`/`sqrt.rn` remains necessary. | same | same |
| **Inline PTX.** Upstream prefers typed intrinsics over `ptx_asm!`, but that rule is for upstream crates. `PtxOps` with `register_only` is correct here. | - | - |
| **Default backend** is llc. libNVVM is used only with explicit `CUDA_OXIDE_EMIT_NVVM_IR`. | same | same |

---

## (B) Code findings, ranked by impact

Unless a finding says otherwise, its fix keeps the bit-exact contract (only integers, ordering or memory spaces change) and works at the pin.

### 1. Per-thread stack frames in the marching-cubes kernels (mc_count 176 B, mc_emit 240 B)

- **Where:**
  - `tsdf.rs:931-951`, `:1068-1078`, `:1174-1184`: `Hood.bases: [u32; 27]` is copied from shared `BASES` into a per-thread array, then indexed at runtime in `slot()`. That puts it in `.local` and adds a trap compare on `bases[ni]`.
  - `tsdf.rs:956-966`: `conf()` builds `f = [0f32; 8]` and passes `&f` to `cube_index`, which also goes to `.local`.
- **Evidence:** PTX lines ~2680-2729 hold the `st.local` of the `-1` (NONE) initialiser and of the 8 zeros.
- **Upstream rule:** cuda-oxide `AGENTS.md` §2 "Zero stack spills"; `compiler-optimizations.md` "Diagnosing locals that remain stack-backed".
- **Fix:**
  - `Hood` holds `bases: *const u32 = SharedArray::as_raw_mut_ptr(&raw mut BASES)` and reads `*bases.add(ni)`.
  - `Hood::conf` computes `ci |= ((v < 0.0) as usize) << c` inline. This code is device-only, so the shared `cube_index` stays as it is.
- **Verify:**
  - ptxas reports a 0 B stack frame.
  - The gate's "every ld/st names a state space" check still passes. A pointer stored in a struct is exactly where address-space inference can be lost.
  - `CUDA_OXIDE_VERBOSE=1 cargo run -p kornia-oxide-kernels --example build_ptx` names any local that survives, including raycast's 32 B one. The variable reaches the nested build, because `build_ptx.rs:155-167` only strips `CARGO*`, `RUSTC*` and `RUSTFLAGS`.

### 2. Serial scans and contiguous chunking in `tsdf_mc_emit`

- **Serial scans:** `tsdf.rs:1214-1221` and `:1282-1289` each run a 256-step scan in tid 0 over shared memory while 255 threads wait at the barrier. That happens twice per block.
- **Contiguous chunks:** `:1194-1211` and `:1265-1279` give each thread a contiguous range of `ce` edges or cubes. This causes:
  - load imbalance, because `mesh_edge_point` (the expensive part) runs only on used edges;
  - strided u16 accesses to `IDS`;
  - uncoalesced vertex stores.
- **Upstream:** `warp-level-programming.md` "Filtering with ballot" and "Inclusive prefix sum".
- **Fix:**
  - Process edges round-robin, 256 per round (edge `r*256 + tid`).
  - Offset within the warp = `popc(ballot_sync(FULL, used) & lanemask_lt())`.
  - Combine the 8 per-warp totals in shared memory and carry a running base across rounds.
  - Vertex ids stay in ascending edge order, so the output is identical. Use the same pattern for the triangle (cube) pass.

### 3. Hash-probe and atomic traffic in the touch pass

- **Where:** `tsdf.rs:114-188`, `:261-272`. Every (pixel, block) pair does a `hash_find`, a `mark` swap and, on a hit, an atomic append.
- **Why it is costly:** adjacent lanes are adjacent pixels in a row. They mostly touch the same block keys and contend on the same `mark[p]`.
- **Upstream:** `examples/hashmap_v3/src/lib.rs:283-335`.
- **Fix:**
  - Iterate the touch boxes warp-uniformly: loop while `any_sync(FULL, more)`.
  - Call `warp::match_any_i64_sync(FULL, key_or_sentinel)`, and only the group leader calls `Touch::key`.
  - Set semantics keep this bit-exact.
- **Cost:** this is the largest restructure in the list. Measure touch time with nsys on a Replica frame first.

### 4. Raycast repeats the full hash lookup for every sample

- **Where:** `tsdf.rs:795-818` `View::slot`, reached from `raycast_pixel` (`tsdf_core.rs:1230`) and from the normal computation. Consecutive samples usually fall in the same block, which spans B voxels.
- **Fix:** a one-entry per-ray cache from block key to page base in the device `View`. This needs `Cell` or `&mut` in `VoxelSampler::sample`; the CPU twin can keep the default.
- **Bit-exact:** the lookups return the same values.
- **Source:** not an upstream rule. It is the nvblox-style practice.

### 5. No launch bounds, although the kernels hard-code 256 threads

- **Where:** `tsdf.rs:374` `TILE[256]`, `:1050` `WSUM` for 8 warps, `:1105` `for k in 0..8`, `:1151` `CHUNK[256]`. A launch with any other block size would silently corrupt memory.
- **Fix:** `#[launch_bounds(256)]` on every TSDF kernel. It exists at the pin and emits `.maxntid 256,1,1`, so the driver rejects other block sizes. It also lets ptxas allocate registers for 256 threads instead of 1024.
- **Gate:** `ptx_gate.rs:186-200` `header()` takes the name from the text after `.entry`, and the directive follows the parameter list. Confirm once with `tests/ptx_gate.rs`.

### 6. Bounds-check traps in the hot loops

- **Integrate:** the check on `depth[idx]` (`tsdf_core.rs:500`) repeats the `pinhole_coords` gate.
- **Marching cubes:** checks on `TRI_TABLE`, `EDGE_OWNER` and `bases`.
- **Fix:** `#[kernel(unchecked_indexing)]` on `tsdf_integrate_*` and `tsdf_mc_*`. The flag is per kernel; the host twin keeps its checks.
- **Preconditions:** run `compute-sanitizer` clean first, then check the PTX. The flag only covers code that rustc MIR-inlined into the kernel.

### 7. MC shared memory sized for B=16 on every launch

- **Where:** `tsdf.rs:1148-1151` declare 34.7 KB statically. B=8 needs about 6.5 KB (CONF 512 B + IDS 4,374 B).
- **Fix:** a `DynamicSharedArray` sized per B, passed as `shared_mem_bytes` in `launch_kernel`. That argument is 0 at `cuda.rs:193`.
- **Gain:** more resident blocks per SM. This is secondary while registers already limit residency.
- **Caveat:** the skill's "Smem loses both ways" law was measured on elementwise kernels only. Measure here.

### 8. `BLOCKS_PER_SM = 8` comment says "resident"

- **Where:** `cuda.rs:49-50`. At 62-78 registers per thread and 256 threads per block, roughly 3-4 blocks per SM fit in the 64K-register file.
- **Effect:** the extra blocks are harmless with grid-stride loops. Fix the comment, or derive the value with `cuOccupancyMaxActiveBlocksPerMultiprocessor`.

### 9. (Low) Per-step params go through device memory

- **Where:** `cuda.rs:850`, `760`, `769`. Each step does an htod copy of `StepParams` and the per-observation params, and each kernel starts with a dependent global load (`let st = *step`).
- **Fix:** pass them by value as a byval `.param` (supported at the pin).
- **Gain:** saves 1+n API copies per step and one load latency per kernel.

### 10. (Low) No `.nc` loads on data parameters

- Raw pointers carry no noalias, so data parameters never use `.nc` loads. On sm_87 that costs almost nothing (memory note: `__ldg` is a no-op there).
- Do not switch to `&[T]` to get it: the doc measured slices 12-13% slower on resize, and writes through `&[T]` are UB.

### 11. (Robustness) Silent dropped insert after a full probe cycle

- **Where:** the CAS loops in `tsdf_assign_misses` (`tsdf.rs:467-475`) and `tsdf_rehash` (`:552-562`). If a loop goes a full cycle without inserting, the block is lost silently.
- **Reachable today?** No. `hash_slots` (`cuda.rs:273-276`) is at least 2x capacity, so the load factor is at most 0.5.
- **Fix:** set an abort bit when the loop exhausts, as touch does with `ABORT_MISS_SET`.

### Checked, no issue

- `PtxOps` uses `register_only` (`tsdf_ops.rs:36,46`), which matches upstream's ordering table.
- The plain `vals` and `miss_list` stores after a CAS are consumed only by later kernels.
- Every `sync_threads` sits under block-uniform control flow.
- `add_warp_sum` and `warp_sum` are reached by all lanes after the loops (256-thread blocks).
- The u64 integer counters keep results deterministic.
- Shared-memory alignments are natural, so upstream's misalignment fix does not affect these kernels.
- `tsdf_rank_misses` is O(m²) (`:392-424`), which is fine at current miss counts. Revisit with a radix sort if first-frame misses exceed about 10^4.

---

## (C) Skill and doc edits (exact text)

### `/mnt/data/src/kornia-rs/docs/cuda-oxide.md`

Replace lines 181-188 with:
```
   - raw pointers, not slices (slices add a bounds branch: 12-13% on resize
     upscale); `noalias` comes only from reference params (`&T`, `&[T]`,
     rustc FnAbi) and `readonly` is never emitted, so raw-pointer inputs
     never get `ld.global.nc` (no-op on sm_87 anyway);
   - **all loads before the first store**: raw-pointer loads/stores still
     serialize when interleaved (2x on resize upscale);
   - **zero stack frame**: no runtime-indexed local arrays (copy-out of a
     `SharedArray` into `[T; N]` lands in `.local`). Check with
     `ptxas -v -arch=sm_87 kernels.ptx` (the build never runs ptxas, so
     upstream's spill warning cannot fire) and
     `CUDA_OXIDE_VERBOSE=1 ... build_ptx` (names the surviving local);
   - `#[launch_bounds(256)]` on any kernel sized for 256 threads
     (`SharedArray<_, 256>`, 8-warp loops);
   - shared memory: index `S[i]` or `SharedArray::as_raw_mut_ptr(&raw mut S)`;
     never hold `&T`/`&[T]` into a slot across `sync_threads()` (reads merge);
   - mirror each constant with a comment naming the CPU constant it copies;
   - backend gaps at 6d47ae4: `Ord::min` works; `f32::min`/`max` compile but
     the gate (G2) rejects them; `to_int_unchecked` does not compile (fixed
     upstream, #1595); no `__ldg`; `redux.sync` needs sm_80 (unusable in the
     sm_75 PTX); SharedArray indexing is unchecked at this pin (checked, with
     a trap, after it).
```

Append to the FMA paragraph (line 128):
```
`f32::mul_add`/`sqrt` lower to libdevice (`__nv_fmaf`) in this and later
cuda-oxide revs, which the gate forbids: keep `PtxOps`' inline `fma.rn`/
`sqrt.rn` with `options(register_only)` (pure register asm may be moved; fine).
```

### `/home/nvidia/.claude/skills/kornia-cuda-kernel-dev/SKILL.md` (no cuda-oxide content today)

Insert after line 92 (the end of "Device caches"):
```
## cuda-oxide kernels (crates/kornia-oxide-kernels/device)

- Read docs/cuda-oxide.md first; pin is NVIDIA/cuda-rust 6d47ae4 on
  nightly-2026-10-11. Upstream book: cuda-oxide/cuda-oxide-book.
- After every kernel change: `ptxas -v -arch=sm_87` on kernels.ptx. Target
  0 B stack frame, 0 spills. TSDF fcd488e9 shipped mc_count 176 B / mc_emit
  240 B / raycast 32 B stack: a `[u32; 27]` copied from shared and indexed at
  runtime, and a `[f32; 8]` passed by reference.
- `#[launch_bounds(256)]` on every 256-thread kernel; fcd488e9 had none.
- Block-wide ordered compaction/scan: ballot + popc(lanemask_lt) + per-warp
  totals, never a tid-0 serial loop behind barriers. Integer-only, so
  bit-exact.
- Hash touch/insert: dedup in the warp with `match_any_i64_sync` (sm_70)
  before global atomics (upstream hashmap_v3 `insert_kernel_dedup`). Inactive
  lanes pass a sentinel key so the collective stays full-warp.
- Atomic-mutated buffers: raw pointers + `DeviceAtomic*::from_ptr`, or
  `&[DeviceAtomicU32]`. Never write through a plain `&[u32]` (readonly/noalias
  UB, upstream #151).
- Not usable here: `redux.sync` (sm_80; the PTX is sm_75), `DisjointSlice`/
  `launch_contract`/`Uniform` (need `#[cuda_module]` launchers; ours are
  raw cudarc), `#[constant]` tables with data-dependent indices (serialize).
- Grid-stride residency is register-bound: 62-78 regs at 256 threads leaves
  about 3-4 blocks/SM, not the `BLOCKS_PER_SM` constant.
```

Two corrections in the same file:
- Line 14 (the f32 geometry rule) is NVRTC-only. Append: "(cuda-oxide: `CUDA_OXIDE_NO_FMA=1` + explicit inline `fma.rn` at contract sites; PTX gate G1-G4)."
- Lines 99-107 ("Smem loses BOTH ways"): append "Measured on elementwise/separable kernels only. Not established for reuse-heavy kernels (MC's 8 corners per voxel); measure."

### Repo skills

- None of the files under `/mnt/data/src/kornia-rs/.claude/skills/` hold kernel-authoring guidance. The kornia-arch:27 and release:25-32 references are accurate.
- Add to `kornia-code-review/SKILL.md` after line 164:
```
- cuda-oxide kernels: ptxas -v shows 0 B stack per entry, `.maxntid` present
  where shared arrays assume a block size, no tid-0 serial loops behind
  `sync_threads()`.
```

---

## (D) Pin bump 6d47ae4 -> 348895e

**Verdict: defer it to a separate change after the TSDF PR.**

**Gains for these kernels (modest):**
- `--nvptx-short-ptr` gives 32-bit shared addressing in mc_* and rank.
- `#[nounroll]` and partial unrolling of block-stride loops.
- Real mapping of `#[inline]` / `#[inline(never)]`.
- Entry-block alloca hoisting.
- `to_int_unchecked`.
- None of findings 1-11 need the bump; all are fixable at the pin.

**Costs and risks:**
1. **Toolchain.** The pin's nightly-2026-10-11 becomes stable `1.99.1` + `RUSTC_BOOTSTRAP=1` (`cuda-oxide/rust-toolchain.toml`).
   - Touches `device/rust-toolchain.toml`, `install-toolchain.sh`, the backend sha256, the `.backend` stamp format (`<rev> <nightly>`), and the docs.
   - The backend uses the toolchain's own llc/opt, so the LLVM version may change. That means a full parity and bench rerun.
2. **SharedArray indexing becomes bounds-checked.** Every `TILE`/`CONF`/`IDS`/`CHUNK` access gains a compare and trap: a silent perf regression in rank and mc, unless those kernels get `#[kernel(unchecked_indexing)]`.
3. **FMA default unchanged.** Contraction stays on and `CUDA_OXIDE_NO_FMA` is still honoured (`device_codegen.rs:1239`). Low risk; the gate still catches regressions.
4. **CUDA 12.6 / R540: no blocker found.**
   - The default backend is llc, not libNVVM.
   - The PTX ISA floor logic is unchanged (ISA is raised only for features or full debug), so `.version 6.3` should hold. Confirm with the gate on the first bumped build.

The one-PR rule argues for folding the bump in. Mixing a toolchain swap and a codegen change into the TSDF parity PR would make any parity failure ambiguous, so I recommend keeping it separate. That is a judgment call for the owner.

**Files referenced:**
- `crates/kornia-oxide-kernels/device/src/{tsdf.rs,tsdf_ops.rs,tsdf_core.rs}`
- `crates/kornia-3d/src/tsdf/cuda.rs`
- `crates/kornia-oxide-kernels/{ptx_gate.rs,examples/build_ptx.rs}`
- `docs/cuda-oxide.md`
- Upstream: `/mnt/data/src/cuda-rust-upstream/cuda-oxide/{AGENTS.md, cuda-oxide-book/, crates/mir-importer/src/translator/facts.rs, crates/mir-importer/src/translator/terminator/intrinsics/memory.rs, crates/cuda-oxide-codegen/src/ptx.rs, crates/rustc-codegen-cuda/examples/hashmap_v3/src/lib.rs}`
