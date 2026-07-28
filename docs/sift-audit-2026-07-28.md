# SIFT optimisation audit — 2026-07-28

Working notes, not results. Three-pass audit: an inline first pass against a
fresh `nsys` profile, four read-only finder agents with distinct lenses, then an
adversarial skeptic pass over the 12 highest-value claims. **Verdicts are
integrated below** — see "SKEPTIC VERDICTS" for the summary table. Nothing has
been implemented. Line numbers are as of commit `07f5f1d2`.

## SKEPTIC VERDICTS (adversarial verification, source-level)

| finding | verdict |
|---|---|
| B1 dead border writes | CONFIRMED + corrections (see below) |
| D1 red[] smem | CONFIRMED-CORRECTED: transition is 1→2 blocks/SM not 2→3 (1024 B/block reservation omitted); carveout claim is ASSUMPTION — code sets only the legacy PREFER_L1 hint (kernels.rs:499 → cuda.rs:517-521), never the carveout attribute. B5 half (dead levels) fully confirmed bitwise. Gate on ncu. |
| D3 extrema registers | **mechanism REFUTED**: all derivatives die inside the loop body (detect.rs:246-280); only xi/xr/xc/rr/cc/layer/step live across iterations, and body k+1's loads depend on rr/cc written at :290-292 so nothing hoists. `#pragma unroll 1` would be I-cache only. Reg count UNVERIFIABLE from source — `-Xptxas -v` decides; if it IS ~96, the cause is one body's working set and unroll-1 is the wrong fix. Traffic floor derivation stands. |
| D4 blur_h divergence | CONFIRMED; closed-form refl101 valid iff n>=n2+1 (met with margin); `n<2*n2` fallback guard safe. Cover blur_hv_fused_src:380 in any helper swap. |
| C1 downsample udiv | CONFIRMED, identical integer; only-caller verified; CUDA twin verified; sy is per-row (only x-div is per-pixel). |
| C2 pr staging | identity CONFIRMED (forces fully-unrolled t-loop — 4 explicit blocks); **mechanism corrected**: the loads ARE forwardable (contained+aligned), the real win is store-port pressure (~33% of stores). 3-10 ms unsubstantiated until LDST replay counters. |
| B2 submit starvation | mechanism real, **1.7 ms REFUTED**: launch count is keypoint-independent yet the fast-tier TOTAL gap is 1.0 ms — overstated >=2x. Octave-0's ~2 ms of queued blur absorbs most refills. |
| B8 host bubble | window corrected to :464-507 (the :508-582 span is INSIDE t_desc — since() syncs). sort_by_cached_key applies to plan.rs:513 only; sorted_dedup_order has no Ord key without a hand-built ordered-bits tuple. Disambiguating measurement: Instant around :464-507; B2 = gap − that − remainders. |
| B3/D7 per-layer counts | **FALSIFIED BY IMPLEMENTATION** (2026-07-28, oracle test `budget_caps_the_count_and_keeps_the_strongest`, 472≠471): `adjustLocalExtrema` reassigns a keypoint's LAYER during refinement, so `k[5]` is the *refined* layer while slot contiguity is by *launch* layer. The old `layer_filter` partitioned by refined layer, pairing each keypoint with the right Gaussian image; per-launch-layer slices pair some with the wrong image. Three finders and the skeptic all missed it. A device-side fix would need a partition by refined layer — likely not worth it against the 0.1-0.35 ms bound. |
| D5 orientation shfl | CONFIRMED; exact-tier-only; **gate emitted source on threads==32** — KORNIA_SIFT_ORI_T can set 64..1024 in exact mode and a shfl-only rewrite breaks it. omax warp-max proven exact (no NaN per hal.rs:224-227 rsqrts special case; no −0.0 anywhere in the chain). Saving is barriers/latency only — occupancy capped by the one-warp shape, not smem. |
| B4 dead memset | CONFIRMED; **rotation blocker**: Tensor::from_cudaslice takes ownership — plan-owned rotation needs a borrowing/Arc tensor variant first. |
| D8 runtime div | CONFIRMED both sites; orientation's incremental form is map-preserving hence bit-exact. |
| C3 collect vectorise | CONFIRMED-CORRECTED: clip_j is a tight superset, rejects only at run ends, but that still makes ~20-35% of 4-wide blocks partial → **scale estimate by ~0.7** (≈3-5 ms). The :322 "5.50M/2.63M" comment is the unclipped-square ratio, not a fallback-rate datum. Third contraction trap: r_rot (:356) is mul,mul,add — never vfmaq. |

New items found by the skeptic in passing:
- **hist o-bin 9 read-but-never-written**: fold reads hist[idx+NN+1] (:232,:415,
  CPU :400) but writes never reach offset 9 — the +0.0 add is a provable no-op,
  deletable in the same patch as B1.
- The CPU "360 zeroed / 240 readable" number in the earlier notes is WRONG:
  readable = 160 floats. (Corrected in-place below? No — treat this line as the
  correction; the win is smaller than stated.)
- B9 understated: each blur_h launch also runs format! + a full
  kernel_digest over all 27 taps to build the cache key (pyramid.rs:360-364),
  per launch, not per compile.
- B1 precision: in the ATOMIC kernels, skipping changes contention → arrival
  order of surviving atomics; those kernels are already documented non-bit-exact
  (:277-281), but the finding must not claim their outputs unchanged.
- B1 NEON bookkeeping: per-lane predicates need r0/c0 spilled alongside ib —
  2 extra vst1q_s32 per 4 samples.
- D10 (fast kernel hard-codes 512 threads) confirmed while checking D1.
- algo-contract's refinement-derivative recompute confirmed byte-identical.

## Baseline (nsys, this build, 752x480 mh01_frame1)

| | wall | GPU busy | gap |
|---|---|---|---|
| exact `fo=-1` (2515 kp) | 17.5 ms | 15.5 | 2.4 |
| fast + budget 500 | 9.56 ms | ~8.6 | 1.0 |

Per frame: descriptor_block 7.62 ms, orientation_block 2.24, blur_v_dog 2.12,
blur_h_tiled 1.66, find_extrema 1.38, descriptor_fast(500) 0.65, base ops 0.51.
API side: 8 blocking `cuMemcpyDtoHAsync` (avg 1.3 ms inside the call — pageable
D2H drains the stream), 25 DtoD, ~120 launches ≈ 1.8 ms/frame CPU submit.

CPU single-thread stages: pyramid/blur 64.5 ms (carotene: 42.7), descriptors
83.3 (78.6), extrema+orientation 35.4 (38.7, ahead).

## Confirmed by two independent agents

### A1. `ranges` in-loop snapshots are dead code
`plan.rs:409-413` copies `ori_count -> ranges[range_i]` 18x/frame D2D; nothing
reads it before `plan.rs:563` overwrites it with host-computed `starts` (only
reader `:577`, after the overwrite). Orphaned by the retain_best reorder; the
design comment at `:398-405` describes the superseded revision. Explains 18 of
the 25 measured DtoD. Delete `:409-413` + `range_i`.
**~0.1-0.15 ms + 18 fewer graph nodes. Certain. Both agents, independently.**

## Single-source findings — CUDA (pending skeptic verification)

### B1. Descriptor histogram: 36% of scatter writes hit never-read border cells
All four descriptor kernels, both backends. The fold reads only interior cells
(rows/cols 1..4 of the 6x6 cell grid); 20 of 36 cells are write-only. With
r0 = floor(rbin) uniform in {-1..3}, P(a target cell is dead) = 1 - 0.8^2 = 36%
of the 8 accumulations per sample. Skipping is bit-exact on every tier —
disjoint addresses, no reordering of live accumulations. Four hoisted predicates
per sample gate four adjacent pairs.
CUDA: `cuda/sift/descriptor.rs:397-405`/`:213-221`/`:617-625` (scatter),
`:411-417`/`:228-236`/`:629-635` (fold). CPU:
`features/sift/descriptor.rs:196-210` scatter, `:247-255` tail, `:396-405` fold.
**CUDA 0.8-1.7 ms (of ~4.8 ms atomic share; dead adds are the less-contended
addresses so expect the low end). CPU 3-8 ms of 83.3. Deadness: proof sketch
solid. Magnitude: medium confidence.**

### B2. Host gap reframe: drains destroy queue depth; submit stops overlapping
The 6 mid-loop `clone_dtoh(kp_count)` drains (`plan.rs:391`) don't just block —
each empties the queue, after which ~19 serial submissions at ~15 us refill it
while the GPU starves. ~1.7 ms/frame estimated. The value of deferring counts is
hiding the 1.8 ms submit cost, NOT saving the blocking time (which is mostly
legitimate GPU work being waited on).
**Most of the 2.4/1.0 ms gap. Medium-high.**

### B3. Per-layer counts are free; orientation launches 3x the blocks needed
`plan.rs:374-389` launches find_extrema per layer sequentially on one stream, so
layer L's keypoints occupy a contiguous slot range in `kp`. Snapshot the counter
into a 4-int device array between launches; the existing per-octave D2H reads 16
bytes instead of 4. Orientation (`plan.rs:406-434`, one block per keypoint,
grid = whole octave count) then takes per-layer grids; `layer_filter`
(orientation.rs:265-266) becomes redundant. ~5000 of ~7500 blocks/frame retire
on the filter today.
**0.1-0.35 ms. Deadness certain; magnitude medium.**
TRAP (host-orchestration): a naive conservative grid at max_keypoints=8192 is
~147K blocks/frame — a net LOSS of 0.3-0.7 ms. Per-layer counts first, then
device-side count + grid-stride with a fixed ~1024-block grid.

### B4. own_descriptors: dead 1.29 MB memset/frame + the graph-capture blocker
`kornia-py/src/cuda_ext/cuda_sift.rs:113-118`: `alloc_zeros` then D2D overwrites
every byte — memset 100% dead (~20-30 us). Precedent for uninit alloc with the
same SAFETY shape: `cuda_ext/mod.rs:1239,1247`. Stream-ordered alloc is also
what breaks graph capture (`mod.rs:1687-1690`). Better: 2-deep rotation of
plan-owned output buffers removes the alloc AND the D2D (~25 us), serves the
two-frame matching case the D2D exists for. Make deeper history opt-in.
**~0.05 ms + capture unblocked. High.**

### B5. Descriptor block reduction runs two provably-zero tree levels
`cuda/sift/descriptor.rs:422-428`, `:438-443` (fast twin `:638-645`,
`:655-660`): NTHREADS=512, DLEN=128, so red[128..511]==0 and the off=256/128
levels add exact zeros. Start at off=64: bitwise identical, removes 4
`__syncthreads` x2 per keypoint of ~18.
**0.1-0.3 ms. Deadness certain.**

### B6. Pinned-memory conversion: worth ~nothing directly, and dangerous naive
cudarc's `memcpy_dtoh` into `Vec` does NOT synchronise (`SyncOnDrop::Sync(None)`);
correctness rests on pageable-D2H-blocks semantics. Swap to pinned and the read
races — silently. And `CudaContext::alloc_pinned` is WRITE-COMBINED (wrong
direction; reading ori_kp from WC is slower than pageable). Right primitive:
`kornia_tensor::zeros_pinned` (cacheable). 6 of 8 copies are 4 bytes; only
ori_kp (~60 KB) benefits: 20-40 us.
**Direct gain <0.05 ms. Value is graph-capture enablement only. Mechanism high.**

### B7. Graph capture: ~83 of ~120 launches already shape-static
Blur+DoG chains (60), detect (18), base ops, downsamples — capturable as one
graph once B2/B3 land. Breakers, in order: the mid-loop clone_dtoh; four
pageable H2D (`plan.rs:537,560,563,589` — need persistent pinned staging);
own_descriptors' stream-ordered alloc (B4); variable descriptor launch count
(`:544-558`). Capture the blur/detect PREFIX only — that's the 80% and needs no
descriptor-side topology work. Budget the ~1.25 ms saving AGAINST B2, not on
top of it (both recover the same wall-GPU gap).
`Graph.capture` disables cudarc event tracking context-wide (`mod.rs:1748-1751`)
— see B10 stream hazard.

### B8. Unmeasured host bubble after the ori drain: 0.3-1.0 ms
`plan.rs:468` drains, then `:476-563` is pure host work on an empty queue:
all_kps build, `sorted_dedup_order` (6-level total_cmp indirect sort, ~28K
cache-hostile comparisons), `sort_by_key` re-evaluating a double-indirect key
O(n log n) times, fresh `vec!` allocs for din/perm/describe. Invisible to the
stage timers (they bracket launches only). Gap scaling (2.4 ms at n=2515 vs 1.0
at n=500) is consistent. Actions: time `:464-593` FIRST; then
`sort_by_cached_key` (stable, key-identical — no ordering change) and plan-owned
reusable buffers.
**0.3-1.0 ms. Medium-high mechanism; medium split.**

### B9. env::var on the hot path: `tile_p()` and `ostride()`
`cuda/sift/pyramid.rs:331-337` called 2x per launch, ~74 environment scans +
format! per frame; `cuda/sift/descriptor.rs:296-302` per fast launch. Five
sibling knobs already use OnceLock.
**0.05-0.2 ms host. Certain, low value.**

### B10. PlanKey hazards (correctness, not perf)
(a) Two-size alternation on one Sift instance rebuilds the plan every frame
(tens of MB alloc+memset); PlanSlot should be a 2-entry LRU.
(b) PlanKey omits the stream; plan buffers are ordered today only by cudarc
event tracking, which Graph.capture disables context-wide. Add stream identity
to the key or reject stream changes. Latent now, load-bearing once capture
lands. `cuda_ext/cuda_sift.rs:174-186`.

### B11. keypoints_to_list: ~2515 Python objects/frame
`kornia-py/src/sift.rs:119-136`: one pyclass alloc per keypoint, ~0.4-0.6 ms +
250 KB churn/frame. The docstring's `np.fromiter` advice re-walks objects
through the interpreter, adding cost. Fix: additive columnar API
(one (N,6) f32 + one (N,) i32 packed array, ~10-20 us). Keep the list return.
**0.4-0.6 ms off the Python frame. High mechanism, medium magnitude.**

### B12. First-pass items (inline audit, still open)
- **u8 + `__dp4a` matcher**: descriptors are exact integers 0..255 in f32; u8 is
  lossless with i32 accumulator. ~8x fewer inner ALU ops, 4x traffic.
  `matcher.rs:52-97` inner loop is f32 float4 + butterfly. Biggest win outside
  detect ms — moves the SLAM frontend total. High confidence.
- **Exact descriptor sample packing** (plan W2): `descriptor.rs:369-382` still
  sweeps the full square, rejects ~51% of lanes at `:373`. Fast kernel got its
  grid narrowed (`:513`); exact didn't. Ceiling ~1.0-1.3 ms (atomics invariant).
  Note (algo-contract): the DEFAULT block kernel lacks it too, not just Exact.
- **find_extrema 2x over its ~0.7 ms streaming floor** — mechanism not yet
  identified. Low-medium.

## Single-source findings — CUDA microarch (kernel-microarch agent, pending skeptic)

### D1. `red[NTHREADS]` smem sizing IS the descriptor's 2-blocks/SM cap
`descriptor.rs:338`: red 2048 B brings static smem to 4000 B; PREFER_L1 grants
8 KB ⇒ floor(8192/4000)=2 blocks/SM. **The 2-blocks/SM cap previously
attributed to smem *staging* is actually the reduction scratch.** Sizing red to
min(NTHREADS,128) (=512 B, since DLEN=128) gives 2464 B ⇒ 3 blocks/SM (+50%
residency) AND removes the same two dead tree levels as B5 (red[128..511] is
exact +0.0; sums of squares, no -0.0 possible). Same fix in the fast kernel
(`:558,:642-645,:657-660`).
**0.3-0.8 ms of the non-atomic ~2.8 ms half; fast tier 0.05-0.15. Bit-identical
both tiers. High on the smem arithmetic; gate with
launch__occupancy_limit_shared_mem.** Subsumes B5 (adds the occupancy half).

### D2. No kernel declares `__launch_bounds__`
`descriptor.rs:320,:540; orientation.rs:254; detect.rs:160; kernels.rs:217,:283`.
Block sizes are compile-time literals but ptxas never told — registers
allocated against an assumed 1024-thread block. Bonus: orientation at 32
threads is a single warp, but its four `__syncthreads` still emit BAR.SYNC;
`__launch_bounds__(32)` lets ptxas elide (~100 barriers/keypoint).
**0.1-0.4 ms pipeline-wide. One line per kernel. Medium (check -Xptxas -v).**

### D3. find_extrema is NOT traffic-bound — register pressure from the unrolled
refinement loop. `detect.rs:241`: MAX_INTERP_STEPS=5 is compile-time, nvcc
unrolls, ~10 derivatives + pointers + det3 temporaries live across 5 bodies ⇒
~96 regs ⇒ 2 blocks/SM (33%) on a kernel whose 99.9% path is a latency-bound
gather. Fix: `#pragma unroll 1` — code layout only, zero numerics.
Independent traffic derivation: prev/next planes read only by the ~3% passing
the 8-neighbour test ⇒ ~36 MB/frame ⇒ 0.45-0.66 ms floor (confirms first-pass
estimate) — and KILLS layer-fusion as a fix (planes already sparse).
**0.3-0.6 ms. Bit-identical. Medium-high diagnosis.**

### D4. blur_h_tiled diverges intra-warp; blur_v is warp-uniform — that's the
46 vs 73 GB/s. `kernels.rs:225` tests x (warp axis) vs `:289` testing y.
Warp 0 of every row runs BOTH border and interior paths; the border path is
P*K unrolled taps each calling `refl101` — a while-loop the compiler can't
prove single-step ⇒ ~430 instr vs 92, diverged warp ~5.7x, kernel ~1.39x.
Fixes (compose, bit-exact — identical indices): closed-form refl101
(`i<0 ? -i : i>=n ? 2n-i-2 : i`, valid since n2<=13 < 16-px octave floor;
keep loop as n<2*n2 fallback) + warp-uniform region test (64-col granularity).
Inflation 1.39 → ~1.09. **0.2-0.4 ms. Medium-high. NOT the falsified smem/H+V
territory.** blur_v itself is AT its roofline (72.6 GB/s) — leave it.

### D5. Orientation one-warp block stages through smem + barriers where shfl works
`orientation.rs:231-243,:271-273,:299-316`: chunk data staged via 3 shared
arrays + 2 barriers per chunk; `__shfl_sync` moves the same triples with no
barrier, fold order untouched (q ascending, chunks in order). Also
`:330-334`: omax is a serial 36-element scan by tid0 between barriers;
fmaxf warp-max via 5 `__shfl_xor` is provably identical (no NaN reaches hist).
**0.2-0.6 ms of 2.24, exact tier. Medium.**

### D6. Descriptor group launches serialize on one stream, ragged tails
`plan.rs:564-580`: ~15 launches, one block/keypoint, 16-24 resident; each
group's last wave half-idle, runtime spread ~3x within a group (radius²).
No inter-dependency (disjoint desc_all rows; ranges/desc_live read-only) ⇒
round-robin 2-4 streams + join event. **0.15-0.35 ms. Medium.**

### D7. Per-layer ranges for orientation — THIRD independent derivation
Same as B3/algo-contract#2; adds: the D2D snapshot idiom at plan.rs:410-413
(the code A1 deletes) is the right MECHANISM for the fix — B3 resurrects it
with an actual reader. Also records: any future find_extrema fusion must keep
three counters or it destroys the contiguity B3 relies on (moot per D3).

### D8. Runtime integer division in three inner loops
- `descriptor.rs:365-366`: s/side, s%side per sample, side runtime ⇒ ~20 instr
  x 4.65M samples ≈ 0.15-0.2 ms. Incremental carry (di=NTHREADS/side hoisted)
  or full remap — sample↔thread map not contractual (atomics reorder anyway).
- `orientation.rs:304`: s/nx, s%nx, ~2.4M samples; incremental form preserves
  the exact map ⇒ bit-exact. 0.05-0.1 ms.
- `kernels.rs:421`: sift_downsample_nearest — THE GPU TWIN OF C1. Two runtime
  divs/px; caller always passes dw==sw/2 ⇒ warp-uniform sx=2*x fast path.
  0.03-0.04 of 0.09 ms.
**Combined 0.2-0.3 ms. Bit-exact. Medium-high.**

### D9. find_extrema loads misaligned by 20 B systematically
`detect.rs:170-171`: +IMG_BORDER(5) puts lane 0 at column 5+32k ⇒ every
warp access spans 5 sectors where 4 suffice (+25% LSU sector requests).
Fix: index from 0, predicate the border (near-warp-uniform). Only pays after
D3 restores occupancy. **0.05-0.2 ms. Medium/low-medium.**

### D10. sift_descriptor_fast never swept its block size
`descriptor.rs:782-789` hard-codes 512, ignoring desc_block_threads(); at
T=128 red is 512 B, 7 levels, ~2464 B smem, 4x blocks/SM. Loop utilisation
identical at 128/256/512. Cache key already includes T. **Fast only,
0.1-0.2 ms at budget 500. Cheap sweep.**

### D11. Smaller bit-exact items (kernel-microarch)
- `detect.rs:186-228`: val>0/val<0 arms load the same 8 curr neighbours twice
  under mixed-sign warps; branch-free s*val form is exact (no ±0 possible).
  0.05-0.1 ms.
- `plan.rs:442-456`: downsample→buf_a→D2D into pyr[octv+1][0]; split_at_mut
  (the plan.rs:356 trick) removes the 3.85 MB round trip. ~0.05 ms.
- `descriptor.rs:343-353`: 512 threads redundantly compute cosf/sinf/radius/
  diag; pass diag as arg, hoist rest. 0.05-0.1 ms.
- `descriptor.rs:411-417`: fold runs on 16 of 512 threads; one-thread-per-
  output (128 active) is the identical expression per element. ~0.03 ms.
- `descriptor.rs:525-538`: sift_tex clamps provably dead (caller established
  1<=x<=w-2). Fast only, ~0.01-0.05 ms.

### Additional negatives (kernel-microarch)
blur_v_dog at roofline (72.6 GB/s, warp-uniform borders, right shape); no
cross-kernel gradient reuse possible (different radii, different keypoint
sets, descriptor deferred past budget — a fused epilogue cannot exist);
address math already 32-bit where it matters; orientation smoothing %36
already multiply-shift; gather/upsample/dog at floors; matcher butterfly
minimal — dp4a is the only lever there; hal.rs tables L1-resident, cap
0.2 ms even if free.

## Smaller CPU items (algo-contract)

- Refinement recomputes last-iteration derivatives post-loop, both backends
  (CUDA `detect.rs:305-320` vs `:246-255`; CPU `detect.rs:406-430` vs
  `:319-329`). Bitwise identical to carry in registers. 0.02-0.05 ms; on CUDA
  only if ncu says registers aren't the limiter.
- One dead downsample+copy per frame: both backends build octave N+1's base then
  break on the `< 16` guard (CUDA `plan.rs:437-458` vs `:335`; CPU
  `pipeline.rs:473-485` vs `:411`). ~0.01 ms, hoist the test.
- CPU `hist` zeroing: 360 floats zeroed/keypoint, only 240 readable
  (`features/sift/descriptor.rs:301`). ~0.1 ms; pairs with B1.
- CPU rebuilds Gaussian kernels every frame (`pipeline.rs:345-348`, ~180
  f64::exp); CUDA caches them at plan build. Microseconds; hoist into
  SiftWorkspace.
- SPECULATIVE: duplicate keypoints oriented then deduped
  (`pipeline.rs:576-608`). Saving = duplicate_rate x orientation cost; rate
  unmeasured. Measure before building. CPU-side hash-set pass is cheap; CUDA
  device dedup probably not worth it.

## Single-source findings — CPU/NEON (neon-cpu agent, pending skeptic)

### C1. `downsample` runs a 64-bit udiv per output pixel that provably equals 2*x
`pipeline.rs:170-172`: `(x*sw)/dw` with `sw ∈ {2dw, 2dw+1}` (only caller passes
halves) reduces algebraically to `2x` for all `x < dw`. A78's divider is
non-pipelined, ~15 cyc, in a load+store loop. ~481K output px at fo=-1.
**~4.8 ms single-thread (~0.8 ms at 6T). And it is INVISIBLE to the stage
probes** — sits between `t_ori`'s end (`:471`) and the next `tb` (`:416`).
Identical integer, not merely bit-exact. Very high confidence.

### C2. Descriptor scatter's `pr` staging buffer is a store-forwarding round trip
`descriptor.rs:188-209`: 8x vst1q into a stack array then 16x vld1 at
half-offsets — partial-overlap loads A78 cannot forward, each replays through
L1; also 8 extra stores on the single store port per 4 samples. Replaceable
register-to-register with vzip1q/vzip2q + vget_low/high (and vgetq_lane for
`ib`, `:177-178`). TRAP: keep the `t`-outer loop order — lanes alias histogram
addresses, so slot-outer reorders accumulation and silently breaks bit-exactness.
Precompute the 8 zips before the `t` loop.
**3-10 ms of 83.3 (stall cost unboundable from source — perf LDST replay
counters first). High mechanism, medium magnitude.**

### C3. Descriptor collect loop scalar and store-port bound
`descriptor.rs:354-378`: 5 scalar stores + 4 bounds-checked loads per accepted
sample, ~2.6M samples — a 5 cyc/sample floor ≈ 8.7 ms before arithmetic.
clip_j already makes 4-wide blocks almost always all-accepted: mask, then 4
vector loads + 5 vector stores, scalar fallback else. TRAPS: `c_rot` is
mul-then-sub (two roundings) — vmulq+vsubq, never vfmsq; the weight expression
is mul,mul,add,mul — never vfmaq. **4-7 ms. Medium-high.**

### C4. `upsample2x` horizontal pass scalar+branchy; weights are a fixed 2-phase
`pipeline.rs:113-148`: `tap()`'s two conditionals block vectorisation. 2x
upscale weights are exactly {0.75,0.25}/{0.25,0.75} over consecutive pixels:
4 loads -> 8 outputs via two vmulq+vfmaq+zips; peel first/last pair for clamps.
**2-4 ms of ~12 ms t_base. Bit-exact: identical expression. Medium-high.**

### C5. Blur H border: serial-FMA scalar taps + refl101 per tap
`scalespace.rs:83-90` + `:103-110`: ~872 scalar taps/row ≈ 3500 cyc/row vs
10200 for the whole vector interior at w=752; border/interior ratio doubles
each octave, exceeds interior by octave 3. Fix = FilterEngine shape:
materialise a 3*n2 reflected edge buffer per side, run row_interior over it —
identical taps, identical order. **2-4 ms. High.**

### C6. Blur V: 24-byte tuple loads per tap in the inner loop; no 2-row blocking
`scalespace.rs:332-342, 358-377`: `for &(pa,pb,kc) in pairs` adds 2-3 load
slots per tap over the 8 data loads; runtime slice blocks unrolling. Only 3% of
rows need reflection — split on `const REFLECT: bool`, strength-reduce interior
indices (~10% off V). Bigger: adjacent output rows share 2*n2 of 2*n2+1 inputs;
2 output rows per pass cuts loaded bytes ~47%, moves V to the ALU bound.
**~20% of V ≈ 4-6 ms. Medium (register pressure risk).**

### C7. The "22 ms blur gap" is mostly a scope mismatch — mechanism resolved
`pipeline.rs:30-32` records ours 45.0 vs cv2 42.7 for the five layer blurs
(0.95x/core). The 64.5 additionally contains t_base (~12 ms) and octaves 1-4
(+33%), neither in cv2's 42.7. Reconstructed 45*1.33+12 ≈ 72 brackets 64.5.
No structural carotene trick we lack; the recoverable parts are C1/C4/C5/C6.
Also answered: row-buffer ring is contract-legal but costs 163% redundant H
work per rayon task at ksize 27 — a wash at 6T, do C5/C6 first. Non-temporal
stores are the wrong tool (every buffer re-read within L2 promptly).

### C8. Small CPU items (neon-cpu)
- `map_init` orientation closure allocs a Vec per keypoint
  (`pipeline.rs:459-465`): ~2500-5000 malloc/free per frame; for_each_init with
  per-worker Vec + fold join. 0.2-0.5 ms.
- `assign_orientations` uses 3x Vec::push per sample where the descriptor path
  deliberately converted to sized+get_unchecked (`orient.rs:109-130`; count is
  closed-form at `:96-97`). ~0.5-1 ms.
- nrm2 + final quantisation scalar over 128 (`descriptor.rs:57-73`, `:416-418`);
  ~0.3 ms, vectorisable bit-exactly (nrm2's 4 accumulators ARE 4 lanes; FRINTN
  for round_ties_even). DO NOT touch the second accumulator `:409-414` — its
  scalar-serial shape is oracle-pinned, asymmetry deliberate.
- Rayon fork-join below ~32k px is pure loss (~62 regions/frame; octave-4 V
  task ~5 us vs 2-5 us spawn cost); serial threshold. 0.5-1.5 ms of the 6T
  total. Low confidence.

### C9. Open inconsistency to pin before trusting stage numbers
Stage sum 64.5+35.4+83.3 = 183.2 ms vs `pipeline.rs:11` claiming ~138 ms exact
fo=-1. Docstring stale, or stages measured under a different config. Resolve
before attributing further CPU wins. (C1 being probe-invisible may be part of
the discrepancy.)

### Additional negatives (neon-cpu)
No small-radius descriptor regime (scl is octave-normalised, radius 17-34
everywhere, tails <=0.2%); histogram fold vectorisation is noise; vextq in the
H pass is a wash (load and ALU ports both 2-wide, balanced); no measurable
false sharing (rayon split points 64-byte-congruent); mag/atan2 batch buffers
correctly reused via grow_to; DoG-hot extrema fusion dead (3 planes = 17 MB vs
2 MB L3); prefilter's scalar confirm stays (NaN guard, runs on 0.09%).

## Verified-tight (do not re-audit)

Extrema contrast prefilter placement (both backends test before neighbour
loads); border exclusion by launch geometry; adjustLocalExtrema iteration
economy; orientation weight underflow (min exp(-9), nothing discardable);
orientation smoothing single-pass, no tail; ori_kp threshold filter pre-atomic;
pyramid plane liveness (every Gaussian and DoG plane read); base upsample sized
exactly; normalisation fold reads no border bin; post-budget buffers sized at n;
matcher host side (2 D2H, both needed, post-launch); MatchStore::ensure guard;
no per-frame kernel recompilation; NEON prefilter's scalar confirm redundant for
accepted lanes but runs on 0.09% and guards NaN — leave.

## Standing falsifications (measured; do not re-try without a new mechanism)

Smem staging / H+V band fusion (5 regressions; PREFER_L1 carveout mechanism),
f16/half2, approximate transcendental HAL (whole exact HAL 0.73 ms), fastmath
knob (was a regression), `__constant__` vs baked literals, multi-px-per-thread
ILP (4-for-4 regressions), 32-bit index math in FMA kernels, x-tiling the
vertical blur, interleaved planes, batch-helper sharing at width 4.

## Known errata in agent reports

- host-orchestration inferred the nsys profile came from the Rust example
  ("costs additive to your numbers"). Wrong: the profile WAS the Python binding
  (prof_sift.py), so own_descriptors' alloc+memset and keypoints_to_list ARE
  inside the measured wall times.

## Adjacent (out of module)

`examples/cuda_camera_sift/src/main.rs:177` uploads a 1.44 MB pageable Vec
per frame; pinned staging + the wait_prev_upload pattern from
`cuda_ext/mod.rs:709-714`.

## Final order (post-skeptic)

Measure-first gates (cheap, do before ordering anything behind them):
- Instant around plan.rs:464-507 → pins B8 directly, and B2 by subtraction
- `-Xptxas -v` on find_extrema → decides whether D3 has any real target
- ncu launch stats on descriptor → decides D1's actual block-count transition
- perf LDST replay counters on the CPU descriptor → sizes C2 honestly

CUDA:
1. A1 + B4-memset + B9 — pure deletions, zero risk (B4 rotation deferred
   behind the Tensor ownership blocker)
2. B1 (+ o-bin-9 no-op delete) — biggest verified bit-exact kernel item;
   claim bitwise only for exact CUDA + CPU, geometric for atomic kernels
3. B12 dp4a matcher — biggest frontend win, isolated, skeptic-untouched
4. B3/D7 per-layer counts — triple-confirmed, host needs the starts anyway
5. D1 red[128] + D2 __launch_bounds__ + D10 sweep — one NVRTC-source pass,
   gated on the ncu numbers
6. D4 refl101 closed form + warp-uniform border test
7. D8 + D11 small items alongside whatever kernel they sit in
8. B2/B7 deferred-readback + prefix graph — after the B8 measurement re-budgets
   the gap; expect less than first claimed
9. D5 orientation shfl — exact tier, gated on threads==32
10. D6 multi-stream descriptor groups — last; smallest verified structural win

CPU (single-thread ms; ÷~4 for the 6T wall):
1. C9 — pin the stage-sum inconsistency first; C1 is probe-invisible
2. C1 — downsample udiv→2x, ~4.8 ms ST, identical integer, trivial
3. B1 CPU side — dead border writes (bitwise; smaller zeroing win than first
   noted — 160 readable floats, not 240)
4. C2 — pr staging → 4 explicit zip blocks (store-port win, not forwarding);
   size by replay counters first
5. C3 — vectorised collect, ~3-5 ms after the 0.7 partial-block scaling;
   three contraction traps: c_rot, r_rot, weight
6. C5 + C6 — blur border buffer + V restructure, 6-10 ms combined
7. C4 — upsample H vectorise, 2-4 ms
8. C8 items opportunistically
