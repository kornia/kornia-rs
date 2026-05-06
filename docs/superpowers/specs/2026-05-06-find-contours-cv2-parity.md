# Spec: bit-exact cv2 parity for `find_contours`

**Status:** scoping / not started
**Goal:** make `find_contours` produce coordinate-by-coordinate identical output to `cv2.findContours` across all 4 fixtures (pic1-4) × 2 modes (EXTERNAL, LIST) × 2 methods (SIMPLE, NONE), without losing the ~3× perf margin we already have on pic4.
**Non-goal:** re-implement RETR_CCOMP / RETR_TREE parity (out of scope; current users want EXTERNAL).

## What the gap actually looks like (from `analyze_pic4_gap.py`)

Empirical investigation of pic4 EXT, after phase 0 landed:

```
kornia: 846 contours, cv2: 881 contours, gap: 35
  starts only in cv2:    107
  starts only in kornia:  71
  common starts:         775
```

So **~92% of pic4's outer contours have identical start points** between
kornia and cv2. The remaining 178 starts disagree (107 + 71); the net
difference of 35 happens because cv2's disagreements happen to merge into
contour-counts that aren't fully balanced by kornia's disagreements.

Topology histograms of the disagreement sets:
```
cv2-only:    01111000×44, 00111000×26, 00000000×13, 01110000×6, ...
kornia-only: 01111000×18, 01110000×16, 00111000×10, 00000000×8, ...
```

The two lists share **7 of the same 8-conn topology signatures** — meaning
both implementations *find the same contour blob*, but pick a different
start pixel (typically off by 1 row or 1 column). This is the smoking gun
for "marking convention divergence" — when cv2 marks the start of a
multi-row outer as `-nbd`, the next-row scan crosses into "outside"
state at a different column than kornia's `+nbd`-marked equivalent,
producing a shifted re-detection on the row below.

Implication: a targeted patch to the start-marking heuristic CAN'T close
the gap — the disagreement is per-pixel and depends on the trace's full
direction history, which is exactly what cv2's wrapped-direction marking
encodes. Phase 1 (full port of `icvFetchContour`) is the right scope.

## Where we are today (2026-05-06)

`check_correctness.py` results after commit `b4c1444`:

| fixture | EXTERNAL simple | EXTERNAL none | LIST simple | LIST none |
|---------|----------------:|--------------:|------------:|----------:|
| pic1    | ✅ bit-exact    | ✅ bit-exact  | ❌ 20 vs 17 | ❌ 20 vs 17 |
| pic3    | ✅ bit-exact    | ✅ bit-exact  | ❌ 108 vs 121 | ❌ 108 vs 121 |
| pic4    | ❌ 846 vs 881   | ❌ 846 vs 881 | ❌ 1082 vs 931 | ❌ 1082 vs 931 |

(pic2 is 1-vs-1 in EXTERNAL but skipped by the script because LIST output would be ~5723 contours.)

So the gaps are in **two distinct places**, not one:

1. **EXTERNAL on pic4** — under-detection (-35). Documented in `RESULTS_CONTOURS.md`: cv2 marks `±nbd` based on whether the neighbor scan **wrapped** in direction-space; we mark based on left/right pixel state. Fixes 35 contours on pic4.
2. **LIST on every fixture** — both over- AND under-detection depending on the fixture. Distinct from the EXTERNAL bug. Affects how hole borders get re-detected on subsequent rows.

## Root cause: cv2's `icvFetchContour` is internally consistent

```c
// OpenCV contours.cpp:574-582 (paraphrased)
if ((unsigned)(s - 1) < (unsigned)s_end)   // wrapped in direction space
    *i3 = (schar)(nbd | -128);              // mark -nbd (right edge)
else if (*i3 == 1)
    *i3 = nbd;                              // mark +nbd (left edge)
```

cv2's marking, emission, halt-detection, and SIMPLE-mode chain compression are coupled — they form a self-consistent set:
- The marking convention controls which neighbors get walked vs skipped on the **next row's** scan.
- The halt rule (`i4==i0 && i3==i1`) depends on the marking having created the right pattern when re-entering the start.
- SIMPLE-mode `dir_in != dir_out` tests assume the direction sequence cv2 produces.

**Last attempt (in commit history):** literal port of cv2's main loop broke 9 tests including `hollow_square` and the nested-image tests. Reverted. Lesson: replace the four pieces atomically, not one at a time.

## Plan: 4 phases, each individually testable

### Phase 0 — capture baseline outputs (1 day)

Before changing any code, dump current kornia output (all 4 fixtures × 2 modes × 2 methods) to JSON snapshots committed under `tests/snapshots/`. Then dump cv2 output to the same format. Diff scripts compare both at every phase. Goal: never blind-flying-while-refactoring.

**Files:**
- Create: `crates/kornia-imgproc/tests/snapshots/cv2_pic{1,3,4}_{ext,list}_{simple,none}.json`
- Create: `crates/kornia-imgproc/tests/snapshots/kornia_pic{1,3,4}_{ext,list}_{simple,none}.json`
- Create: `crates/kornia-imgproc/examples/dump_snapshots.sh` (one-shot regen)

### Phase 1 — port cv2's marking convention (3-5 days)

**Files:**
- Modify: `crates/kornia-imgproc/src/contours.rs` — `trace_border` (the inner loop that decides `nbd` vs `-nbd` on the start pixel).

Replace our left/right-state marking with cv2's wrapped-direction marking. This breaks all four pieces simultaneously, so we need to land all of phase 1-3 in one commit (or feature-flag both implementations, switch in a separate commit).

**Strategy:** keep a `trace_border_legacy` (current impl) under a `#[cfg(any())]` until the new path passes all snapshots. Toggle by build feature — never two paths in the same build.

### Phase 2 — port cv2's halt detection + emission timing (2-3 days)

The `i4==i0 && i3==i1` halt is subtly different from our `i2_idx == start_idx && i3_idx == first_nb_idx`. cv2 emits `i3` (the *next* pixel) before halt-checking; we emit `i2` (the *current* pixel). This shifts the SIMPLE-mode point set by one step.

### Phase 3 — port cv2's SIMPLE-mode compression (1-2 days)

cv2's chain compression accumulates direction codes per step and emits a corner only when the direction changes for ≥2 consecutive steps. Ours emits on every direction change. The sequence depends on phase 2's emission timing.

### Phase 4 — fix LIST mode hole detection (2-3 days)

Separate from the EXTERNAL gap. Current LIST output over/under-counts holes:
- pic1: 20 vs 17 (we over-detect 3 — likely re-detecting a hole on a later row)
- pic3: 108 vs 121 (we under-detect 13 — likely halting a hole trace before it loops back)
- pic4: 1082 vs 931 (compounds with EXTERNAL fix; recheck after phases 1-3)

**Investigation required first:** dump pic1 LIST contours and find the 3 we have but cv2 doesn't. Likely a `lnbd` reset bug or a hole-start condition (`pixel >= 1 && right == 0 && !is_outer`) that fires for cv2-skipped cases.

## Testing strategy

After every phase, all of these must pass:

- `cargo test --release -p kornia-imgproc --lib` — 182 unit tests (existing)
- `cargo test --release -p kornia-imgproc --test contours_real_images` — 5 integration tests (existing — tighten thresholds at end of phase 4)
- `cargo run --release --example synth_ext_count -p kornia-imgproc` — 14 synthetic shapes (existing)
- `python3 crates/kornia-imgproc/examples/check_correctness.py` — coord-by-coord parity (currently 4/12; target 12/12 by end)

## Risk & mitigation

- **Perf regression** — cv2's algorithm has more per-pixel work than ours (extra direction tracking, deferred emission). Bench at end of every phase. If pic4 EXTERNAL drops below 1.5× cv2, revisit. The EXTERNAL skip from `cvFindNextContour:1131` is independent of `icvFetchContour` and stays.
- **Hole infinite loops** — past attempt broke nested-image hole detection. Snapshot tests catch this immediately; legacy fallback build feature lets us bisect.
- **LIST scope creep** — phases 1-3 fix EXTERNAL only. Phase 4 may surface architectural issues that need their own design pass; if so, ship phases 1-3 separately.

## Estimated effort

10-13 working days end-to-end, broken into 4 commits (one per phase). Phase 1-3 must land together; phase 4 can be deferred if it blows up.

## Decision needed

Two options for kicking off:

1. **Start phase 0 now** — produce the snapshot harness this session, validates the test infrastructure works before any algo changes.
2. **Wait** — the current state ships pic1/3 EXT bit-exact + pic4 EXT 91% within tolerance + ~3× cv2 speed. If users only need EXTERNAL counts (the dominant cv2 use case), we may not need full parity.

Recommend option 1: phase 0 has zero risk and gives us the diff harness regardless.
