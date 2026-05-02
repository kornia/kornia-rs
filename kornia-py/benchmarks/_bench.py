"""Reusable Python benchmarking helpers for kornia-py.

Designed to give honest sub-millisecond numbers — the kind of timings
where naive ``time.perf_counter()`` loops are dominated by GC pauses,
scheduler preemption, and Python loop overhead.

Methodology:
  - Warmup: at least ``min_warmup_iters`` calls AND ``warmup_seconds`` of
    elapsed time, whichever takes longer. Slow ops (>200ms) get N priming
    runs even if the seconds budget is tiny.
  - Calibration: median of ``min_warmup_iters`` ns-resolution samples
    estimates the per-call cost; iteration count is then sized to
    ``target_seconds`` of total work, clamped to ``[min_iters, max_iters]``.
  - During the timed loop: GC disabled, every call timed individually
    with ``time.perf_counter_ns()`` so a single GC-pause outlier doesn't
    pollute the reported "typical" time.
  - Reports min / p50 / p95 / mean / stdev. **For sub-millisecond ops
    always read the min**; mean is biased high by scheduler noise that
    has nothing to do with the kernel.

Note: the bench loop itself adds ~80-150ns of Python overhead per call
on CPython 3.12. Don't trust ``min_ms`` for ops faster than ~200ns —
the harness floor dominates.

Typical use::

    from _bench import bench, compare, print_table
    results = compare({
        "PIL":    lambda: pil_img.convert("L"),
        "cv2":    lambda: cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY),
        "kornia": lambda: img.to_grayscale(),
    }, target_seconds=1.0)
    print_table("RGB->L 1080p", results)
"""

from __future__ import annotations

import gc
import statistics
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class BenchResult:
    """Distribution of per-call timings, in milliseconds."""

    name: str
    n: int
    min_ms: float
    p50_ms: float
    p95_ms: float
    mean_ms: float
    stdev_ms: float
    total_seconds: float
    raw_ms: list[float] = field(repr=False, default_factory=list)


def _calibrate_one_call_ns(fn: Callable[[], object], samples: int = 5) -> int:
    """Median-of-N ns-resolution per-call cost. One sample lands on a
    scheduler tick often enough to mis-size the iteration count by 10×;
    median of 5 is the cheapest-honest estimate."""
    pc = time.perf_counter_ns
    times = []
    for _ in range(samples):
        t = pc()
        fn()
        times.append(pc() - t)
    return max(int(statistics.median(times)), 1)


def bench(
    fn: Callable[[], object],
    *,
    name: str = "",
    target_seconds: float = 1.0,
    min_iters: int = 100,
    max_iters: int = 100_000,
    warmup_seconds: float = 0.2,
    min_warmup_iters: int = 3,
    keep_raw: bool = False,
) -> BenchResult:
    """Best-of-N benchmark with GC disabled and per-call timings."""
    # Warmup: at least N calls AND ``warmup_seconds`` elapsed.
    t_end = time.perf_counter() + warmup_seconds
    warmups = 0
    while warmups < min_warmup_iters or time.perf_counter() < t_end:
        fn()
        warmups += 1

    one_call_ns = _calibrate_one_call_ns(fn)
    iters = int(target_seconds * 1e9 / one_call_ns)
    iters = max(min_iters, min(max_iters, iters))

    times_ns: list[int] = []
    append = times_ns.append
    pc = time.perf_counter_ns

    was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        for _ in range(iters):
            t = pc()
            fn()
            append(pc() - t)
    finally:
        if was_enabled:
            gc.enable()

    times_ms = sorted(t / 1e6 for t in times_ns)
    n = len(times_ms)
    return BenchResult(
        name=name,
        n=n,
        min_ms=times_ms[0],
        p50_ms=statistics.median(times_ms),
        p95_ms=times_ms[min(int(n * 0.95), n - 1)],
        mean_ms=statistics.mean(times_ms),
        stdev_ms=statistics.pstdev(times_ms) if n > 1 else 0.0,
        total_seconds=sum(times_ns) / 1e9,
        raw_ms=times_ms if keep_raw else [],
    )


def compare(
    fns: dict[str, Callable[[], object]],
    **kwargs,
) -> dict[str, BenchResult]:
    """Run :func:`bench` on each ``(name, fn)`` pair, preserving dict order."""
    return {name: bench(fn, name=name, **kwargs) for name, fn in fns.items()}


def compat_print(
    name: str, result: BenchResult, *, label_width: int = 40, quiet: bool = False
) -> float:
    """Format a single ``bench`` result on one line and return ``min_ms``.

    Used by the per-script bench shims that pre-date this module — those
    files keep their ``def bench(name, fn, n=, warmup=)`` shape and
    delegate to ``bench()`` here, then ``compat_print`` prints the result
    and returns the min for downstream comparisons.
    """
    if not quiet:
        print(f"  {name:<{label_width}s} {result.min_ms:8.3f} ms (min, n={result.n})")
    return result.min_ms


def print_table(
    label: str,
    results: dict[str, BenchResult] | Iterable[BenchResult],
    *,
    fastest_marker: bool = True,
    file=None,
) -> None:
    """Print a one-line-per-runner table. Marks the fastest min with `*`."""
    if isinstance(results, dict):
        rows = list(results.values())
    else:
        rows = list(results)
    if not rows:
        return

    fastest_min = min(r.min_ms for r in rows) if fastest_marker else float("inf")
    print(f"\n{label}", file=file)
    print(
        f"  {'runner':<14} {'min ms':>9} {'p50 ms':>9} {'p95 ms':>9} "
        f"{'mean ms':>9} {'n':>7}",
        file=file,
    )
    for r in rows:
        marker = " *" if abs(r.min_ms - fastest_min) < 1e-9 else "  "
        print(
            f" {marker} {r.name:<13} {r.min_ms:>9.3f} {r.p50_ms:>9.3f} "
            f"{r.p95_ms:>9.3f} {r.mean_ms:>9.3f} {r.n:>7}",
            file=file,
        )


def speedup_vs(results: dict[str, BenchResult], baseline: str) -> dict[str, float]:
    """Return ``baseline_min / runner_min`` per runner. >1 means faster
    than baseline."""
    if baseline not in results:
        raise KeyError(f"baseline {baseline!r} not in results")
    base = results[baseline].min_ms
    return {name: base / r.min_ms for name, r in results.items()}


__all__ = [
    "BenchResult",
    "bench",
    "compare",
    "compat_print",
    "print_table",
    "speedup_vs",
]
