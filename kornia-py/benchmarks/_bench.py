"""Reusable Python benchmarking helpers for kornia-py.

Designed to give honest sub-millisecond numbers — the kind of timings where
naive ``time.perf_counter()`` loops are dominated by GC pauses, scheduler
preemption, and Python loop overhead. Used by the codec / Image-API
benchmarks under ``kornia-py/benchmarks/``.

Why not ``timeit`` / ``pytest-benchmark``?
  - ``timeit`` reports a single mean over an opaque number of iterations
    and can't pause GC mid-loop without extra plumbing.
  - ``pytest-benchmark`` is a reasonable choice for unit tests but adds a
    plugin dep and pulls bench numbers into the unit-test report. We want
    a self-contained module that scripts under ``benchmarks/`` can import
    without a pytest harness.

Methodology:
  - ``warmup_seconds`` of throwaway calls before timing starts (page-fault
    in caches, prime the JIT-y bits of pyo3 etc.).
  - During the timed loop: GC disabled, every call timed individually with
    ``time.perf_counter_ns()`` (no division-by-N at the end so a single
    GC-pause outlier doesn't pollute the reported "typical" time).
  - Auto-tunes the iteration count to fit ``target_seconds`` of total work
    with a ``min_iters`` floor for sub-microsecond ops.
  - Reports min / p50 / p95 / mean / stdev — not just mean. **For
    sub-millisecond ops always read the min**; mean is biased high by
    scheduler noise that has nothing to do with the kernel.

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
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable


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

    @property
    def best(self) -> float:
        """Convenience alias for ``min_ms`` — the right number to report
        for a sub-millisecond op (least-jitter call)."""
        return self.min_ms


def bench(
    fn: Callable[[], object],
    *,
    name: str = "",
    target_seconds: float = 1.0,
    min_iters: int = 100,
    max_iters: int = 100_000,
    warmup_seconds: float = 0.2,
    keep_raw: bool = False,
) -> BenchResult:
    """Best-of-N benchmark with GC disabled and per-call timings.

    Args:
        fn: zero-arg callable to time. The return value is discarded.
        name: optional label included in ``BenchResult.name``.
        target_seconds: budget for the timed loop. Iteration count is
            auto-tuned so total time approaches this. The loop will not run
            fewer than ``min_iters`` or more than ``max_iters`` calls.
        min_iters: floor on iteration count. Useful when calls are
            sub-microsecond and would otherwise need millions of reps.
        max_iters: ceiling on iteration count. Useful when calls take >1s
            and we don't want a 10-minute bench.
        warmup_seconds: untimed warmup before the bench loop starts.
        keep_raw: if True, ``BenchResult.raw_ms`` carries the full sample
            vector for downstream histograms / export.

    Returns:
        A :class:`BenchResult` with the timing distribution.
    """
    # Warmup: untimed runs to settle caches + lazy init.
    t_end = time.perf_counter() + warmup_seconds
    while time.perf_counter() < t_end:
        fn()

    # Calibrate: how many iterations approximate ``target_seconds``?
    t0 = time.perf_counter()
    fn()
    one_call = max(time.perf_counter() - t0, 1e-9)
    iters = int(target_seconds / one_call)
    iters = max(min_iters, min(max_iters, iters))

    times_ns: list[int] = [0] * iters

    was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        # Tight per-call timing loop. perf_counter_ns avoids float
        # arithmetic in the hot path.
        pc = time.perf_counter_ns
        for i in range(iters):
            t = pc()
            fn()
            times_ns[i] = pc() - t
    finally:
        if was_enabled:
            gc.enable()

    times_ms = [t / 1e6 for t in times_ns]
    times_ms.sort()
    n = len(times_ms)
    return BenchResult(
        name=name,
        n=n,
        min_ms=times_ms[0],
        p50_ms=times_ms[n // 2],
        p95_ms=times_ms[min(int(n * 0.95), n - 1)],
        mean_ms=statistics.mean(times_ms),
        stdev_ms=statistics.pstdev(times_ms) if n > 1 else 0.0,
        total_seconds=sum(times_ns) / 1e9,
        raw_ms=times_ms if keep_raw else [],
    )


def compare(
    fns: dict[str, Callable[[], object]],
    *,
    target_seconds: float = 1.0,
    min_iters: int = 100,
    max_iters: int = 100_000,
    warmup_seconds: float = 0.2,
) -> dict[str, BenchResult]:
    """Run :func:`bench` on each ``(name, fn)`` pair, returning the results
    keyed by name. Order of the input ``dict`` is preserved.
    """
    return {
        name: bench(
            fn,
            name=name,
            target_seconds=target_seconds,
            min_iters=min_iters,
            max_iters=max_iters,
            warmup_seconds=warmup_seconds,
        )
        for name, fn in fns.items()
    }


def print_table(
    label: str,
    results: dict[str, BenchResult] | Iterable[BenchResult],
    *,
    fastest_marker: bool = True,
    file=None,
) -> None:
    """Print a one-line-per-runner table. Marks the fastest min with `*`.

    Pass either a ``dict`` from :func:`compare` or any iterable of
    ``BenchResult``.
    """
    if isinstance(results, dict):
        rows = list(results.values())
    else:
        rows = list(results)
    if not rows:
        return
    if file is None:
        file = sys.stdout

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
    """Return ``baseline_min / runner_min`` for each runner. >1 means
    faster than baseline; <1 means slower."""
    if baseline not in results:
        raise KeyError(f"baseline {baseline!r} not in results")
    base = results[baseline].min_ms
    return {name: base / r.min_ms for name, r in results.items()}


__all__ = ["BenchResult", "bench", "compare", "print_table", "speedup_vs"]
