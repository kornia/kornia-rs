"""Benchmark strided & 2D NumPy array ingestion in ``Image.from_numpy``.

Measures single-pass strided gather against contiguous-copy baselines across
single-channel (2D / ``[:, :, :1]``) and multi-channel (3ch / 4ch, u8 / f32)
NumPy views, using ``_bench.bench`` with GC disabled and ``RAYON_NUM_THREADS=1``.

Usage::

    pixi run py-build-release
    pixi run python kornia-py/benchmarks/bench_from_numpy_strided.py
    pixi run python kornia-py/benchmarks/bench_from_numpy_strided.py --json /tmp/head.json
    pixi run python kornia-py/benchmarks/bench_from_numpy_strided.py --compare /tmp/main.json /tmp/head.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import tracemalloc
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "1")

import numpy as np

import kornia_rs
from kornia_rs.image import Image

from _bench import BenchResult, bench


def _sha256_file(path_str: str) -> str:
    h = hashlib.sha256()
    with open(path_str, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _git_head(ext_path: str) -> str:
    if ext_path == "<unknown>":
        return "unknown"
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(ext_path).resolve().parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "wheel"


def _build_cases() -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(0)
    gray_1080 = rng.integers(0, 256, (1080, 1920), dtype=np.uint8)
    rgb_1080 = rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    rgba_1080 = rng.integers(0, 256, (1080, 1920, 4), dtype=np.uint8)
    chw_1080 = rng.integers(0, 256, (3, 1080, 1920), dtype=np.uint8)
    rgb_1080_f32 = rng.random((1080, 1920, 3), dtype=np.float32)
    rgb_4k = rng.integers(0, 256, (2160, 3840, 3), dtype=np.uint8)

    return [
        # Single-channel (2D + 3D C==1) cases.
        ("1080p rgb[:,:,:1]", rgb_1080[:, :, :1]),
        ("1080p gray.T", gray_1080.T),
        ("1080p gray[:,::-1]", gray_1080[:, ::-1]),
        ("1080p gray[:,::2]", gray_1080[:, ::2]),
        ("1080p gray[::2,:]", gray_1080[::2, :]),
        # Multi-channel & wider-dtype cases.
        ("1080p rgb[::2,:,:]", rgb_1080[::2, :, :]),
        ("1080p rgb[::-1,:,:]", rgb_1080[::-1, :, :]),
        ("1080p rgb[:,::2,:]", rgb_1080[:, ::2, :]),
        ("1080p rgb[:,:,::-1]", rgb_1080[:, :, ::-1]),
        ("1080p rgba[:,:,::-1]", rgba_1080[:, :, ::-1]),
        ("1080p chw->hwc", chw_1080.transpose(1, 2, 0)),
        ("1080p f32[:,::2,:]", rgb_1080_f32[:, ::2, :]),
        ("4k rgb.transpose(1,0,2)", rgb_4k.transpose(1, 0, 2)),
        ("4k rgb[::2,::2,:]", rgb_4k[::2, ::2, :]),
    ]


def _peak_py_mib(fn) -> float:
    tracemalloc.start()
    try:
        fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak / (1024.0 * 1024.0)


def _provenance() -> dict[str, str]:
    native_mod = getattr(kornia_rs, "kornia_rs", kornia_rs)
    ext_path = getattr(native_mod, "__file__", "<unknown>")
    ext_hash = _sha256_file(ext_path) if ext_path != "<unknown>" and os.path.exists(ext_path) else "n/a"
    return {
        "git_head": _git_head(ext_path),
        "ext_path": ext_path,
        "ext_sha256": ext_hash,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "machine": platform.machine(),
        "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS", "unset"),
    }


def _print_provenance(meta: dict[str, str]) -> None:
    print("Provenance:")
    for key, val in meta.items():
        print(f"  {key}: {val}")


def _compare_json(main_path: Path, head_path: Path) -> None:
    a = json.loads(main_path.read_text())
    b = json.loads(head_path.read_text())
    print("Baseline A:")
    _print_provenance(a["meta"])
    print("Candidate B:")
    _print_provenance(b["meta"])
    a_rows = {r["case"]: r for r in a["results"]}
    print()
    print(
        f"  {'case':<24} {'A min':>9} {'B min':>9} {'speedup':>8} "
        f"{'A p50':>9} {'B p50':>9} {'A pyMiB':>8} {'B pyMiB':>8}"
    )
    for row in b["results"]:
        case = row["case"]
        ref = a_rows.get(case)
        if ref is None:
            continue
        spd = ref["min_ms"] / row["min_ms"]
        print(
            f"  {case:<24} {ref['min_ms']:>9.3f} {row['min_ms']:>9.3f} {spd:>7.2f}x "
            f"{ref['p50_ms']:>9.3f} {row['p50_ms']:>9.3f} "
            f"{ref['py_peak_mib']:>8.2f} {row['py_peak_mib']:>8.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="Write JSON results to this path")
    parser.add_argument(
        "--compare",
        nargs=2,
        type=Path,
        metavar=("BASE_JSON", "HEAD_JSON"),
        help="Compare two JSON runs produced by --json and exit",
    )
    parser.add_argument("--target-seconds", type=float, default=0.5)
    parser.add_argument("--min-iters", type=int, default=20)
    args = parser.parse_args()

    if args.compare is not None:
        _compare_json(args.compare[0], args.compare[1])
        return

    meta = _provenance()
    _print_provenance(meta)

    cases = _build_cases()
    rows: list[dict[str, object]] = []
    print()
    print(
        f"  {'case':<24} {'shape':<18} {'min ms':>9} {'p50 ms':>9} "
        f"{'p95 ms':>9} {'mean ms':>9} {'py MiB':>8} {'n':>6}"
    )
    for label, view in cases:
        res: BenchResult = bench(
            lambda v=view: Image.from_numpy(v, copy=False),
            name=label,
            target_seconds=args.target_seconds,
            min_iters=args.min_iters,
        )
        py_mib = _peak_py_mib(lambda v=view: Image.from_numpy(v, copy=False))
        shape_str = "x".join(str(d) for d in view.shape)
        print(
            f"  {label:<24} {shape_str:<18} {res.min_ms:>9.3f} {res.p50_ms:>9.3f} "
            f"{res.p95_ms:>9.3f} {res.mean_ms:>9.3f} {py_mib:>8.2f} {res.n:>6}"
        )
        rows.append(
            {
                "case": label,
                "shape": list(view.shape),
                "dtype": str(view.dtype),
                "n": res.n,
                "min_ms": res.min_ms,
                "p50_ms": res.p50_ms,
                "p95_ms": res.p95_ms,
                "mean_ms": res.mean_ms,
                "py_peak_mib": py_mib,
            }
        )

    if args.json is not None:
        args.json.write_text(json.dumps({"meta": meta, "results": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
