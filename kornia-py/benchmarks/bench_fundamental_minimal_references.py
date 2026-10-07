#!/usr/bin/env python3
"""Time one- and three-root F7 samples in Rust, pydegensac C, and Kornia CPU.

Native fits run inside C/Rust loops, excluding Python/subprocess overhead.
Kornia uses prepared float64 CPU tensors and includes PyTorch dispatch.
"""
import argparse
import ctypes
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rust-probe", type=Path, required=True)
    parser.add_argument("--pydegensac-library", type=Path, required=True)
    parser.add_argument("--samples-npy", type=Path, required=True)
    parser.add_argument("--kornia-root", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=200000)
    parser.add_argument("--rounds", type=int, default=7)
    args = parser.parse_args()
    samples = np.load(args.samples_npy)
    serialized = "".join(json.dumps(sample.tolist()) + "\n" for sample in samples)
    models = [json.loads(line) for line in subprocess.check_output([str(args.rust_probe)], input=serialized.encode()).decode().splitlines()]
    lib = ctypes.CDLL(str(args.pydegensac_library))
    array = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    lib.pyd7_once.argtypes = [array, ctypes.c_int, array]
    lib.pyd7_once.restype = ctypes.c_int
    lib.pyd7_bench.argtypes = [array, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
    lib.pyd7_bench.restype = ctypes.c_int
    sys.path.insert(0, str(args.kornia_root.resolve()))
    import torch
    from kornia.geometry.epipolar.fundamental import find_fundamental
    torch.set_num_threads(1)
    rows = []
    for roots in (1, 3):
        index = next(i for i, fs in enumerate(models) if len(fs) == roots
                     and lib.pyd7_once(np.ascontiguousarray(samples[i]), 0, np.empty((3, 9))) == roots
                     and lib.pyd7_once(np.ascontiguousarray(samples[i]), 1, np.empty((3, 9))) == roots)
        sample = np.ascontiguousarray(samples[index])
        line = (json.dumps(sample.tolist()) + "\n").encode()
        tensors = torch.as_tensor(sample, dtype=torch.float64)[None]
        def kornia_call():
            return find_fundamental(tensors[..., :2], tensors[..., 2:], method="7POINT")
        with torch.no_grad():
            for _ in range(20):
                kornia_call()
        subprocess.check_output([str(args.rust_probe), "10000"], input=line)
        timed = {"rust_fit": [], "rust_public": [], "pydegensac_raw": [], "pydegensac_hartley": [], "kornia_single_cpu": []}
        for repetition in range(args.rounds):
            order = ("rust", "raw", "normalized")
            order = order[repetition % 3:] + order[:repetition % 3]
            for backend in order:
                if backend == "rust":
                    result = json.loads(subprocess.check_output([str(args.rust_probe), str(args.iterations)], input=line))
                    assert result["count"] == roots
                    timed["rust_fit"].append(result["fit_ns"])
                    timed["rust_public"].append(result["public_ns"])
                else:
                    checksum = ctypes.c_double()
                    start = time.perf_counter_ns()
                    count = lib.pyd7_bench(sample, args.iterations, int(backend == "normalized"), ctypes.byref(checksum))
                    elapsed = (time.perf_counter_ns() - start) / args.iterations
                    assert count == roots and np.isfinite(checksum.value)
                    timed["pydegensac_hartley" if backend == "normalized" else "pydegensac_raw"].append(elapsed)
            with torch.no_grad():
                start = time.perf_counter_ns()
                for _ in range(100):
                    kornia_call()
                timed["kornia_single_cpu"].append((time.perf_counter_ns() - start) / 100)
        rows.append({"roots": roots, "sample_index": index, "sample": sample.tolist(),
                     "median_ns": {name: statistics.median(times) for name, times in timed.items()}, "timings_ns": timed})
    result = {"metadata": {"platform": platform.platform(), "torch": torch.__version__,
              "iterations": args.iterations, "rounds": args.rounds, "torch_threads": 1,
              "rust_probe_sha256": hashlib.sha256(args.rust_probe.read_bytes()).hexdigest(),
              "pydegensac_library_sha256": hashlib.sha256(args.pydegensac_library.read_bytes()).hexdigest(),
              "samples_sha256": hashlib.sha256(args.samples_npy.read_bytes()).hexdigest(),
              "units": "ns per fit; native loops exclude FFI/process overhead; Kornia includes PyTorch dispatch"}, "results": rows}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
