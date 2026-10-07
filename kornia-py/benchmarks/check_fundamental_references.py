#!/usr/bin/env python3
"""Compare all minimal F7 solutions against current Kornia and pydegensac source.

Use build_pydegensac_minimal.py to compile the source-exact C reference and
optimization/solver_probe.rs for the Rust probe. Run in an environment with
NumPy, Torch and the local Kornia dependencies. Real seven-match samples are
supplied as an N x 7 x 4 .npy array so no HDF5 dependency is needed here.
"""
import argparse
import hashlib
from pathlib import Path
import ctypes
import json
import subprocess
import sys

import numpy as np
import torch
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--rust-probe", type=Path, required=True)
parser.add_argument("--pydegensac-library", type=Path, required=True)
parser.add_argument("--kornia-root", type=Path, required=True)
parser.add_argument("--samples-npy", type=Path, required=True)
parser.add_argument("--json", type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.kornia_root.resolve()))
from kornia.geometry.epipolar.fundamental import find_fundamental
LIB = ctypes.CDLL(str(args.pydegensac_library.resolve()))
D = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
I = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
LIB.nullspace.argtypes = [D, D, ctypes.c_int, I]
LIB.nullspace.restype = ctypes.c_int
LIB.slcm.argtypes = [D, D, D]
LIB.rroots3.argtypes = [D, D]
LIB.rroots3.restype = ctypes.c_int
PROBE = str(args.rust_probe.resolve())
LIB.pyd7_once.argtypes = [D, ctypes.c_int, D]
LIB.pyd7_once.restype = ctypes.c_int


def pydegen_minimal(pairs):
    x1, x2 = pairs[:, :2], pairs[:, 2:]
    a = np.zeros((9, 9), dtype=np.float64)
    a[:7] = np.c_[
        x2[:, 0] * x1[:, 0], x2[:, 0] * x1[:, 1], x2[:, 0],
        x2[:, 1] * x1[:, 0], x2[:, 1] * x1[:, 1], x2[:, 1],
        x1[:, 0], x1[:, 1], np.ones(7),
    ]
    basis = np.empty((9, 9), dtype=np.float64)
    if LIB.nullspace(a, basis, 9, np.empty(18, dtype=np.int32)) != 2:
        return []
    p, roots = np.empty(4), np.empty(3)
    LIB.slcm(basis[0], basis[1], p)  # mutates basis[1], matching exp_ranF.c
    n = LIB.rroots3(p, roots)
    return [basis[0] * r + basis[1] * (1.0 - r) for r in roots[:n]]


def pydegen_normalized(pairs):
    models = np.empty((3, 9), dtype=np.float64)
    count = LIB.pyd7_once(np.ascontiguousarray(pairs), 1, models)
    return list(models[:count])


def rust_minimal(samples):
    text = "".join(json.dumps(s.tolist()) + "\n" for s in samples)
    got = subprocess.check_output([PROBE], input=text.encode())
    # Mat3F64's Into<[f64; 9]> preserves its column-major storage.
    return [[np.asarray(f).reshape(3, 3).T for f in json.loads(line)] for line in got.decode().splitlines()]


def kornia_minimal(pairs):
    pts = torch.as_tensor(pairs, dtype=torch.float64)[None]
    fs = find_fundamental(pts[..., :2], pts[..., 2:], method="7POINT")[0].detach().numpy()
    return [f.reshape(-1) for f in fs if np.linalg.norm(f) > 0]


def normal(f):
    f = np.asarray(f).reshape(3, 3)
    return f / np.linalg.norm(f)


def distance(a, b):
    a, b = normal(a), normal(b)
    return min(np.linalg.norm(a - b), np.linalg.norm(a + b))


def errors(fs, pairs):
    x1 = np.c_[pairs[:, :2], np.ones(7)]
    x2 = np.c_[pairs[:, 2:], np.ones(7)]
    return [(np.max(np.abs(np.einsum("ni,ij,nj->n", x2, np.asarray(f).reshape(3, 3), x1))), abs(np.linalg.det(np.asarray(f).reshape(3, 3)))) for f in fs]


def make_samples(n=1000):
    rng = np.random.default_rng(12345)
    result = []
    for _ in range(n):
        # Generic non-planar two-view correspondences, then random independent pixel similarities.
        X = np.c_[rng.normal(size=(7, 2)), rng.uniform(2.0, 8.0, 7)]
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
        angle = rng.uniform(-0.7, 0.7)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
        t = rng.normal(size=3)
        Y = X @ R.T + t
        x1, x2 = X[:, :2] / X[:, 2:], Y[:, :2] / Y[:, 2:]
        for x, scale, offset in ((x1, 10 ** rng.uniform(-3, 4), rng.normal(size=2) * 100), (x2, 10 ** rng.uniform(-3, 4), rng.normal(size=2) * 100)):
            x *= scale; x += offset
        result.append(np.c_[x1, x2])
    return result


def check(samples):
    rust = rust_minimal(samples)
    bad = []
    summary = {"rust_vs_pydegen_normalized_miss": 0, "pydegen_normalized_vs_rust_miss": 0, "kornia_vs_rust_miss": 0, "rust_nonfinite": 0, "kornia_nonfinite": 0, "pydegen_nonfinite": 0, "pydegen_normalized_nonfinite": 0, "total": len(samples), "rust_empty": 0, "pydegen_empty": 0, "kornia_empty": 0, "rust_bad_residual": 0, "rust_bad_rank": 0, "rust_vs_kornia_miss": 0, "rust_vs_pydegen_miss": 0}
    for i, (s, rf) in enumerate(zip(samples, rust)):
        pf, kf, nf = pydegen_minimal(s), kornia_minimal(s), pydegen_normalized(s)
        for key, fs in (("rust", rf), ("pydegen", pf), ("kornia", kf), ("pydegen_normalized", nf)):
            if not fs and key != "pydegen_normalized": summary[f"{key}_empty"] += 1
            summary[f"{key}_nonfinite"] += sum(not np.isfinite(f).all() for f in fs)
        for e, d in errors(rf, s):
            if e > 1e-6: summary["rust_bad_residual"] += 1
            if d > 1e-6: summary["rust_bad_rank"] += 1
        # Different pivots/parameterizations can legitimately make a root numerically unstable.
        for label, fs in (("kornia", kf), ("pydegen", pf), ("pydegen_normalized", nf)):
            for f in fs:
                if not rf or min(distance(f, g) for g in rf) > 1e-6:
                    summary[f"rust_vs_{label}_miss"] += 1
                    if len(bad) < 5: bad.append((i, label, len(rf), len(fs), min(distance(f,g) for g in rf)))
        for label, fs in (("kornia", kf), ("pydegen_normalized", nf)):
            for f in rf:
                if not fs or min(distance(f, g) for g in fs) > 1e-6:
                    summary[f"{label}_vs_rust_miss"] += 1
    return {"summary": summary, "examples": bad}


def singular_endpoint_fixture():
    # Kornia's `test_singular_pencil_endpoints`: both initial LU basis matrices
    # can be singular although the pencil has isolated valid rank-two roots.
    x1 = np.array([[0, -2], [-2, 1], [0, 1], [0, 0], [-1, 2], [-1, 1], [2, -1]], dtype=float)
    x2 = np.array([[1, 2], [2, 2], [0, -2], [-1, 2], [0, -1], [-1, -2], [2, 2]], dtype=float)
    return np.c_[x1, x2][None]


def main():
    result = {
        "random_1000": check(make_samples()),
        "real_1000": check(list(np.load(args.samples_npy))),
        "singular_endpoint": check(singular_endpoint_fixture()),
    }
    result["metadata"] = {
        "torch": torch.__version__, "numpy": np.__version__, "kornia_root": str(args.kornia_root),
        "rust_probe_sha256": hashlib.sha256(args.rust_probe.read_bytes()).hexdigest(),
        "pydegensac_library_sha256": hashlib.sha256(args.pydegensac_library.read_bytes()).hexdigest(),
        "samples_sha256": hashlib.sha256(args.samples_npy.read_bytes()).hexdigest(),
        "kornia_solver_sha256": hashlib.sha256((args.kornia_root / "kornia/geometry/epipolar/fundamental.py").read_bytes()).hexdigest(),
        "comparison": "bidirectional projective model distances <= 1e-6; invalid zero-padded Kornia slots excluded",
        "random_seed": 12345,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    with args.json.open("w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result, indent=2))
    for name in ("random_1000", "real_1000", "singular_endpoint"):
        summary = result[name]["summary"]
        required_zero = ("rust_empty", "rust_nonfinite", "rust_bad_residual", "rust_bad_rank",
                         "rust_vs_kornia_miss", "kornia_vs_rust_miss", "kornia_nonfinite",
                         "pydegen_normalized_nonfinite")
        if name in ("real_1000", "singular_endpoint") and summary["rust_vs_pydegen_miss"]:
            raise SystemExit(f"Raw pydegensac reference validation failed: {name}")
        if name == "random_1000" and (summary["rust_vs_pydegen_normalized_miss"] or summary["pydegen_normalized_vs_rust_miss"]):
            raise SystemExit("Normalized synthetic reference validation failed")
        if any(summary[key] for key in required_zero):
            raise SystemExit(f"Reference validation failed: {name}")


if __name__ == "__main__":
    main()
