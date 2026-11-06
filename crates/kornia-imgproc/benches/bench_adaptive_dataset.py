"""Benchmark the local Rust adaptive mean implementation against OpenCV CPU.

Build first: cargo build --release -p kornia-imgproc --example bench_adaptive_dataset
Then run this script with --images pointing to extracted DIBCO test images.
Neither image decoding, process startup nor correctness checks are timed.
"""

import argparse
from contextlib import ExitStack
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile
import time

import cv2
import numpy as np


WINDOWS = [3, 5, 15, 31, 51, 67, 101]
WARMUP, ROUNDS, ITERATIONS, OFFSET = 10, 5, 200, 10
WARMUP_MIN_SECONDS = 0.25


def paired(binary, opencv_binary, image, directory, index):
    """Collect alternating native/reference rounds and masks for one decoded image."""
    source = directory / "source.raw"
    image.tofile(source)
    height, width = image.shape
    destination = np.empty_like(image)
    native, reference = {}, {}
    with ExitStack() as stack:
        workers = []
        def launch(executable, masks):
            """Start a persistent worker whose input and output storage is reused."""
            worker = stack.enter_context(subprocess.Popen(
                [str(executable), str(source), str(width), str(height), str(masks), "--server"],
                text=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=1,
                env={**os.environ, "RAYON_NUM_THREADS": "1"},
            ))
            workers.append(worker)
            def request(command):
                """Send one command and fail if the worker exits before replying."""
                worker.stdin.write(command + "\n")
                worker.stdin.flush()
                line = worker.stdout.readline()
                if not line:
                    raise RuntimeError(f"Benchmark worker {executable} exited unexpectedly")
                return line.strip()
            return request
        request = launch(binary, directory / "rust")
        cv_request = launch(opencv_binary, directory / "opencv") if opencv_binary else None
        try:
            for window in WINDOWS:
                def operation():
                    """Apply the Python OpenCV reference into the preallocated destination."""
                    cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY, window, OFFSET, dst=destination)
                request(f"W {window}")
                if cv_request:
                    cv_request(f"W {window}")
                else:
                    start = time.perf_counter()
                    for _ in range(WARMUP):
                        operation()
                    while time.perf_counter() - start < WARMUP_MIN_SECONDS:
                        operation()
                def cv_round():
                    """Time native C++ calls when supplied, otherwise Python binding calls."""
                    if cv_request:
                        return float(cv_request(f"R {window} {ITERATIONS}"))
                    start = time.perf_counter_ns()
                    for _ in range(ITERATIONS):
                        operation()
                    return (time.perf_counter_ns() - start) / ITERATIONS / 1e6
                ks, cs = [], []
                for round_index in range(ROUNDS):
                    if (index + round_index) % 2:
                        cs.append(cv_round())
                        ks.append(float(request(f"R {window} {ITERATIONS}")))
                    else:
                        ks.append(float(request(f"R {window} {ITERATIONS}")))
                        cs.append(cv_round())
                request(f"M {window}")
                mask = np.fromfile(directory / "rust" / f"{window}.raw", dtype=np.uint8).reshape(image.shape)
                if cv_request:
                    cv_request(f"M {window}")
                    destination = np.fromfile(directory / "opencv" / f"{window}.raw", dtype=np.uint8).reshape(image.shape)
                native[window] = (ks, mask)
                reference[window] = (cs, destination.copy())
        finally:
            for worker in workers:
                worker.stdin.close()
            for worker in workers:
                if worker.wait() != 0:
                    raise RuntimeError("Benchmark worker failed")
    return native, reference


def main():
    """Decode the dataset and save median/best-of-N times, masks counts and provenance."""
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rust-binary", type=Path,
                        default=root / "target/release/examples/bench_adaptive_dataset")
    parser.add_argument("--rust-source", type=Path,
                        default=root / "crates/kornia-imgproc/src/threshold.rs")
    parser.add_argument("--opencv-binary", type=Path,
                        help="Native C++ OpenCV worker; otherwise time the Python binding")
    args = parser.parse_args()
    binary = args.rust_binary.resolve(strict=True)
    opencv_binary = args.opencv_binary.resolve(strict=True) if args.opencv_binary else None
    extensions = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}
    paths = sorted(p for p in args.images.rglob("*") if p.suffix.lower() in extensions)
    if not paths:
        raise ValueError("No input images found")
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    args.output.mkdir(parents=True, exist_ok=True)
    rows, inputs = [], []
    for index, path in enumerate(paths):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not decode {path}")
        image = np.ascontiguousarray(image)
        name = path.relative_to(args.images).as_posix()
        print(f"[{index + 1}/{len(paths)}] {name} {image.shape[1]}x{image.shape[0]}", flush=True)
        inputs.append({"image": name, "width": image.shape[1], "height": image.shape[0],
                       "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "grayscale_sha256": hashlib.sha256(image.tobytes()).hexdigest()})
        with tempfile.TemporaryDirectory(prefix="adaptive-bench-") as scratch:
            native, cv = paired(binary, opencv_binary, image, Path(scratch), index)
            for window in WINDOWS:
                ks, kmask = native[window]
                cs, cmask = cv[window]
                mismatches = int(np.count_nonzero(kmask != cmask))
                k_ms, c_ms = statistics.median(ks), statistics.median(cs)
                k_best, c_best = min(ks), min(cs)
                rows.append({"image": name, "width": image.shape[1], "height": image.shape[0],
                             "window": window, "offset": OFFSET, "kornia_ms": k_ms,
                             "opencv_ms": c_ms, "opencv_over_kornia": c_ms / k_ms,
                             "kornia_best_ms": k_best, "opencv_best_ms": c_best,
                             "best_opencv_over_kornia": c_best / k_best,
                             "mismatch_pixels": mismatches,
                             "kornia_rounds_ms": ks, "opencv_rounds_ms": cs})
                print(f"  {window:3} median K={k_ms:.4f}ms CV={c_ms:.4f}ms "
                      f"best-of-{ROUNDS} K={k_best:.4f}ms CV={c_best:.4f}ms "
                      f"mismatches={mismatches}", flush=True)
    fields = [key for key in rows[0] if not key.endswith("rounds_ms")]
    with (args.output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "platform": platform.platform(), "python": platform.python_version(),
        "opencv": cv2.__version__, "opencv_threads_requested": 1,
        "opencv_threads": cv2.getNumThreads(),
        "opencl_enabled": cv2.ocl.useOpenCL(), "warmup_min_calls": WARMUP,
        "warmup_min_seconds": WARMUP_MIN_SECONDS,
        "rounds": ROUNDS, "iterations_per_round": ITERATIONS, "paired_round_order": True,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=root, text=True),
        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "rust_binary": str(binary), "rust_binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "threshold_source": str(args.rust_source.resolve()),
        "threshold_source_sha256": hashlib.sha256(args.rust_source.read_bytes()).hexdigest(),
        "inputs": inputs,
        "reference_call_boundary": "native-cpp" if opencv_binary else "python-binding",
        "rust_scratch": "AdaptiveThresholdMean owns scratch reused across timed calls",
        "rust_worker_source_sha256": hashlib.sha256((root / "crates/kornia-imgproc/examples/bench_adaptive_dataset.rs").read_bytes()).hexdigest(),
        "runner_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    build_info = cv2.getBuildInformation()
    if opencv_binary:
        metadata.update({
            "opencv_binary": str(opencv_binary),
            "opencv_binary_sha256": hashlib.sha256(opencv_binary.read_bytes()).hexdigest(),
            "opencv": subprocess.check_output([str(opencv_binary), "--version"], text=True).strip(),
            "opencv_threads": int(subprocess.check_output([str(opencv_binary), "--threads"], text=True)),
            "decoder_opencv": cv2.__version__,
            "opencv_worker_source_sha256": hashlib.sha256((Path(__file__).parent / "bench_adaptive_opencv.cpp").read_bytes()).hexdigest(),
        })
        build_info = subprocess.check_output([str(opencv_binary), "--info"], text=True)
    metadata["opencv_build_information"] = build_info
    (args.output / "results.json").write_text(json.dumps({"metadata": metadata, "results": rows}, indent=2) + "\n")
    print("\nwindow,mean_median_kornia_ms,mean_median_opencv_ms,median_speedup,mean_best_kornia_ms,mean_best_opencv_ms,best_speedup,mismatched_pixels")
    for window in WINDOWS:
        group = [row for row in rows if row["window"] == window]
        k_ms = statistics.mean(row["kornia_ms"] for row in group)
        c_ms = statistics.mean(row["opencv_ms"] for row in group)
        k_best = statistics.mean(row["kornia_best_ms"] for row in group)
        c_best = statistics.mean(row["opencv_best_ms"] for row in group)
        differences = sum(row["mismatch_pixels"] for row in group)
        print(f"{window},{k_ms:.6f},{c_ms:.6f},{c_ms / k_ms:.3f},"
              f"{k_best:.6f},{c_best:.6f},{c_best / k_best:.3f},{differences}")
    if any(row["mismatch_pixels"] for row in rows):
        raise SystemExit("Output differences found; inspect saved results before interpreting speed comparisons")


if __name__ == "__main__":
    main()
