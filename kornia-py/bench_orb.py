"""Benchmark kornia-rs ORB vs OpenCV ORB.

Mirrors bench_augmentations.py style: random uint8 images at 640x480 and
1920x1080, 200 iterations, 10 warmups, median wall-clock per call.
"""
import json
import os
import time
import numpy as np
import kornia_rs as K
import cv2

N_ITERS = int(os.environ.get("ORB_BENCH_N", "200"))
N_WARMUP = int(os.environ.get("ORB_BENCH_WARMUP", "10"))


def bench(name, fn, n=N_ITERS, warmup=N_WARMUP):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    elapsed = (time.perf_counter() - t0) / n * 1000
    print(f"  {name:40s} {elapsed:8.3f} ms")
    return elapsed


def run_benchmarks():
    results = {}

    for label, (h, w) in [("640x480", (480, 640)), ("1920x1080", (1080, 1920))]:
        print(f"\n{'='*60}")
        print(f"Image size: {label} (HxW={h}x{w}, gray uint8)")
        print(f"{'='*60}")

        rng = np.random.default_rng(42)
        # ORB needs structure to find corners; pure noise yields very few
        # keypoints. Mix a checker-ish pattern with noise so both libraries see
        # roughly the same number of strong corners.
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        base = (((xx // 8) + (yy // 8)) & 1).astype(np.uint8) * 180 + 30
        noise = rng.integers(-40, 40, (h, w), dtype=np.int16)
        gray = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        orb_cv = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8,
                                 edgeThreshold=31, firstLevel=0, WTA_K=2,
                                 scoreType=cv2.ORB_HARRIS_SCORE,
                                 patchSize=31, fastThreshold=20)

        print("\n--- ORB detect + compute ---")
        # Light sanity: confirm both libraries produce a similar keypoint count.
        kps_k, _, desc_k = K.features.orb_detect_and_compute(gray)
        kps_cv, desc_cv = orb_cv.detectAndCompute(gray, None)
        print(f"  (kornia found {len(kps_k)} keypoints, opencv found {len(kps_cv)})")

        results[label] = {
            "kornia":  bench("kornia-rs",  lambda: K.features.orb_detect_and_compute(gray)),
            "opencv":  bench("opencv",     lambda: orb_cv.detectAndCompute(gray, None)),
        }

    return results


if __name__ == "__main__":
    results = run_benchmarks()
    print("\n" + json.dumps(results, indent=2))
