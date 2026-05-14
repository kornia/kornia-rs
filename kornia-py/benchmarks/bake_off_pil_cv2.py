"""Head-to-head bake-off: kornia vs PIL vs OpenCV.

Run: ``pixi run -e py312 python kornia-py/benchmarks/bake_off_pil_cv2.py``

Uses the reusable :mod:`_bench` helper for proper sub-millisecond
timing (best-of-N min, GC disabled). Reports each op three times — once
per library — and prints the speedup vs the best competitor.

This file is intentionally separate from ``tests/test_against_pil_cv2.py``
which gates CI on a coarse 1.3× envelope. This script gives the *real*
numbers for the PR description / docs.
"""

import io

import numpy as np

try:
    from PIL import Image as PIL, ImageFilter
except ImportError:
    raise SystemExit("PIL not installed — `uv pip install pillow`")
try:
    import cv2
except ImportError:
    raise SystemExit("cv2 not installed — `uv pip install opencv-python`")

from kornia_rs.image import Image
from _bench import compare, print_table, speedup_vs


def main():
    H, W = 1080, 1920
    arr = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
    pil_img = PIL.fromarray(arr)
    k_img = Image(arr)
    buf = io.BytesIO()

    def reset_buf():
        buf.seek(0); buf.truncate(0)

    # Pre-build encoded blobs for the decode races
    jbytes = bytes(k_img.encode("jpeg"))
    pbytes = bytes(k_img.encode("png"))
    jnp = np.frombuffer(jbytes, dtype=np.uint8)
    pnp = np.frombuffer(pbytes, dtype=np.uint8)

    cases = [
        ("encode JPEG q=95", {
            "PIL":    lambda: (reset_buf(), pil_img.save(buf, format="JPEG", quality=95)),
            "cv2":    lambda: cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, 95]),
            "kornia": lambda: k_img.encode("jpeg"),
        }),
        ("encode PNG level=1 (fdeflate)", {
            "PIL":    lambda: (reset_buf(), pil_img.save(buf, format="PNG")),
            "cv2":    lambda: cv2.imencode(".png", arr),
            "kornia": lambda: k_img.encode("png", compress_level=1),
        }),
        ("encode WebP", {
            "PIL":    lambda: (reset_buf(), pil_img.save(buf, format="WEBP")),
            "cv2":    lambda: cv2.imencode(".webp", arr),
            "kornia": lambda: k_img.encode("webp"),
        }),
        ("encode TIFF", {
            "PIL":    lambda: (reset_buf(), pil_img.save(buf, format="TIFF")),
            "cv2":    lambda: cv2.imencode(".tiff", arr),
            "kornia": lambda: k_img.encode("tiff"),
        }),
        ("decode JPEG", {
            "PIL":    lambda: PIL.open(io.BytesIO(jbytes)).load(),
            "cv2":    lambda: cv2.imdecode(jnp, cv2.IMREAD_COLOR),
            "kornia": lambda: Image.decode(jbytes),
        }),
        ("decode PNG", {
            "PIL":    lambda: PIL.open(io.BytesIO(pbytes)).load(),
            "cv2":    lambda: cv2.imdecode(pnp, cv2.IMREAD_COLOR),
            "kornia": lambda: Image.decode(pbytes),
        }),
        ("resize 1080p->720p", {
            "PIL":    lambda: pil_img.resize((1280, 720), PIL.LANCZOS),
            "cv2":    lambda: cv2.resize(arr, (1280, 720), interpolation=cv2.INTER_LANCZOS4),
            "kornia": lambda: k_img.resize(1280, 720),
        }),
        ("flip_horizontal", {
            "PIL":    lambda: pil_img.transpose(PIL.FLIP_LEFT_RIGHT),
            "cv2":    lambda: cv2.flip(arr, 1),
            "kornia": lambda: k_img.flip_horizontal(),
        }),
        ("crop 512x512", {
            "PIL":    lambda: pil_img.crop((100, 100, 612, 612)),
            "cv2":    lambda: arr[100:612, 100:612].copy(),
            "kornia": lambda: k_img.crop(100, 100, 512, 512),
        }),
        ("to_grayscale", {
            "PIL":    lambda: pil_img.convert("L"),
            "cv2":    lambda: cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY),
            "kornia": lambda: k_img.to_grayscale(),
        }),
        ("gaussian_blur k=3", {
            "PIL":    lambda: pil_img.filter(ImageFilter.GaussianBlur(radius=1.0)),
            "cv2":    lambda: cv2.GaussianBlur(arr, (3, 3), 1.0),
            "kornia": lambda: k_img.gaussian_blur(3, 1.0),
        }),
        ("tobytes", {
            "PIL":    lambda: pil_img.tobytes(),
            "cv2":    lambda: arr.tobytes(),
            "kornia": lambda: k_img.tobytes(),
        }),
    ]

    summary: list[tuple[str, str, float]] = []  # (op, winner, kornia_speedup_vs_best_competitor)
    for label, fns in cases:
        # 0.5s per case keeps total wall time ~20s for 12 cases × 3 runners.
        results = compare(fns, target_seconds=0.5)
        print_table(f"{label} (1080p)", results)
        # speedup vs the best non-kornia runner
        non_k_min = min(r.min_ms for n, r in results.items() if n != "kornia")
        k_min = results["kornia"].min_ms
        speedup = non_k_min / k_min  # >1 = kornia faster
        winner = "kornia" if k_min == min(r.min_ms for r in results.values()) else (
            min(((n, r.min_ms) for n, r in results.items()), key=lambda x: x[1])[0]
        )
        summary.append((label, winner, speedup))

    # Summary table
    print("\n=== Summary (best-of-N, kornia speedup vs best competitor) ===")
    print(f"  {'op':<28} {'winner':<8} {'kornia speedup':>16}")
    wins = ties = losses = 0
    for op, winner, sp in summary:
        flag = "win" if sp > 1.05 else ("tie" if sp > 0.90 else "loss")
        if flag == "win": wins += 1
        elif flag == "tie": ties += 1
        else: losses += 1
        print(f"  {op:<28} {winner:<8} {sp:>14.2f}x  {flag}")
    print(f"\n  kornia: {wins} wins, {ties} ties, {losses} losses across {len(summary)} ops")


if __name__ == "__main__":
    main()
