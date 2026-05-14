"""Robotics-focused codec comparison: speed vs compression vs FPS budget.

Run: ``pixi run -e py312 python kornia-py/benchmarks/robotics_codec_comparison.py``

Compares JPEG / PNG / TIFF / AVIF across quality / compression levels at
1080p RGB on a synthetic image with the structure profile of a typical
robotics scene (smooth gradients + mid-frequency texture + high-frequency
detail). Output: per-codec-setting min encode time (best of N — the only
honest number for sub-millisecond ops on a noisy machine), min decode
time, output size, and which FPS budgets it fits (30 / 60 / 120 FPS).

Optional dep: ``pillow-avif-plugin`` (only needed for the AVIF rows).

Uses the reusable :mod:`_bench` helper (sibling file) for proper
warmup + GC-disable + best-of-N timing — see ``_bench.py`` for
methodology notes.
"""

import io

import numpy as np
from PIL import Image as PIL

from _bench import bench


def make_robot_scene(h=1080, w=1920):
    """Synthetic image matching robotics scene statistics."""
    rng = np.random.default_rng(42)
    yy, _ = np.mgrid[0:h, 0:w].astype(np.float32)
    sky = (yy / h * 80 + 100)[..., None] * np.array([0.7, 0.85, 1.0])
    blob_field = rng.standard_normal((h // 16, w // 16, 3)).astype(np.float32) * 30
    blobs_lo = np.clip(blob_field + 127, 0, 255).astype(np.uint8)
    blobs = np.array(
        PIL.fromarray(blobs_lo, "RGB").resize((w, h), PIL.BILINEAR),
        dtype=np.float32,
    ) - 127
    fine = rng.standard_normal((h, w, 3)).astype(np.float32) * 8
    return np.clip(sky + blobs + fine, 0, 255).astype(np.uint8)


def main():
    arr = make_robot_scene(1080, 1920)
    print(f"\nTest image: 1080x1920 RGB, mean={arr.mean():.0f}, std={arr.std():.0f}")
    print(f"Raw size: {arr.nbytes / 1024:.0f} KB")

    from kornia_rs.image import Image
    k_img = Image(arr)
    pil_img = PIL.fromarray(arr)
    try:
        import pillow_avif  # noqa: F401
        avif_ok = True
    except ImportError:
        avif_ok = False

    rows = []

    def add_row(label, setting, enc_fn, dec_fn=None, *, target_seconds=1.0):
        e = bench(enc_fn, name=label, target_seconds=target_seconds)
        # Run the encoder once to capture an output for size + decode bench.
        out = enc_fn()
        size = len(out) if isinstance(out, (bytes, bytearray)) else 0
        d_min = None
        if dec_fn is not None:
            d = bench(dec_fn, name=f"{label} decode", target_seconds=target_seconds)
            d_min = d.min_ms
        rows.append((label, setting, e.min_ms, d_min, size))

    # JPEG quality sweep (subsampling 4:2:0 default — matches cv2/PIL)
    for q in (50, 75, 85, 95):
        out = k_img.encode("jpeg", quality=q)
        blob = bytes(out)
        add_row("kornia JPEG 4:2:0", f"q={q}",
                lambda q=q: k_img.encode("jpeg", quality=q),
                lambda b=blob: Image.decode(b))

    # PNG compress_level sweep
    for level in (0, 1, 6, 9):
        # level=9 is slow; bound the total time.
        target = 0.5 if level >= 6 else 1.0
        out = k_img.encode("png", compress_level=level)
        blob = bytes(out)
        add_row("kornia PNG", f"level={level}",
                lambda lv=level: k_img.encode("png", compress_level=lv),
                lambda b=blob: Image.decode(b),
                target_seconds=target)

    # TIFF (lossless u8)
    out = k_img.encode("tiff")
    blob = bytes(out)
    add_row("kornia TIFF u8", "lossless",
            lambda: k_img.encode("tiff"),
            lambda b=blob: Image.decode(b))

    if avif_ok:
        for q, speed in [(50, 8), (75, 8), (85, 6), (95, 4)]:
            buf = io.BytesIO()

            def _enc(qq=q, ss=speed, _buf=buf):
                _buf.seek(0); _buf.truncate(0)
                pil_img.save(_buf, format="AVIF", quality=qq, speed=ss)
                return _buf.getvalue()

            out = _enc()

            def _dec(b=out):
                return PIL.open(io.BytesIO(b)).load()

            # AVIF q=95 is *very* slow — drop the budget to keep wall clock sane.
            target = 1.0 if q < 85 else 0.5
            add_row("AVIF (libavif)", f"q={q} sp={speed}", _enc, _dec, target_seconds=target)

    # Print
    print(
        f"\n{'codec':<22} {'setting':<14} {'enc ms':>9} {'dec ms':>9} "
        f"{'KB':>9} {'30':>4} {'60':>4} {'120':>5}"
    )
    print("-" * 82)
    for codec, setting, enc, dec, size in rows:
        b30 = "OK " if enc <= 33.3 else "   "
        b60 = "OK " if enc <= 16.7 else "   "
        b120 = "OK  " if enc <= 8.3 else "    "
        dec_s = f"{dec:>9.3f}" if dec is not None else f"{'-':>9}"
        print(
            f"{codec:<22} {setting:<14} {enc:>9.3f} {dec_s} "
            f"{size/1024:>9.1f} {b30:>4} {b60:>4} {b120:>5}"
        )

    print(
        "\nReported numbers are best-of-N min times (warmup + gc.disable + "
        "per-call timing).\nPicking min eliminates GC-pause / scheduler "
        "outliers for sub-millisecond ops."
    )
    print("\nSmallest 6 outputs (lower = better for bandwidth/archival):")
    for codec, setting, _, _, size in sorted(rows, key=lambda r: r[4])[:6]:
        print(f"  {size/1024:>7.1f} KB  {codec} {setting}")


if __name__ == "__main__":
    main()
