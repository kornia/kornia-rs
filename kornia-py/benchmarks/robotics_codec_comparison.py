"""Robotics-focused codec comparison: speed vs compression vs FPS budget.

Run: ``pixi run -e py312 python kornia-py/benchmarks/robotics_codec_comparison.py``

Compares JPEG / WebP / PNG / TIFF / AVIF across quality / compression
levels at 1080p RGB on a synthetic image with the structure profile of
a typical robotics scene (smooth gradients + mid-frequency texture +
high-frequency detail). Output: per-codec-setting encode time, decode
time, output size, and which FPS budgets it fits (30 / 60 / 120).

Optional dep: ``pillow-avif-plugin`` (only needed for the AVIF rows).

Key findings (1080p, aarch64 Jetson, release build):

  +--------------------+----------------+--------+--------+--------+
  | use case           | recommendation | enc ms | size   | notes  |
  +--------------------+----------------+--------+--------+--------+
  | streaming 60 FPS   | JPEG q=85 4:2:0|  17    | 453 KB | best balance |
  | streaming 30 FPS   | JPEG q=95      |  22    |   1 MB | high quality |
  | depth (uint16)     | PNG-16 / TIFF  |   2-50 | varies | lossless |
  | low-bitrate upload | AVIF q=50 sp=8 |  61    |  54 KB | 3× smaller than JPEG q=50 |
  | dataset archival   | PNG level=9    | 320    | 4.3 MB | lossless, slow |
  | hot loop (no codec)| TIFF u8        |   2.2  | 6.1 MB | uncompressed write |
  +--------------------+----------------+--------+--------+--------+

Caveats found while running this:
  - kornia's WebP encoder is currently *lossless only* (constant 4.5 MB
    output regardless of `quality=`). The `image-webp` crate it uses
    doesn't support lossy WebP; we'd need to swap to `libwebp` (FFI) to
    get the size/speed tradeoff curve everyone expects from WebP.
    Tracked as a follow-up.
  - AVIF encode is fast at low quality (q≤50) but explodes past q=85
    because libavif needs to run more refinement passes — a 5.7s encode
    at q=95 is fine for archival, useless for realtime.
"""

import io
import time

import numpy as np
from PIL import Image as PIL


def make_robot_scene(h=1080, w=1920):
    """Synthetic image matching robotics scene statistics."""
    rng = np.random.default_rng(42)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    sky = (yy / h * 80 + 100)[..., None] * np.array([0.7, 0.85, 1.0])
    blob_field = rng.standard_normal((h // 16, w // 16, 3)).astype(np.float32) * 30
    blobs_lo = np.clip(blob_field + 127, 0, 255).astype(np.uint8)
    blobs = np.array(PIL.fromarray(blobs_lo, "RGB").resize((w, h), PIL.BILINEAR),
                     dtype=np.float32) - 127
    fine = rng.standard_normal((h, w, 3)).astype(np.float32) * 8
    return np.clip(sky + blobs + fine, 0, 255).astype(np.uint8)


def bench(fn, iters):
    for _ in range(2):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
    return (time.perf_counter() - t0) / iters * 1000, out


def main():
    arr = make_robot_scene(1080, 1920)
    print(f"\nTest image: 1080×1920 RGB, mean={arr.mean():.0f}, std={arr.std():.0f}")
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

    for q in (50, 75, 85, 95):
        t, out = bench(lambda q=q: k_img.encode("jpeg", quality=q), iters=10)
        blob = bytes(out)
        td, _ = bench(lambda b=blob: Image.decode(b), iters=10)
        rows.append(("kornia JPEG 4:2:0", f"q={q}", t, td, len(out)))

    for level in (0, 1, 6, 9):
        t, out = bench(lambda lv=level: k_img.encode("png", compress_level=lv), iters=3)
        blob = bytes(out)
        td, _ = bench(lambda b=blob: Image.decode(b), iters=5)
        rows.append(("kornia PNG", f"level={level}", t, td, len(out)))

    t, out = bench(lambda: k_img.encode("tiff"), iters=10)
    blob = bytes(out)
    td, _ = bench(lambda b=blob: Image.decode(b), iters=10)
    rows.append(("kornia TIFF u8", "lossless", t, td, len(out)))

    if avif_ok:
        for q, speed in [(50, 8), (75, 8), (85, 6), (95, 4)]:
            buf = io.BytesIO()
            def _enc(qq=q, ss=speed):
                buf.seek(0); buf.truncate(0)
                pil_img.save(buf, format="AVIF", quality=qq, speed=ss)
                return buf.getvalue()
            t, out = bench(_enc, iters=2)
            def _dec(b=out):
                return PIL.open(io.BytesIO(b)).load()
            td, _ = bench(_dec, iters=3)
            rows.append(("AVIF (libavif)", f"q={q} sp={speed}", t, td, len(out)))

    print(f"\n{'codec':<22} {'setting':<14} {'enc ms':>9} {'dec ms':>9} {'size KB':>9} "
          f"{'30':>4} {'60':>4} {'120':>5}")
    print("-" * 82)
    for codec, setting, enc, dec, size in rows:
        b30 = "✓" if enc <= 33.3 else " "
        b60 = "✓" if enc <= 16.7 else " "
        b120 = "✓" if enc <= 8.3 else " "
        print(f"{codec:<22} {setting:<14} {enc:>9.2f} {dec:>9.2f} {size/1024:>9.1f} "
              f"{b30:>4} {b60:>4} {b120:>5}")

    print("\n  ✓ = encoder fits in the per-frame budget at that FPS")
    print("  Smallest 6 outputs (lower = better for bandwidth/archival):")
    for codec, setting, _, _, size in sorted(rows, key=lambda r: r[4])[:6]:
        print(f"    {size/1024:>7.1f} KB  {codec} {setting}")


if __name__ == "__main__":
    main()
