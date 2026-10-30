"""Baselines for `bench_depth.rs` on the same synthetic inputs.

Times, when the package is importable (each is skipped otherwise):

* Open3D legacy `geometry.PointCloud.create_from_depth_image` (CPU).
* Open3D tensor `t.geometry.PointCloud.create_from_depth_image` (CPU and,
  if Open3D was built with CUDA, `CUDA:0`).
* Registration has no single-call Open3D equivalent; the closest pipeline is
  tensor `create_from_depth_image` -> `transform` -> `project_to_depth_image`
  (a z-buffered projection), timed on CPU and CUDA.
* OpenCV contrib `cv2.rgbd.depthTo3d` and `cv2.rgbd.registerDepth`
  (`depthDilation=True` is OpenCV's hole-closing splat), CPU only.
* NumPy, always: a vectorised unprojection (precomputed ray tables, boolean
  compaction) and a registration (transform, round to the nearest pixel,
  z-buffer with `np.minimum.at`, splat radius 0 and 1).

Usage: python3 bench_depth_baselines.py [--rounds N]
       python3 bench_depth_baselines.py --write-fixtures DIR
Prints best-of-N milliseconds per op and size. `--write-fixtures` (needs
Open3D) writes `unproject_depth_open3d.bin` and `register_depth_open3d.bin`,
the parity fixtures read by the kornia-3d tests, into DIR (the repository's
`tests/data`) instead of timing anything.
"""

import argparse
import pathlib
import struct
import time

import numpy as np

SIZES = [(640, 480), (1280, 720)]
MM = 0.001


def depth_u16(w, h):
    """Mirror of `depth_u16` in bench_depth.rs (xorshift64, same constants)."""
    mask64 = (1 << 64) - 1
    state = 0x2545F4914F6CDD1D
    out = np.empty(w * h, dtype=np.uint16)
    for i in range(w * h):
        x, y = i % w, i // w
        state ^= (state << 13) & mask64
        state ^= state >> 7
        state ^= (state << 17) & mask64
        if state % 20 == 0:
            out[i] = 0
            continue
        plane = 1500 + (x * 1000 // w) + (y * 500 // h)
        in_box = w // 3 < x < w // 2 and h // 3 < y < 2 * h // 3
        out[i] = 800 if in_box else plane
    return out.reshape(h, w)


def intrinsics(w, h):
    f = 0.9 * w
    return f, f, 0.5 * w - 0.5, 0.5 * h - 0.5


def dst_from_src():
    a = 0.02
    t = np.eye(4)
    t[:3, :3] = [[np.cos(a), 0.0, np.sin(a)], [0.0, 1.0, 0.0], [-np.sin(a), 0.0, np.cos(a)]]
    t[0, 3] = 0.05
    return t


def best_ms(fn, rounds):
    fn()  # warm-up
    best = float("inf")
    for _ in range(rounds):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def rays(w, h):
    """Per-pixel `(x - cx) / fx` and `(y - cy) / fy`, computed once per size."""
    fx, fy, cx, cy = intrinsics(w, h)
    xn = ((np.arange(w, dtype=np.float32) - np.float32(cx)) * np.float32(1.0 / fx))
    yn = ((np.arange(h, dtype=np.float32) - np.float32(cy)) * np.float32(1.0 / fy))
    return np.broadcast_to(xn, (h, w)), np.broadcast_to(yn[:, None], (h, w))


def numpy_unproject(depth, xn, yn, scale):
    z = depth.astype(np.float32) * np.float32(scale)
    keep = z > 0
    zk = z[keep]
    return np.stack([xn[keep] * zk, yn[keep] * zk, zk], axis=-1)


def numpy_register(depth, xn, yn, scale, rt, w, h, radius):
    fx, fy, cx, cy = intrinsics(w, h)
    pts = numpy_unproject(depth, xn, yn, scale)
    r = rt[:3, :3].astype(np.float32)
    t = rt[:3, 3].astype(np.float32)
    q = pts @ r.T + t
    q = q[q[:, 2] > 0]
    iz = np.float32(1.0) / q[:, 2]
    u = np.floor(np.float32(fx) * q[:, 0] * iz + np.float32(cx + 0.5)).astype(np.int32)
    v = np.floor(np.float32(fy) * q[:, 1] * iz + np.float32(cy + 0.5)).astype(np.int32)
    val = np.rint(q[:, 2] / np.float32(scale)).astype(np.int32)
    ok = (val >= 1) & (val <= 65534)
    u, v, val = u[ok], v[ok], val[ok]
    zbuf = np.full(w * h, 65535, dtype=np.int32)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            uu, vv = u + dx, v + dy
            inb = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
            np.minimum.at(zbuf, vv[inb] * w + uu[inb], val[inb])
    zbuf[zbuf == 65535] = 0
    return zbuf.astype(np.uint16).reshape(h, w)


def bench_numpy(depth, w, h, rounds):
    xn, yn = rays(w, h)
    pts = numpy_unproject(depth, xn, yn, MM)
    print(f"numpy unproject_u16_cpu {w}x{h}: {best_ms(lambda: numpy_unproject(depth, xn, yn, MM), rounds):.3f} ms ({len(pts)} points)")
    rt = dst_from_src()
    for radius in (0, 1):
        ms = best_ms(lambda: numpy_register(depth, xn, yn, MM, rt, w, h, radius), rounds)
        print(f"numpy register_u16_r{radius}_cpu {w}x{h}: {ms:.3f} ms")


def bench_open3d(depth, w, h, rounds):
    try:
        import open3d as o3d
    except ImportError:
        print("open3d not installed; skipping")
        return
    fx, fy, cx, cy = intrinsics(w, h)
    intr = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    img = o3d.geometry.Image(depth)
    ms = best_ms(
        lambda: o3d.geometry.PointCloud.create_from_depth_image(
            img, intr, depth_scale=1.0 / MM, depth_trunc=1e9
        ),
        rounds,
    )
    print(f"open3d legacy unproject_u16_cpu {w}x{h}: {ms:.3f} ms")

    k = o3d.core.Tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], o3d.core.Dtype.Float64)
    ext = o3d.core.Tensor(dst_from_src(), o3d.core.Dtype.Float64)
    devices = [o3d.core.Device("CPU:0")]
    if o3d.core.cuda.is_available():
        devices.append(o3d.core.Device("CUDA:0"))
    for dev in devices:
        tag = "cuda" if dev.get_type() == o3d.core.Device.DeviceType.CUDA else "cpu"
        timg = o3d.t.geometry.Image(o3d.core.Tensor(depth, device=dev))
        sync = (lambda: o3d.core.cuda.synchronize()) if tag == "cuda" else (lambda: None)

        def unproject():
            pc = o3d.t.geometry.PointCloud.create_from_depth_image(
                timg, k, depth_scale=1.0 / MM, depth_max=1e9
            )
            sync()
            return pc

        ms = best_ms(unproject, rounds)
        print(f"open3d tensor unproject_u16_{tag} {w}x{h}: {ms:.3f} ms")

        def register():
            pc = o3d.t.geometry.PointCloud.create_from_depth_image(
                timg, k, depth_scale=1.0 / MM, depth_max=1e9
            )
            out = pc.project_to_depth_image(w, h, k, ext, depth_scale=1.0 / MM, depth_max=1e9)
            sync()
            return out

        ms = best_ms(register, rounds)
        print(f"open3d tensor register_u16_{tag} {w}x{h} (unproject+project): {ms:.3f} ms")


def bench_opencv(depth, w, h, rounds):
    try:
        import cv2

        rgbd = cv2.rgbd
    except (ImportError, AttributeError):
        print("cv2.rgbd (opencv-contrib) not available; skipping")
        return
    fx, fy, cx, cy = intrinsics(w, h)
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    ms = best_ms(lambda: rgbd.depthTo3d(depth, k), rounds)
    print(f"opencv rgbd.depthTo3d_u16_cpu {w}x{h}: {ms:.3f} ms")
    rt = dst_from_src()
    dist = np.zeros(5)
    for dil in (False, True):
        ms = best_ms(
            lambda: rgbd.registerDepth(k, k, dist, rt, depth, (w, h), depthDilation=dil),
            rounds,
        )
        print(f"opencv rgbd.registerDepth_u16_dilation={dil}_cpu {w}x{h}: {ms:.3f} ms")


def write_fixtures(out_dir):
    """A small random `u16` depth image and Open3D's unprojection of it.

    Little endian: `u32` width and height, `f64` fx, fy, cx, cy and
    `depth_scale`, the `u16` depth pixels row-major, then a `u64` count and
    the output `x, y, z` `f64` triplets of the legacy
    `geometry.PointCloud.create_from_depth_image` (row-major order).
    """
    import open3d as o3d

    rng = np.random.default_rng(7)
    w, h = 40, 30
    fx, fy, cx, cy = 37.5, 41.25, 19.3, 14.6
    depth_scale = 1000.0
    depth = rng.integers(1, 65536, size=(h, w), dtype=np.uint16)
    depth[rng.random((h, w)) < 0.2] = 0
    cam = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    pc = o3d.geometry.PointCloud.create_from_depth_image(
        o3d.geometry.Image(depth), cam, depth_scale=depth_scale, depth_trunc=1.0e9
    )
    pts = np.ascontiguousarray(np.asarray(pc.points), dtype="<f8")
    blob = struct.pack("<II5d", w, h, fx, fy, cx, cy, depth_scale)
    blob += np.ascontiguousarray(depth, dtype="<u2").tobytes()
    blob += struct.pack("<Q", len(pts)) + pts.tobytes()
    (pathlib.Path(out_dir) / "unproject_depth_open3d.bin").write_bytes(blob)


def register_reference(depth, src_k, dst_k, rt, dw, dh, depth_scale):
    """Float64 nearest-pixel z-buffer registration, the model of both Open3D
    `project_to_depth_image` and kornia `DepthImage::register_into` (radius 0).

    Returns the per-source-pixel destination `(u, v, qz)` (NaN where the
    sample is missing) and the destination depth in units (0 = empty)."""
    h, w = depth.shape
    fx, fy, cx, cy = src_k
    dfx, dfy, dcx, dcy = dst_k
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    z = depth.astype(np.float64) / depth_scale
    z[depth == 0] = np.nan
    p = np.stack([(xs - cx) / fx * z, (ys - cy) / fy * z, z], axis=-1)
    q = p @ rt[:3, :3].T + rt[:3, 3]
    u = dfx * q[..., 0] / q[..., 2] + dcx
    v = dfy * q[..., 1] / q[..., 2] + dcy
    out = np.zeros((dh, dw))
    ui, vi = np.floor(u + 0.5), np.floor(v + 0.5)
    ok = np.isfinite(q[..., 2]) & (q[..., 2] > 0)
    ok &= (ui >= 0) & (ui < dw) & (vi >= 0) & (vi < dh)
    for y, x in zip(*np.nonzero(ok)):
        cell = (int(vi[y, x]), int(ui[y, x]))
        val = q[y, x, 2] * depth_scale
        if out[cell] == 0 or val < out[cell]:
            out[cell] = val
    return u, v, q[..., 2], ok, out


def write_register_fixture(out_dir):
    """A small `u16` depth image registered into a second camera by Open3D.

    Open3D has no single registration call: the tensor
    `t.geometry.PointCloud.create_from_depth_image` (source intrinsics)
    followed by `project_to_depth_image` (destination intrinsics, extrinsic
    `dst_from_src`) is the same model as kornia's `DepthImage::register_into` at splat
    radius 0, a nearest-pixel projection into a z-buffer (Open3D keeps the
    output unrounded in `f32`).

    The two differ only in rounding (`f32` arithmetic, Open3D divides by
    `depth_scale`), which can flip a decision that is within rounding of a
    tie, so such source pixels are set to 0 before Open3D runs, using the
    `f64` model above: a projection within 1e-3 px of a pixel boundary, a
    destination depth within 1e-3 units of a `.5` (the `u16` output rounds),
    and every point of a destination pixel whose two nearest depths differ
    by less than 1e-5 (relative) without being equal. The remaining
    configuration is decided far from any tie, and Open3D's output is
    checked against the `f64` model before it is written.

    Little endian: `u32` width and height, `f64` fx, fy, cx, cy of the
    source; the same for the destination; `f64` `depth_scale`; `f64`
    `dst_from_src` as a row-major 3x4 `[R | t]` (metres); the `u16` source
    pixels row-major; then the destination's `f32` Open3D depth (units,
    0 = empty) row-major.
    """
    import open3d as o3d

    rng = np.random.default_rng(11)
    w, h = 40, 30
    src_k = (37.5, 41.25, 19.3, 14.6)
    dw, dh = 44, 32
    dst_k = (40.0, 43.5, 21.7, 15.2)
    depth_scale = 1000.0
    a, b = 0.03, -0.02
    ry = np.array([[np.cos(a), 0.0, np.sin(a)], [0.0, 1.0, 0.0], [-np.sin(a), 0.0, np.cos(a)]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(b), -np.sin(b)], [0.0, np.sin(b), np.cos(b)]])
    rt = np.eye(4)
    rt[:3, :3] = ry @ rx
    rt[:3, 3] = [0.05, -0.02, 0.03]

    # A slanted background plane with a near box (occlusions), ~15% missing.
    ys, xs = np.mgrid[0:h, 0:w]
    depth = (2500 + 20 * xs + 10 * ys + rng.integers(0, 40, size=(h, w))).astype(np.uint16)
    box = (xs >= 12) & (xs < 22) & (ys >= 8) & (ys < 20)
    depth[box] = 900 + rng.integers(0, 30, size=int(box.sum()))
    depth[rng.random((h, w)) < 0.15] = 0

    u, v, qz, ok, _ = register_reference(depth, src_k, dst_k, rt, dw, dh, depth_scale)
    near_half = lambda t: np.abs(t - np.floor(t) - 0.5) < 1e-3
    amb = ok & (near_half(u) | near_half(v) | near_half(qz * depth_scale))
    cells = {}
    for y, x in zip(*np.nonzero(ok & ~amb)):
        key = (int(np.floor(v[y, x] + 0.5)), int(np.floor(u[y, x] + 0.5)))
        cells.setdefault(key, []).append((qz[y, x], y, x))
    for pts in cells.values():
        zs = sorted(p[0] for p in pts)
        if len(zs) > 1 and zs[0] != zs[1] and zs[1] - zs[0] < 1e-5 * zs[0]:
            for _, y, x in pts:
                amb[y, x] = True
    depth[amb] = 0
    _, _, _, _, want = register_reference(depth, src_k, dst_k, rt, dw, dh, depth_scale)

    f64 = o3d.core.Dtype.Float64
    k_src = o3d.core.Tensor([[src_k[0], 0, src_k[2]], [0, src_k[1], src_k[3]], [0, 0, 1]], f64)
    k_dst = o3d.core.Tensor([[dst_k[0], 0, dst_k[2]], [0, dst_k[1], dst_k[3]], [0, 0, 1]], f64)
    pc = o3d.t.geometry.PointCloud.create_from_depth_image(
        o3d.t.geometry.Image(o3d.core.Tensor(depth)), k_src, depth_scale=depth_scale, depth_max=1e9
    )
    out = pc.project_to_depth_image(
        dw, dh, k_dst, o3d.core.Tensor(rt, f64), depth_scale=depth_scale, depth_max=1e9
    )
    o3d_depth = np.asarray(out.as_tensor().numpy(), dtype=np.float32).reshape(dh, dw)
    assert ((o3d_depth != 0) == (want != 0)).all(), "Open3D coverage differs from the f64 model"
    assert np.allclose(o3d_depth, want, rtol=1e-5, atol=0), "Open3D depth differs from the f64 model"
    assert (want != 0).sum() > 0.5 * dw * dh and amb.sum() < 0.05 * w * h

    blob = struct.pack("<II4d", w, h, *src_k)
    blob += struct.pack("<II4d", dw, dh, *dst_k)
    blob += struct.pack("<d", depth_scale)
    blob += np.ascontiguousarray(rt[:3, :4], dtype="<f8").tobytes()
    blob += np.ascontiguousarray(depth, dtype="<u2").tobytes()
    blob += np.ascontiguousarray(o3d_depth, dtype="<f4").tobytes()
    (pathlib.Path(out_dir) / "register_depth_open3d.bin").write_bytes(blob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--write-fixtures", metavar="DIR")
    args = ap.parse_args()
    if args.write_fixtures:
        write_fixtures(args.write_fixtures)
        write_register_fixture(args.write_fixtures)
        return
    for w, h in SIZES:
        depth = depth_u16(w, h)
        bench_numpy(depth, w, h, args.rounds)
        bench_open3d(depth, w, h, args.rounds)
        bench_opencv(depth, w, h, args.rounds)


if __name__ == "__main__":
    main()
