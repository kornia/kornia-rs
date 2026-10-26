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
Prints best-of-N milliseconds per op and size.
"""

import argparse
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
    val = np.rint(q[:, 2] * np.float32(1.0 / scale)).astype(np.int32)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=50)
    args = ap.parse_args()
    for w, h in SIZES:
        depth = depth_u16(w, h)
        bench_numpy(depth, w, h, args.rounds)
        bench_open3d(depth, w, h, args.rounds)
        bench_opencv(depth, w, h, args.rounds)


if __name__ == "__main__":
    main()
