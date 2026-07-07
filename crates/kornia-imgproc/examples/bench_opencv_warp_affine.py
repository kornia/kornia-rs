"""Warp-affine benchmark: cv2 CPU, cv2 CUDA, PyTorch GPU vs kornia-rs GPU.

Run:
    python3 crates/kornia-imgproc/examples/bench_opencv_warp_affine.py

Compare against the Rust GPU kernel with:
    cargo run --example bench_cuda_warp_affine --features cuda --release

Requires:
  - opencv-python (CPU baseline, always)
  - OpenCV built with -DWITH_CUDA=ON (cv2.cuda section)
  - torch with CUDA (PyTorch GPU section)
"""

import time
import cv2
import numpy as np

WARMUP = 50
ITERS  = 200

# Same cases as bench_cuda_warp_affine.rs: 45° centre rotation, same-size canvas.
CASES = [(256, 224), (512, 448), (1024, 896), (1920, 1080)]

try:
    import torch
    import torch.nn.functional as F
    TORCH_CUDA = torch.cuda.is_available()
except ImportError:
    TORCH_CUDA = False

HAS_CV2_CUDA = (
    cv2.cuda.getCudaEnabledDeviceCount() > 0
    and hasattr(cv2.cuda, "warpAffine")
)

# ── helpers ───────────────────────────────────────────────────────────────────

def gb_per_sec(w, h, ms):
    """1 src read + 1 dst write, 3-ch f32."""
    return (w * h * 3 * 4 * 2) / (ms * 1e-3) / 1e9

def rotation_M(w, h, angle_deg=45.0):
    cx, cy = w / 2.0, h / 2.0
    return cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)

def opencv_M_to_torch_theta(M_fwd, W, H):
    """Convert a forward OpenCV 2×3 pixel-space matrix to PyTorch affine_grid
    theta (normalised, inverse mapping).

    PyTorch affine_grid with align_corners=True uses:
        x_norm = 2*x/(W-1) - 1

    This function computes the 2×3 theta such that:
        grid_sample(src, affine_grid(theta, (1,C,H,W))) ≡ warpAffine(src, M_fwd)
    """
    M_inv = cv2.invertAffineTransform(M_fwd)          # dst_pixel → src_pixel
    a, b, tx = M_inv[0]
    d, e, ty  = M_inv[1]
    # Expand pixel-space inverse into normalised coordinates.
    sx = (W - 1) / 2.0
    sy = (H - 1) / 2.0
    theta = np.array([
        [a,       b * sy / sx,  (a + b * sy / sx - 1) + tx / sx],
        [d * sx / sy, e,        (d * sx / sy + e - 1) + ty / sy],
    ], dtype=np.float32)
    return theta

# ── cv2 CPU ───────────────────────────────────────────────────────────────────

def bench_cv2_cpu():
    print(f"\n=== cv2 {cv2.__version__} CPU  warpAffine bilinear  ({ITERS} iters) ===")
    print(f"  {'case':<16}  {'ms/iter':>9}  {'GB/s':>8}  {'speedup vs kornia-rs GPU':>26}")
    print("  " + "-" * 66)
    # kornia-rs GPU bilinear numbers from bench_cuda_warp_affine (for reference)
    rs_gpu_ms = {(256,224): 0.025, (512,448): 0.092, (1024,896): 0.274, (1920,1080): 0.572}
    for w, h in CASES:
        src = np.random.rand(h, w, 3).astype(np.float32)
        M   = rotation_M(w, h)
        for _ in range(WARMUP):
            cv2.warpAffine(src, M, (w, h), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        t0 = time.perf_counter()
        for _ in range(ITERS):
            cv2.warpAffine(src, M, (w, h), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        ms = (time.perf_counter() - t0) * 1e3 / ITERS
        speedup = ms / rs_gpu_ms.get((w, h), ms)
        print(f"  {w}×{h:<10}  {ms:>9.3f}  {gb_per_sec(w,h,ms):>8.2f}  {speedup:>6.1f}× slower than RS GPU")

# ── cv2 CUDA ──────────────────────────────────────────────────────────────────

def bench_cv2_cuda():
    if not HAS_CV2_CUDA:
        print(f"\n=== cv2 CUDA warpAffine — SKIPPED ===")
        if cv2.cuda.getCudaEnabledDeviceCount() == 0:
            print("  No CUDA-enabled OpenCV device found.")
            print("  Build OpenCV from source with -DWITH_CUDA=ON -DCUDA_ARCH_BIN=7.5")
        else:
            print("  cv2.cuda.warpAffine not found in this build.")
        return

    print(f"\n=== cv2 CUDA  warpAffine bilinear  ({ITERS} iters) ===")
    print(f"  {'case':<16}  {'ms/iter':>9}  {'GB/s':>8}  {'vs RS GPU':>12}")
    print("  " + "-" * 52)
    rs_gpu_ms = {(256,224): 0.025, (512,448): 0.092, (1024,896): 0.274, (1920,1080): 0.572}
    for w, h in CASES:
        src = np.random.rand(h, w, 3).astype(np.float32)
        M   = rotation_M(w, h)
        gpu_src = cv2.cuda_GpuMat()
        gpu_src.upload(src)
        for _ in range(WARMUP):
            cv2.cuda.warpAffine(gpu_src, M, (w, h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)
        cv2.cuda.Stream_Null().waitForCompletion()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            cv2.cuda.warpAffine(gpu_src, M, (w, h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)
        cv2.cuda.Stream_Null().waitForCompletion()
        ms = (time.perf_counter() - t0) * 1e3 / ITERS
        rs_ms = rs_gpu_ms.get((w, h), ms)
        ratio = ms / rs_ms
        cmp = f"{ratio:.2f}× {'slower' if ratio > 1 else 'faster'}"
        print(f"  {w}×{h:<10}  {ms:>9.3f}  {gb_per_sec(w,h,ms):>8.2f}  {cmp:>12}")

# ── PyTorch GPU ───────────────────────────────────────────────────────────────

def bench_torch_gpu():
    if not TORCH_CUDA:
        print(f"\n=== PyTorch GPU warpAffine — SKIPPED (no CUDA torch) ===")
        return
    print(f"\n=== PyTorch {torch.__version__} GPU  grid_sample bilinear  ({ITERS} iters) ===")
    print(f"  {'case':<16}  {'ms/iter':>9}  {'GB/s':>8}  {'vs RS GPU':>12}")
    print("  " + "-" * 52)
    rs_gpu_ms = {(256,224): 0.025, (512,448): 0.092, (1024,896): 0.274, (1920,1080): 0.572}
    for w, h in CASES:
        src_np = np.random.rand(h, w, 3).astype(np.float32)
        M      = rotation_M(w, h)
        theta  = opencv_M_to_torch_theta(M, w, h)
        theta_t = torch.from_numpy(theta).unsqueeze(0).cuda()   # 1×2×3
        src_t   = torch.from_numpy(src_np).permute(2, 0, 1).unsqueeze(0).cuda()  # 1×3×H×W

        for _ in range(WARMUP):
            grid = F.affine_grid(theta_t, src_t.shape, align_corners=True)
            F.grid_sample(src_t, grid, mode="bilinear", padding_mode="zeros",
                          align_corners=True)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(ITERS):
            grid = F.affine_grid(theta_t, src_t.shape, align_corners=True)
            F.grid_sample(src_t, grid, mode="bilinear", padding_mode="zeros",
                          align_corners=True)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / ITERS
        rs_ms = rs_gpu_ms.get((w, h), ms)
        ratio = ms / rs_ms
        cmp = f"{ratio:.2f}× {'slower' if ratio > 1 else 'faster'}"
        print(f"  {w}×{h:<10}  {ms:>9.3f}  {gb_per_sec(w,h,ms):>8.2f}  {cmp:>12}")

# ── main ──────────────────────────────────────────────────────────────────────

print("=" * 66)
print("  Warp-affine benchmark  —  45° centre rotation, 3-ch f32, same-size canvas")
print(f"  OpenCV {cv2.__version__}  |  cv2.cuda devices: {cv2.cuda.getCudaEnabledDeviceCount()}")
if TORCH_CUDA:
    import torch
    print(f"  PyTorch {torch.__version__}  |  GPU: {torch.cuda.get_device_name(0)}")
print("=" * 66)
print()
print("  kornia-rs GPU reference (run separately):")
print("    cargo run --example bench_cuda_warp_affine --features cuda --release")
print()
print("  kornia-rs results (GTX 1650, 200 iters):")
print("    nearest : 256×224=0.011ms  512×448=0.038ms  1024×896=0.151ms  1920×1080=0.353ms")
print("    bilinear: 256×224=0.025ms  512×448=0.092ms  1024×896=0.274ms  1920×1080=0.572ms")

bench_cv2_cpu()
bench_cv2_cuda()
bench_torch_gpu()

# ── bicubic ───────────────────────────────────────────────────────────────────

def bench_cv2_cpu_bicubic():
    print(f"\n=== cv2 {cv2.__version__} CPU  warpAffine bicubic (INTER_CUBIC)  ({ITERS} iters) ===")
    print(f"  {'case':<16}  {'ms/iter':>9}  {'GB/s':>8}  {'speedup vs kornia-rs GPU':>26}")
    print("  " + "-" * 66)
    rs_gpu_ms = {(256,224): 0.065, (512,448): 0.244, (1024,896): 0.936, (1920,1080): 1.951}
    for w, h in CASES:
        src = np.random.rand(h, w, 3).astype(np.float32)
        M   = rotation_M(w, h)
        for _ in range(WARMUP):
            cv2.warpAffine(src, M, (w, h), flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        t0 = time.perf_counter()
        for _ in range(ITERS):
            cv2.warpAffine(src, M, (w, h), flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        ms = (time.perf_counter() - t0) * 1e3 / ITERS
        speedup = ms / rs_gpu_ms.get((w, h), ms)
        print(f"  {w}×{h:<10}  {ms:>9.3f}  {gb_per_sec(w,h,ms):>8.2f}  {speedup:>6.1f}× slower than RS GPU")

def bench_cv2_cuda_bicubic():
    if not HAS_CV2_CUDA:
        print(f"\n=== cv2 CUDA warpAffine bicubic — SKIPPED (no CUDA OpenCV) ===")
        return
    print(f"\n=== cv2 CUDA  warpAffine bicubic  ({ITERS} iters) ===")
    print(f"  {'case':<16}  {'ms/iter':>9}  {'GB/s':>8}  {'vs RS GPU':>12}")
    print("  " + "-" * 52)
    rs_gpu_ms = {(256,224): 0.065, (512,448): 0.244, (1024,896): 0.936, (1920,1080): 1.951}
    for w, h in CASES:
        src = np.random.rand(h, w, 3).astype(np.float32)
        M   = rotation_M(w, h)
        gpu_src = cv2.cuda_GpuMat()
        gpu_src.upload(src)
        for _ in range(WARMUP):
            cv2.cuda.warpAffine(gpu_src, M, (w, h),
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_CONSTANT)
        cv2.cuda.Stream_Null().waitForCompletion()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            cv2.cuda.warpAffine(gpu_src, M, (w, h),
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_CONSTANT)
        cv2.cuda.Stream_Null().waitForCompletion()
        ms = (time.perf_counter() - t0) * 1e3 / ITERS
        rs_ms = rs_gpu_ms.get((w, h), ms)
        ratio = ms / rs_ms
        cmp = f"{ratio:.2f}× {'slower' if ratio > 1 else 'faster'}"
        print(f"  {w}×{h:<10}  {ms:>9.3f}  {gb_per_sec(w,h,ms):>8.2f}  {cmp:>12}")

def bench_torch_gpu_bicubic():
    if not TORCH_CUDA:
        print(f"\n=== PyTorch GPU warpAffine bicubic — SKIPPED (no CUDA torch) ===")
        return
    print(f"\n=== PyTorch {torch.__version__} GPU  grid_sample bicubic  ({ITERS} iters) ===")
    print(f"  {'case':<16}  {'ms/iter':>9}  {'GB/s':>8}  {'vs RS GPU':>12}")
    print("  " + "-" * 52)
    rs_gpu_ms = {(256,224): 0.065, (512,448): 0.244, (1024,896): 0.936, (1920,1080): 1.951}
    for w, h in CASES:
        src_np = np.random.rand(h, w, 3).astype(np.float32)
        M      = rotation_M(w, h)
        theta  = opencv_M_to_torch_theta(M, w, h)
        theta_t = torch.from_numpy(theta).unsqueeze(0).cuda()
        src_t   = torch.from_numpy(src_np).permute(2, 0, 1).unsqueeze(0).cuda()
        for _ in range(WARMUP):
            grid = F.affine_grid(theta_t, src_t.shape, align_corners=True)
            F.grid_sample(src_t, grid, mode="bicubic", padding_mode="zeros",
                          align_corners=True)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(ITERS):
            grid = F.affine_grid(theta_t, src_t.shape, align_corners=True)
            F.grid_sample(src_t, grid, mode="bicubic", padding_mode="zeros",
                          align_corners=True)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / ITERS
        rs_ms = rs_gpu_ms.get((w, h), ms)
        ratio = ms / rs_ms
        cmp = f"{ratio:.2f}× {'slower' if ratio > 1 else 'faster'}"
        print(f"  {w}×{h:<10}  {ms:>9.3f}  {gb_per_sec(w,h,ms):>8.2f}  {cmp:>12}")

bench_cv2_cpu_bicubic()
bench_cv2_cuda_bicubic()
bench_torch_gpu_bicubic()
