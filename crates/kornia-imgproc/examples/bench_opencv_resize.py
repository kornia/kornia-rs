"""Resize benchmark: cv2 CPU, cv2 CUDA, PyTorch GPU vs kornia-rs GPU.

Run:
    python3 crates/kornia-imgproc/examples/bench_opencv_resize.py

Compare against the Rust GPU kernel with:
    cargo run --example bench_cuda_resize --features cuda --release

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

# Downscale cases matching bench_cuda_resize.rs.
CASES = [
    ((1024, 1024), (512, 512)),
    ((1920, 1080), (960, 540)),
    ((3840, 2160), (1920, 1080)),
]

# All cases including upscale (matches bicubic section of bench_cuda_resize.rs).
CASES_BICUBIC = [
    ((1024, 1024), (512, 512)),
    ((512, 512), (1024, 1024)),
    ((1920, 1080), (960, 540)),
    ((1920, 1080), (3840, 2160)),
    ((3840, 2160), (1920, 1080)),
]

try:
    import torch
    import torch.nn.functional as F
    TORCH_CUDA = torch.cuda.is_available()
except ImportError:
    TORCH_CUDA = False

HAS_CV2_CUDA = (
    cv2.cuda.getCudaEnabledDeviceCount() > 0
    and hasattr(cv2.cuda, "resize")
)

# ── helpers ───────────────────────────────────────────────────────────────────

def gb_per_sec_resize(sw, sh, dw, dh, ms):
    """1 src read (approx) + 1 dst write, 3-ch f32."""
    return (dw * dh * 3 * 4 * 2) / (ms * 1e-3) / 1e9

def case_label(sw, sh, dw, dh):
    return f"{sw}×{sh}→{dw}×{dh}"

# ── cv2 CPU ───────────────────────────────────────────────────────────────────

def bench_cv2_cpu():
    print(f"\n=== cv2 {cv2.__version__} CPU  resize bilinear  ({ITERS} iters) ===")
    print(f"  {'case':<28}  {'ms/iter':>9}  {'GB/s':>8}")
    print("  " + "-" * 50)
    for (sw, sh), (dw, dh) in CASES:
        src = np.random.rand(sh, sw, 3).astype(np.float32)
        for _ in range(WARMUP):
            cv2.resize(src, (dw, dh), interpolation=cv2.INTER_LINEAR)
        t0 = time.perf_counter()
        for _ in range(ITERS):
            cv2.resize(src, (dw, dh), interpolation=cv2.INTER_LINEAR)
        ms = (time.perf_counter() - t0) * 1e3 / ITERS
        print(f"  {case_label(sw,sh,dw,dh):<28}  {ms:>9.3f}  {gb_per_sec_resize(sw,sh,dw,dh,ms):>8.2f}")

    print(f"\n=== cv2 {cv2.__version__} CPU  resize nearest  ({ITERS} iters) ===")
    print(f"  {'case':<28}  {'ms/iter':>9}  {'GB/s':>8}")
    print("  " + "-" * 50)
    for (sw, sh), (dw, dh) in CASES:
        src = np.random.rand(sh, sw, 3).astype(np.float32)
        for _ in range(WARMUP):
            cv2.resize(src, (dw, dh), interpolation=cv2.INTER_NEAREST)
        t0 = time.perf_counter()
        for _ in range(ITERS):
            cv2.resize(src, (dw, dh), interpolation=cv2.INTER_NEAREST)
        ms = (time.perf_counter() - t0) * 1e3 / ITERS
        print(f"  {case_label(sw,sh,dw,dh):<28}  {ms:>9.3f}  {gb_per_sec_resize(sw,sh,dw,dh,ms):>8.2f}")

# ── cv2 CUDA ──────────────────────────────────────────────────────────────────

def bench_cv2_cuda():
    if not HAS_CV2_CUDA:
        print(f"\n=== cv2 CUDA resize — SKIPPED ===")
        if cv2.cuda.getCudaEnabledDeviceCount() == 0:
            print("  No CUDA-enabled OpenCV device found.")
            print("  Build OpenCV from source with -DWITH_CUDA=ON -DCUDA_ARCH_BIN=7.5")
        else:
            print("  cv2.cuda.resize not found in this build.")
        return

    for interp_name, interp_flag in [("bilinear", cv2.INTER_LINEAR), ("nearest", cv2.INTER_NEAREST)]:
        print(f"\n=== cv2 CUDA  resize {interp_name}  ({ITERS} iters) ===")
        print(f"  {'case':<28}  {'ms/iter':>9}  {'GB/s':>8}  {'vs RS GPU':>12}")
        print("  " + "-" * 65)
        # kornia-rs GPU bilinear reference
        rs_ms_bil = {
            (1024,1024,512,512): 0.082, (1920,1080,960,540): 0.101, (3840,2160,1920,1080): 0.385
        }
        for (sw, sh), (dw, dh) in CASES:
            src = np.random.rand(sh, sw, 3).astype(np.float32)
            gpu_src = cv2.cuda_GpuMat()
            gpu_src.upload(src)
            for _ in range(WARMUP):
                cv2.cuda.resize(gpu_src, (dw, dh), interpolation=interp_flag)
            cv2.cuda.Stream_Null().waitForCompletion()
            t0 = time.perf_counter()
            for _ in range(ITERS):
                cv2.cuda.resize(gpu_src, (dw, dh), interpolation=interp_flag)
            cv2.cuda.Stream_Null().waitForCompletion()
            ms = (time.perf_counter() - t0) * 1e3 / ITERS
            rs_ms = rs_ms_bil.get((sw, sh, dw, dh), ms)
            ratio = ms / rs_ms
            cmp = f"{ratio:.2f}× {'slower' if ratio > 1 else 'faster'}"
            print(f"  {case_label(sw,sh,dw,dh):<28}  {ms:>9.3f}  {gb_per_sec_resize(sw,sh,dw,dh,ms):>8.2f}  {cmp:>12}")

# ── PyTorch GPU ───────────────────────────────────────────────────────────────

def bench_torch_gpu():
    if not TORCH_CUDA:
        print(f"\n=== PyTorch GPU resize — SKIPPED (no CUDA torch) ===")
        return
    print(f"\n=== PyTorch {torch.__version__} GPU  interpolate bilinear  ({ITERS} iters) ===")
    print(f"  {'case':<28}  {'ms/iter':>9}  {'GB/s':>8}  {'vs RS GPU':>12}")
    print("  " + "-" * 65)
    rs_ms_bil = {
        (1024,1024,512,512): 0.082, (1920,1080,960,540): 0.101, (3840,2160,1920,1080): 0.385
    }
    for (sw, sh), (dw, dh) in CASES:
        src_np = np.random.rand(sh, sw, 3).astype(np.float32)
        src_t  = torch.from_numpy(src_np).permute(2, 0, 1).unsqueeze(0).cuda()

        for _ in range(WARMUP):
            F.interpolate(src_t, size=(dh, dw), mode="bilinear", align_corners=False)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(ITERS):
            F.interpolate(src_t, size=(dh, dw), mode="bilinear", align_corners=False)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / ITERS
        rs_ms = rs_ms_bil.get((sw, sh, dw, dh), ms)
        ratio = ms / rs_ms
        cmp = f"{ratio:.2f}× {'slower' if ratio > 1 else 'faster'}"
        print(f"  {case_label(sw,sh,dw,dh):<28}  {ms:>9.3f}  {gb_per_sec_resize(sw,sh,dw,dh,ms):>8.2f}  {cmp:>12}")

    print(f"\n=== PyTorch {torch.__version__} GPU  interpolate nearest  ({ITERS} iters) ===")
    print(f"  {'case':<28}  {'ms/iter':>9}  {'GB/s':>8}  {'vs RS GPU':>12}")
    print("  " + "-" * 65)
    rs_ms_nn = {
        (1024,1024,512,512): 0.053, (1920,1080,960,540): 0.064, (3840,2160,1920,1080): 0.237
    }
    for (sw, sh), (dw, dh) in CASES:
        src_np = np.random.rand(sh, sw, 3).astype(np.float32)
        src_t  = torch.from_numpy(src_np).permute(2, 0, 1).unsqueeze(0).cuda()

        for _ in range(WARMUP):
            F.interpolate(src_t, size=(dh, dw), mode="nearest")
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(ITERS):
            F.interpolate(src_t, size=(dh, dw), mode="nearest")
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / ITERS
        rs_ms = rs_ms_nn.get((sw, sh, dw, dh), ms)
        ratio = ms / rs_ms
        cmp = f"{ratio:.2f}× {'slower' if ratio > 1 else 'faster'}"
        print(f"  {case_label(sw,sh,dw,dh):<28}  {ms:>9.3f}  {gb_per_sec_resize(sw,sh,dw,dh,ms):>8.2f}  {cmp:>12}")

# ── main ──────────────────────────────────────────────────────────────────────

print("=" * 66)
print("  Resize benchmark  —  3-ch f32 downscale")
print(f"  OpenCV {cv2.__version__}  |  cv2.cuda devices: {cv2.cuda.getCudaEnabledDeviceCount()}")
if TORCH_CUDA:
    import torch
    print(f"  PyTorch {torch.__version__}  |  GPU: {torch.cuda.get_device_name(0)}")
print("=" * 66)
print()
print("  kornia-rs GPU reference (run separately):")
print("    cargo run --example bench_cuda_resize --features cuda --release")
print()
print("  kornia-rs results (GTX 1650, 200 iters):")
print("    nearest : 1024²→512²=0.053ms  1920×1080→960×540=0.064ms  4K→1080=0.237ms")
print("    bilinear: 1024²→512²=0.082ms  1920×1080→960×540=0.101ms  4K→1080=0.385ms")

bench_cv2_cpu()
bench_cv2_cuda()
bench_torch_gpu()

# ── bicubic ───────────────────────────────────────────────────────────────────

def bench_cv2_cpu_bicubic():
    print(f"\n=== cv2 {cv2.__version__} CPU  resize bicubic (INTER_CUBIC)  ({ITERS} iters) ===")
    print(f"  {'case':<28}  {'ms/iter':>9}  {'GB/s':>8}")
    print("  " + "-" * 50)
    for (sw, sh), (dw, dh) in CASES_BICUBIC:
        src = np.random.rand(sh, sw, 3).astype(np.float32)
        for _ in range(WARMUP):
            cv2.resize(src, (dw, dh), interpolation=cv2.INTER_CUBIC)
        t0 = time.perf_counter()
        for _ in range(ITERS):
            cv2.resize(src, (dw, dh), interpolation=cv2.INTER_CUBIC)
        ms = (time.perf_counter() - t0) * 1e3 / ITERS
        print(f"  {case_label(sw,sh,dw,dh):<28}  {ms:>9.3f}  {gb_per_sec_resize(sw,sh,dw,dh,ms):>8.2f}")

def bench_cv2_cuda_bicubic():
    if not HAS_CV2_CUDA:
        print(f"\n=== cv2 CUDA resize bicubic — SKIPPED (no CUDA OpenCV) ===")
        return
    print(f"\n=== cv2 CUDA  resize bicubic  ({ITERS} iters) ===")
    print(f"  {'case':<28}  {'ms/iter':>9}  {'GB/s':>8}  {'vs RS GPU':>12}")
    print("  " + "-" * 65)
    rs_ms_bicubic = {
        (1024,1024,512,512): 0.120, (512,512,1024,1024): 0.207,
        (1920,1080,960,540): 0.245, (1920,1080,3840,2160): 1.709,
        (3840,2160,1920,1080): 0.959,
    }
    for (sw, sh), (dw, dh) in CASES_BICUBIC:
        src = np.random.rand(sh, sw, 3).astype(np.float32)
        gpu_src = cv2.cuda_GpuMat()
        gpu_src.upload(src)
        for _ in range(WARMUP):
            cv2.cuda.resize(gpu_src, (dw, dh), interpolation=cv2.INTER_CUBIC)
        cv2.cuda.Stream_Null().waitForCompletion()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            cv2.cuda.resize(gpu_src, (dw, dh), interpolation=cv2.INTER_CUBIC)
        cv2.cuda.Stream_Null().waitForCompletion()
        ms = (time.perf_counter() - t0) * 1e3 / ITERS
        rs_ms = rs_ms_bicubic.get((sw, sh, dw, dh), ms)
        ratio = ms / rs_ms
        cmp = f"{ratio:.2f}× {'slower' if ratio > 1 else 'faster'}"
        print(f"  {case_label(sw,sh,dw,dh):<28}  {ms:>9.3f}  {gb_per_sec_resize(sw,sh,dw,dh,ms):>8.2f}  {cmp:>12}")

def bench_torch_gpu_bicubic():
    if not TORCH_CUDA:
        print(f"\n=== PyTorch GPU resize bicubic — SKIPPED (no CUDA torch) ===")
        return
    print(f"\n=== PyTorch {torch.__version__} GPU  interpolate bicubic  ({ITERS} iters) ===")
    print(f"  {'case':<28}  {'ms/iter':>9}  {'GB/s':>8}  {'vs RS GPU':>12}")
    print("  " + "-" * 65)
    rs_ms_bicubic = {
        (1024,1024,512,512): 0.120, (512,512,1024,1024): 0.207,
        (1920,1080,960,540): 0.245, (1920,1080,3840,2160): 1.709,
        (3840,2160,1920,1080): 0.959,
    }
    for (sw, sh), (dw, dh) in CASES_BICUBIC:
        src_np = np.random.rand(sh, sw, 3).astype(np.float32)
        src_t  = torch.from_numpy(src_np).permute(2, 0, 1).unsqueeze(0).cuda()
        for _ in range(WARMUP):
            F.interpolate(src_t, size=(dh, dw), mode="bicubic", align_corners=False)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(ITERS):
            F.interpolate(src_t, size=(dh, dw), mode="bicubic", align_corners=False)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / ITERS
        rs_ms = rs_ms_bicubic.get((sw, sh, dw, dh), ms)
        ratio = ms / rs_ms
        cmp = f"{ratio:.2f}× {'slower' if ratio > 1 else 'faster'}"
        print(f"  {case_label(sw,sh,dw,dh):<28}  {ms:>9.3f}  {gb_per_sec_resize(sw,sh,dw,dh,ms):>8.2f}  {cmp:>12}")

bench_cv2_cpu_bicubic()
bench_cv2_cuda_bicubic()
bench_torch_gpu_bicubic()
