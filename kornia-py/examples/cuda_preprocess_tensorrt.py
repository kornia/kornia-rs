"""Zero-copy camera → TensorRT preprocessing with ``CudaPreprocessor``.

The fused preprocessor turns a raw camera frame (NV12 / YUYV / RGB / …) into a
normalized ``[N, 3, H, W]`` model-input ``CudaTensor`` in one kernel launch. The
output stays on the GPU and hands off to an inference engine with **no host
round-trip**:

- ``t.data_ptr``            → bind directly to a TensorRT input
                              (``ctx.set_tensor_address(name, t.data_ptr)``)
- ``t.__cuda_array_interface__`` → cupy / numba
- ``torch.from_dlpack(t)``  → PyTorch (versioned DLPack when negotiated)

For a steady serving loop, preallocate the output once with ``alloc_output`` and
overwrite it every frame with ``run_into`` (no per-frame allocation), and pass
your engine's execution stream so the preprocess is ordered before
``execute_async_v3`` without a host sync.

Run (Linux only — the published wheels always build with the ``cuda`` feature,
so plain ``pip install kornia-rs`` already has this module; the ``[cuda]``
extra just pulls in the ``nvidia-cuda-nvrtc-cu12`` pip package for machines
whose CUDA toolkit doesn't already provide ``libnvrtc``, e.g. driver-only
installs — Jetson/toolkit installs need nothing extra):
    pip install "kornia-rs[cuda]"       # needs an NVIDIA driver; NVRTC via the extra
    python examples/cuda_preprocess_tensorrt.py
"""

import numpy as np

import kornia_rs
from kornia_rs.cuda import Stream


def main() -> None:
    cuda = getattr(kornia_rs, "cuda", None)
    if cuda is None or not cuda.is_available():
        print("CUDA not available (no driver or built without the 'cuda' feature).")
        return

    W, H = 1280, 720          # camera frame size
    OUT_H, OUT_W = 640, 640   # model input size
    rng = np.random.default_rng(0)

    # A raw packed RGB frame as flat uint8 bytes (use format="nv12" for NV12, etc.).
    frame = rng.integers(0, 256, (W * H * 3,), dtype=np.uint8)

    pre = cuda.CudaPreprocessor(
        mode="letterbox",
        format="rgb",
        f16=True,                         # half-precision engine input
        mean=cuda.IMAGENET_MEAN,
        std=cuda.IMAGENET_STD,
    )

    # ---- one-shot: fused preprocess -> CudaTensor -------------------------------
    t = pre.run(frame, W, H, OUT_H, OUT_W)
    print("model input:", t.shape, t.dtype, t.device, "data_ptr", hex(t.data_ptr))

    # Hand the raw device pointer straight to TensorRT (illustrative):
    #   ctx.set_tensor_address("images", t.data_ptr)
    #   ctx.execute_async_v3(stream_handle)

    # ---- serving loop: preallocate once, run_into each frame, on your stream ----
    engine_stream = Stream.default()      # or Stream.from_handle(trt_stream_handle)
    out = pre.alloc_output(OUT_H, OUT_W)  # reused every iteration — zero per-frame alloc
    for _ in range(3):
        frame = rng.integers(0, 256, (W * H * 3,), dtype=np.uint8)
        pre.run_into(out, frame, W, H, stream=engine_stream)
        # engine_stream is now ordered after the preprocess:
        #   ctx.set_tensor_address("images", out.data_ptr)
        #   ctx.execute_async_v3(engine_stream.cuda_stream_ptr)
    engine_stream.synchronize()
    print("run_into output:", out.shape, out.dtype, "reused ptr", hex(out.data_ptr))

    # ---- optional zero-copy handoffs --------------------------------------------
    try:
        import torch

        if torch.cuda.is_available():
            tt = torch.from_dlpack(t)     # zero-copy CUDA tensor
            print("torch:", tuple(tt.shape), tt.dtype, tt.device)
    except ImportError:
        cai = t.__cuda_array_interface__
        print("__cuda_array_interface__:", {k: cai[k] for k in ("shape", "typestr", "stream")})

    print("OK")


if __name__ == "__main__":
    main()
