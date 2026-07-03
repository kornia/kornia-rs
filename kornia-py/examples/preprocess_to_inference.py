"""Camera-format frames → model inference, GPU-resident end-to-end.

Demonstrates the full kornia-rs CUDA pipeline:

    raw NV12 frames (host)
      → CudaPreprocessor.run_batch      one fused kernel per frame:
                                        YUV decode + letterbox resize +
                                        ImageNet normalize + CHW pack (fp16)
      → torch.from_dlpack               ZERO-COPY device handoff
      → ResNet-18 (fp16)                or a TensorRT engine with --engine trt

No intermediate RGB image, no host round-trip, no copy and no dtype cast at
the framework boundary: both engines consume the fused kernel's fp16 CHW
output directly (the TensorRT engine is built with fp16 I/O).

Usage:
    python preprocess_to_inference.py [--batch 4] [--iters 16] [--engine torch|trt]
"""

import argparse
import os
import time

import numpy as np
import torch

import kornia_rs.cuda as krc

SRC_W, SRC_H = 1920, 1080  # camera stand-in resolution


def make_nv12_frames(n: int, w: int, h: int) -> list[np.ndarray]:
    """Deterministic synthetic NV12 frames (stand-in for a camera)."""
    rng = np.random.default_rng(7)
    return [
        rng.integers(0, 256, (w * h * 3 // 2,), dtype=np.uint8) for _ in range(n)
    ]


def build_torch_model() -> torch.nn.Module:
    from torchvision.models import resnet18

    return resnet18().cuda().half().eval()


def build_trt_engine(batch: int, size: int, path: str) -> None:
    """ONNX-export ResNet-18 and build a TensorRT engine with **fp16 I/O**."""
    import tensorrt as trt

    print(f"Building TensorRT engine (fp16 I/O, batch {batch}) → {path} ...")
    onnx_path = path + ".onnx"
    # Export the half model with a half dummy so the network's I/O is fp16 —
    # the engine then binds the preprocessor's output with no reformat pass.
    torch.onnx.export(
        build_torch_model(),
        torch.zeros(batch, 3, size, size, device="cuda", dtype=torch.float16),
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        # Legacy exporter: TRT 10.3's ONNX parser rejects the dynamo
        # exporter's initializer layout on this JetPack.
        dynamo=False,
    )
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            raise RuntimeError(parser.get_error(0))
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    with open(path, "wb") as f:
        f.write(builder.build_serialized_network(network, config))


class TrtRunner:
    """Minimal TensorRT runner fed by torch device tensors (DLPack side)."""

    def __init__(self, batch: int, size: int, cache: str):
        import tensorrt as trt

        # The engine bakes in its shapes, so the cache key must too — a stale
        # engine bound to differently-shaped raw pointers would misbench (or
        # scribble device memory).
        path = f"{cache}.b{batch}s{size}"
        if not os.path.exists(path):
            build_trt_engine(batch, size, path)
        logger = trt.Logger(trt.Logger.WARNING)
        with open(path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.out = torch.empty(
            tuple(self.engine.get_tensor_shape("logits")),
            device="cuda",
            dtype=torch.float16,
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.ctx.set_tensor_address("input", x.data_ptr())
        self.ctx.set_tensor_address("logits", self.out.data_ptr())
        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return self.out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--iters", type=int, default=16, help="timed batches")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--engine", choices=["torch", "trt"], default="torch")
    ap.add_argument(
        "--trt-cache", default="/tmp/kornia_resnet18_trt.engine", help="engine cache path"
    )
    args = ap.parse_args()

    if not krc.is_available():
        print("CUDA not available — nothing to demo.")
        return
    frames = make_nv12_frames(args.batch, SRC_W, SRC_H)

    # Both engines consume fp16 directly — the fused kernel emits the final
    # model input; nothing between the kernel and the network touches pixels.
    pre = krc.CudaPreprocessor(
        mode="letterbox",
        format="nv12",
        f16=True,
        mean=krc.IMAGENET_MEAN,
        std=krc.IMAGENET_STD,
    )
    if args.engine == "torch":
        # Fixed input shape: let cuDNN autotune conv algorithms during warmup.
        torch.backends.cudnn.benchmark = True
        infer = build_torch_model()
    else:
        infer = TrtRunner(args.batch, args.size, args.trt_cache)

    def step() -> torch.Tensor:
        t = pre.run_batch(frames, SRC_W, SRC_H, args.size, args.size)
        x = torch.from_dlpack(t)  # zero-copy: same device memory
        return infer(x)

    with torch.inference_mode():
        # Warmup: JIT/NVRTC, staging allocation, cuDNN autotune / engine load.
        for _ in range(5):
            step()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(args.iters):
            logits = step()
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / args.iters

    n = args.batch
    print(
        f"{args.engine}: {SRC_H}p NV12 x{n} → fused preprocess {args.size}² → "
        f"inference: {dt * 1e3:.2f} ms/batch = {dt / n * 1e3:.2f} ms/frame "
        f"({n / dt:.0f} FPS)"
    )
    print(f"logits: {tuple(logits.shape)} {logits.dtype}, top-1 of frame 0 = "
          f"{int(logits[0].argmax())}")


if __name__ == "__main__":
    main()
