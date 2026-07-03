"""Camera-format frames → model inference, GPU-resident end-to-end.

Demonstrates the full kornia-rs CUDA pipeline:

    raw NV12 frames (host)
      → CudaPreprocessor.run_batch      one fused kernel per frame:
                                        YUV decode + letterbox resize +
                                        ImageNet normalize + CHW pack
      → torch.from_dlpack               ZERO-COPY device handoff
      → ResNet-18 (fp16)                or a TensorRT engine with --engine trt

No intermediate RGB image, no host round-trip, no tensor copy at the
framework boundary.

Usage:
    python preprocess_to_inference.py [--batch 4] [--frames 64] [--engine torch|trt]
"""

import argparse
import time

import numpy as np
import torch

import kornia_rs.cuda as krc

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_nv12_frames(n: int, w: int, h: int, seed: int = 7) -> list[np.ndarray]:
    """Deterministic synthetic NV12 frames (stand-in for a camera)."""
    rng = np.random.default_rng(seed)
    return [
        rng.integers(0, 256, (w * h * 3 // 2,), dtype=np.uint8) for _ in range(n)
    ]


def build_torch_model() -> torch.nn.Module:
    from torchvision.models import resnet18

    model = resnet18().cuda().half().eval()
    return model


class TrtRunner:
    """Minimal TensorRT runner fed by torch device tensors (DLPack side)."""

    def __init__(self, batch: int, size: int, cache: str):
        import os

        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        if not os.path.exists(cache):
            print(f"Building TensorRT engine (fp16, batch {batch}) → {cache} ...")
            onnx_path = cache + ".onnx"
            model = build_torch_model().float()  # export in fp32; TRT does fp16
            torch.onnx.export(
                model,
                torch.zeros(batch, 3, size, size, device="cuda"),
                onnx_path,
                input_names=["input"],
                output_names=["logits"],
                # Legacy exporter: TRT 10.3's ONNX parser rejects the dynamo
                # exporter's initializer layout on this JetPack.
                dynamo=False,
            )
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
            engine_bytes = builder.build_serialized_network(network, config)
            with open(cache, "wb") as f:
                f.write(engine_bytes)
        runtime = trt.Runtime(logger)
        with open(cache, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.out = torch.empty(batch, 1000, device="cuda", dtype=torch.float32)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # TRT engine input is fp32 (network was parsed fp32; fp16 is internal).
        x = x.float().contiguous()
        self.ctx.set_tensor_address("input", x.data_ptr())
        self.ctx.set_tensor_address("logits", self.out.data_ptr())
        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return self.out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--engine", choices=["torch", "trt"], default="torch")
    ap.add_argument(
        "--trt-cache", default="/tmp/kornia_resnet18_trt.engine", help="engine cache path"
    )
    args = ap.parse_args()

    if not krc.is_available():
        print("CUDA not available — nothing to demo.")
        return
    w, h = 1920, 1080
    frames = make_nv12_frames(args.batch, w, h)

    # fp16 CHW tensors straight out of the fused kernel for the torch path.
    use_f16 = args.engine == "torch"
    pre = krc.CudaPreprocessor(
        mode="letterbox",
        format="nv12",
        f16=use_f16,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    )

    if args.engine == "torch":
        model = build_torch_model()

        def infer(x: torch.Tensor) -> torch.Tensor:
            return model(x)

    else:
        infer = TrtRunner(args.batch, args.size, args.trt_cache)

    def step() -> torch.Tensor:
        t = pre.run_batch(frames, w, h, args.size, args.size)
        x = torch.from_dlpack(t)  # zero-copy: same device memory
        with torch.no_grad():
            return infer(x)

    # Warmup (JIT, cuDNN autotune / engine load).
    for _ in range(5):
        logits = step()
    torch.cuda.synchronize()

    iters = max(1, args.frames // args.batch)
    t0 = time.perf_counter()
    for _ in range(iters):
        logits = step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters

    n = args.batch
    print(
        f"{args.engine}: 1080p NV12 x{n} → fused preprocess {args.size}² → "
        f"inference: {dt * 1e3:.2f} ms/batch = {dt / n * 1e3:.2f} ms/frame "
        f"({n / dt:.0f} FPS)"
    )
    print(f"logits: {tuple(logits.shape)} {logits.dtype}, top-1 of frame 0 = "
          f"{int(logits[0].argmax())}")


if __name__ == "__main__":
    main()
