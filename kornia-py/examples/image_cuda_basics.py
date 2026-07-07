"""Device-aware ``Image`` basics: NumPy/PyTorch-centric CUDA in kornia-rs.

A device ``Image`` is the *same* ``Image`` type as a host one — only its
``.device`` differs. There is no separate CudaImage / upload / download: you
create device images with ``Image.cuda.from_numpy`` (or move an existing image
with ``.to_cuda()``), bring data back with ``.numpy()`` / ``.cpu()``, and share
it zero-copy with torch / cupy / cuda-python via DLPack and the CUDA Array
Interface.

Run:
    pip install "kornia-rs[cuda]"       # needs an NVIDIA driver + NVRTC
    python examples/image_cuda_basics.py
"""

import numpy as np

import kornia_rs
from kornia_rs.image import Image, Stream


def main() -> None:
    cuda = getattr(kornia_rs, "cuda", None)
    if cuda is None or not cuda.is_available():
        print("CUDA not available (no driver or built without the 'cuda' feature).")
        return

    a = np.random.default_rng(0).integers(0, 256, (240, 320, 3), dtype=np.uint8)

    # 1) Host numpy -> device Image (zero-copy the host buffer, then H2D).
    img = Image.cuda.from_numpy(a)
    print("device:", img.device, "shape:", img.shape, "dtype:", img.dtype)

    # 2) .numpy() auto-copies a device image back to host (D2H).
    assert np.array_equal(img.numpy(), a)

    # 3) Move an existing host Image to the GPU and back.
    host = Image.from_numpy(a)
    dev = host.to_cuda()
    assert dev.device == "cuda:0"
    assert np.array_equal(dev.cpu().numpy(), a)

    # 4) GPU color conversion — takes and returns a device Image.
    gray = cuda.gray_from_rgb(img)
    print("gray:", gray.device, gray.channels, "channels")

    # 5) Allocate directly on device.
    zeros = Image.cuda.zeros(width=64, height=48, channels=3, dtype="uint8")
    assert zeros.numpy().sum() == 0

    # 6) Explicit CUDA stream (optional; defaults to the process default stream).
    stream = Stream.default()
    img2 = Image.cuda.from_numpy(a, stream=stream)
    stream.synchronize()
    print("stream handle:", hex(stream.cuda_stream_ptr))

    # 7) Zero-copy interop with PyTorch via DLPack (no host round-trip).
    try:
        import torch

        if torch.cuda.is_available():
            t = torch.from_dlpack(img)  # device Image -> CUDA torch tensor
            print("torch:", t.shape, t.dtype, t.device)
            # ...and back, with device inferred from the tensor.
            img3 = Image.from_dlpack(t)
            assert img3.device == "cuda:0"
    except ImportError:
        pass

    # 8) Zero-copy interop with cupy / numba / cuda-python via the CUDA Array
    #    Interface (device images expose __cuda_array_interface__).
    try:
        import cupy as cp

        arr = cp.asarray(img)  # zero-copy view of the device buffer
        print("cupy:", arr.shape, arr.dtype, "ptr", hex(arr.data.ptr))
    except ImportError:
        cai = img.__cuda_array_interface__
        print("__cuda_array_interface__:", {k: cai[k] for k in ("shape", "typestr", "version")})

    print("OK")


if __name__ == "__main__":
    main()
