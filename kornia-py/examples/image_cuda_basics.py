"""Device-aware ``Image`` basics: NumPy/PyTorch-centric CUDA in kornia-rs.

A device ``Image`` is the *same* ``Image`` type as a host one — only its
``.device`` differs. There is no separate CudaImage / upload / download: you
create device images by moving a host image to the GPU with ``.to_cuda`` (an existing image
with ``.to_cuda()``), bring data back with ``.numpy()`` / ``.cpu()``, and share
it zero-copy with torch / cupy / cuda-python via DLPack and the CUDA Array
Interface.

Run:
    pip install "kornia-rs[cuda]"       # needs an NVIDIA driver + NVRTC
    python examples/image_cuda_basics.py
"""

import numpy as np

import kornia_rs
from kornia_rs.image import Image
from kornia_rs.cuda import Stream


def main() -> None:
    cuda = getattr(kornia_rs, "cuda", None)
    if cuda is None or not cuda.is_available():
        print("CUDA not available (no driver or built without the 'cuda' feature).")
        return

    a = np.random.default_rng(0).integers(0, 256, (240, 320, 3), dtype=np.uint8)

    # 1) Host numpy -> device Image (zero-copy the host buffer, then H2D).
    img = Image.from_numpy(a).to_cuda()
    print("device:", img.device, "shape:", img.shape, "dtype:", img.dtype)

    # 2) .numpy() auto-copies a device image back to host (D2H).
    assert np.array_equal(img.numpy(), a)

    # 3) Move an existing host Image to the GPU and back.
    host = Image.from_numpy(a)
    dev = host.to_cuda()
    assert dev.device == "cuda:0"
    assert np.array_equal(dev.cpu().numpy(), a)

    # 4) Color conversion dispatches on residency: a device Image runs on the GPU
    #    (returns a device Image); the same imgproc op takes a numpy array on CPU.
    gray = kornia_rs.imgproc.gray_from_rgb(img)
    print("gray:", gray.device, gray.channels, "channels")

    # 5) Allocate directly on device.
    zeros = Image.zeros(width=64, height=48, channels=3, dtype="uint8", stream=Stream.default())
    assert zeros.numpy().sum() == 0

    # 6) Explicit CUDA stream (optional; defaults to the process default stream).
    stream = Stream.default()
    img2 = Image.from_numpy(a).to_cuda(stream)
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

    # 8) Zero-copy interop with cupy / numba / cuda-python, both directions.
    try:
        import cupy as cp

        # device Image -> cupy: via the CUDA Array Interface (cp.asarray reads it).
        arr = cp.asarray(img)  # zero-copy view of the device buffer
        print("cupy:", arr.shape, arr.dtype, "ptr", hex(arr.data.ptr))
        # cupy -> device Image: cupy implements __dlpack__, so the same universal
        # Image.from_dlpack used for torch above works unchanged (device inferred).
        img4 = Image.from_dlpack(arr)
        assert img4.device == "cuda:0"
        print("cupy -> Image:", img4.device, img4.shape)
    except ImportError:
        cai = img.__cuda_array_interface__
        print("__cuda_array_interface__:", {k: cai[k] for k in ("shape", "typestr", "version")})

    print("OK")


if __name__ == "__main__":
    main()
