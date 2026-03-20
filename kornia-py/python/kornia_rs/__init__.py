"""kornia_rs — Fast computer vision in Rust, with a PIL-like Python API.

A high-performance image processing library combining Rust speed with Python ergonomics.
Designed to replace PIL/OpenCV with zero-copy NumPy/PyTorch integration and multiprocess safety.

Always RGB (never BGR). Fully serializable for Ray Data and distributed processing.
GIL-free Rust operations enable true parallelism.

Quick Start
-----------

    from kornia_rs.image import Image

    img = Image.load("photo.jpg")
    result = img.resize(256, 256).flip_horizontal().to_grayscale()
    result.save("out.png")

Key Features
------------

- **PIL-like API**: Familiar interface from Image.load() to .save()
- **Always RGB**: No BGR confusion (OpenCV pain point solved)
- **Multiprocess-safe**: Serializable for Ray Data and multiprocessing
- **Zero-copy**: NumPy, PyTorch, and Arrow integration without copies
- **Immutable**: Every operation returns a new Image (safe for parallelism)
- **Fast I/O**: Rust-powered JPEG, PNG, TIFF read/write (2-10x faster than PIL)
- **Chainable**: Fluent transforms: img.resize(...).flip(...).crop(...)

Modules
-------

- **kornia_rs.image**: Image class and image operations
- **kornia_rs.transforms**: Low-level transform functions on numpy arrays
- **kornia_rs.color**: Color space conversions (RGB/HSV/BGR/RGBA/etc)
- **kornia_rs.augmentations**: Random augmentations for ML training

Augmentations
-------------

    from kornia_rs import ColorJitter, RandomHorizontalFlip, RandomCrop, Compose

    transforms = Compose([
        RandomCrop((224, 224)),
        RandomHorizontalFlip(p=0.5),
        ColorJitter(brightness=0.2, contrast=0.2),
    ])

    img = Image.load("photo.jpg")
    augmented = transforms(img)

Zero-Copy Integration
---------------------

NumPy, PyTorch, and Ray Data can access image data without copying:

    import numpy as np
    import torch

    img = Image.load("photo.jpg")

    # NumPy: zero-copy access
    arr = np.array(img)  # No allocation

    # PyTorch: zero-copy tensor
    tensor = torch.from_numpy(arr).permute(2, 0, 1)

    # Ray Data: zero-copy wrapping from Arrow
    img_from_buffer = Image.frombytes(arrow_array)

Multiprocessing
---------------

Image is fully serializable for ProcessPoolExecutor and Ray:

    from concurrent.futures import ProcessPoolExecutor
    from kornia_rs.image import Image

    def process(path):
        return Image.load(path).resize(256, 256).to_grayscale().to_numpy()

    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process, image_paths))

Documentation
--------------

For complete API documentation, see IMAGE_API.md in this directory.

PIL/OpenCV Comparison
---------------------

| Feature | PIL | OpenCV | kornia_rs |
|---------|-----|--------|-----------|
| Default color | RGB | BGR | RGB |
| I/O speed | Slow | Medium | Fast (Rust) |
| Multiprocess | No | No | Yes |
| Serializable | No | No | Yes |
| GIL-free ops | No | No | Yes |
| Zero-copy NumPy | Complex | No | Yes |
| Immutable | No | No | Yes |
| Chaining | No | No | Yes |
"""

# Re-export everything from the compiled Rust module
from kornia_rs.kornia_rs import *  # noqa: F401, F403
from kornia_rs.kornia_rs import __version__  # noqa: F401

# Import submodules from compiled module so kornia_rs.io / kornia_rs.imgproc work
from kornia_rs import kornia_rs as _native  # noqa: F401
import sys as _sys

# Make compiled submodules accessible as kornia_rs.io, kornia_rs.imgproc, etc.
for _submod_name in ('io', 'imgproc', 'image', 'k3d', 'apriltag'):
    _submod = getattr(_native, _submod_name, None)
    if _submod is not None:
        _sys.modules[f'kornia_rs.{_submod_name}'] = _submod

# Inject the PIL-like Image class into kornia_rs.image submodule
# so that `from kornia_rs.image import Image` works
from kornia_rs._image import Image as _Image  # noqa: F401
_image_mod = _sys.modules.get('kornia_rs.image')
if _image_mod is not None:
    _image_mod.Image = _Image
del _Image

# Import augmentations
from kornia_rs.augmentations import (  # noqa: F401
    ColorJitter,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomCrop,
    RandomRotation,
    Compose,
)
