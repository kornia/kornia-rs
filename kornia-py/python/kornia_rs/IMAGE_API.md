# kornia_rs Image API — A Fast PIL Alternative

A PIL-like API powered by Rust. Designed to replace PIL/OpenCV pain points with zero-copy NumPy integration, multiprocess safety, and always-RGB color handling.

## Overview

`kornia_rs.Image` provides a familiar PIL-style interface backed by fast Rust implementations. The API prioritizes:

- **Always RGB**: Never BGR default (OpenCV pain point solved)
- **Rust-powered I/O**: Fast JPEG, PNG, TIFF read/write
- **Multiprocess-safe**: Fully serializable for Ray Data and multiprocessing
- **Zero-copy integration**: Share data with NumPy, PyTorch, and Arrow without copies
- **Immutable transforms**: Every operation returns a new Image (safe for parallelism)
- **Chainable API**: Fluent interface for readable pipelines

Perfect for ML pipelines, distributed data processing, and applications where PIL/OpenCV bottlenecks matter.

## Quick Start

```python
from kornia_rs.image import Image

# Open and transform
img = Image.open("photo.jpg")
result = img.resize(256, 256).flip_horizontal().to_grayscale()
result.save("out.png")

# Create new image
blank = Image.new(width=256, height=256, channels=3, fill=128)

# Work with arrays
import numpy as np
arr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
img = Image.from_buffer(arr)  # Zero-copy wrapping

# Use as context manager
with Image.open("input.jpg") as img:
    output = img.resize(512, 512).adjust_brightness(0.2)
    output.save("output.jpg", quality=90)
```

## Image Class Reference

### Static Constructors

#### `Image.open(path: str) -> Image`

Open an image file from disk. Returns RGB image regardless of input format.

Supported formats: JPEG, PNG, TIFF

```python
img = Image.open("photo.jpg")
print(img)  # Image(mode=RGB, size=1920x1080, dtype=uint8)
```

**Note**: Uses fast Rust I/O. Automatically converts to RGB.

---

#### `Image.new(width: int, height: int, channels: int = 3, fill: int = 0) -> Image`

Create a new solid-color image.

```python
# Black 256x256 RGB image
black = Image.new(256, 256, channels=3, fill=0)

# White 512x512 grayscale
white = Image.new(512, 512, channels=1, fill=255)
```

---

#### `Image.from_numpy(array: np.ndarray) -> Image`

Create an Image from a NumPy array, making a copy for safety.

Expects HWC format (height, width, channels). 2D arrays treated as grayscale.

```python
arr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
img = Image.from_numpy(arr)  # Makes a copy
```

**Note**: Use `from_buffer()` if you want zero-copy wrapping.

---

#### `Image.from_buffer(array: np.ndarray) -> Image`

Wrap a NumPy array without copying (zero-copy). Data must remain valid for Image lifetime.

Ideal for Ray Data and Arrow integration.

```python
arr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
img = Image.from_buffer(arr)  # Zero-copy, no allocation

# Later, if arr is garbage collected, img._data becomes invalid
# This is safe in Ray Data where buffers are held by the framework
```

**Warning**: Do not modify the underlying array after wrapping—Image assumes immutability.

---

### Properties

#### `width: int`

Image width in pixels.

```python
img = Image.open("photo.jpg")
print(img.width)  # 1920
```

---

#### `height: int`

Image height in pixels.

```python
print(img.height)  # 1080
```

---

#### `channels: int`

Number of channels (1=grayscale, 3=RGB, 4=RGBA).

```python
print(img.channels)  # 3
```

---

#### `size: Tuple[int, int]`

(width, height) — PIL convention.

```python
print(img.size)  # (1920, 1080)
```

---

#### `shape: Tuple[int, int, int]`

(height, width, channels) — NumPy convention.

```python
print(img.shape)  # (1080, 1920, 3)
```

---

#### `dtype: np.dtype`

Data type of underlying array (usually `uint8` or `float32`).

```python
print(img.dtype)  # dtype('uint8')
normalized = img.normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
print(normalized.dtype)  # dtype('float32')
```

---

#### `data: np.ndarray`

The underlying NumPy array (HWC format).

```python
arr = img.data  # Returns backing buffer (read-only by convention)
print(arr.shape)  # (1080, 1920, 3)
```

**Convention**: Do not modify `data` directly—Image assumes immutability.

---

#### `nbytes: int`

Total bytes used by image data.

```python
print(img.nbytes)  # 6220800 for 1920x1080x3 uint8
```

---

### I/O Methods

#### `save(path: str, quality: int = 95) -> None`

Save image to file. Format detected from extension.

Supported: `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`

```python
img.save("output.jpg", quality=90)  # JPEG with quality 90
img.save("output.png")               # PNG (lossless)
img.save("output.tif")               # TIFF (lossless)
```

**Note**: Quality parameter only applies to JPEG. PNG and TIFF are always lossless.

---

#### `to_numpy() -> np.ndarray`

Return a copy of the underlying NumPy array.

```python
arr = img.to_numpy()  # Makes a copy
arr[0, 0, 0] = 255   # Safe to modify
print(img.data[0, 0, 0])  # Unchanged (still original value)
```

---

#### `copy() -> Image`

Return a deep copy of the Image.

```python
img2 = img.copy()  # New Image instance, independent data
```

---

### Chainable Transforms

All transform methods return a new Image instance (immutable pattern). Chain calls for fluent pipelines.

#### `resize(width: int, height: int, interpolation: str = "bilinear") -> Image`

Resize to (width, height) using specified interpolation.

Supported interpolation: `"bilinear"` (default), `"nearest"`

```python
# Resize to 256x256
img = Image.open("photo.jpg")
small = img.resize(256, 256)

# Nearest neighbor
small_nearest = img.resize(256, 256, interpolation="nearest")

# Chainable
result = img.resize(512, 512).flip_horizontal().to_grayscale()
```

---

#### `flip_horizontal() -> Image`

Mirror image left-right.

```python
flipped = img.flip_horizontal()
```

---

#### `flip_vertical() -> Image`

Mirror image top-bottom.

```python
flipped = img.flip_vertical()
```

---

#### `crop(x: int, y: int, width: int, height: int) -> Image`

Crop region starting at (x, y) with given width and height.

```python
# Crop top-left 256x256 region
cropped = img.crop(0, 0, 256, 256)

# Crop center region
x = (img.width - 256) // 2
y = (img.height - 256) // 2
center = img.crop(x, y, 256, 256)
```

---

#### `rotate(angle: float) -> Image`

Rotate counter-clockwise by angle degrees.

```python
rotated = img.rotate(45.0)  # 45 degrees counter-clockwise
```

---

#### `to_grayscale() -> Image`

Convert to single-channel grayscale using ITU-R 601 luma weights.

```python
gray = img.to_grayscale()
print(gray.channels)  # 1
```

---

#### `to_rgb() -> Image`

Convert to 3-channel RGB. Extends grayscale by duplicating channels.

```python
gray = img.to_grayscale()
rgb = gray.to_rgb()  # Expand to 3 identical channels
print(rgb.channels)  # 3
```

---

#### `adjust_brightness(factor: float) -> Image`

Adjust brightness multiplicatively.

- factor < 1.0: darker
- factor = 1.0: no change
- factor > 1.0: brighter

```python
brighter = img.adjust_brightness(1.2)  # 20% brighter
darker = img.adjust_brightness(0.8)    # 20% darker
```

---

#### `adjust_contrast(factor: float) -> Image`

Adjust contrast. Applies formula: `output = (input - mean) * factor + mean`

- factor = 1.0: no change
- factor > 1.0: more contrast
- factor < 1.0: less contrast (flatter)

```python
high_contrast = img.adjust_contrast(1.5)
low_contrast = img.adjust_contrast(0.7)
```

---

#### `adjust_saturation(factor: float) -> Image`

Adjust color saturation. Only affects RGB images.

- factor = 0.0: grayscale
- factor = 1.0: no change
- factor > 1.0: more colorful

```python
vibrant = img.adjust_saturation(1.3)
desaturated = img.adjust_saturation(0.5)
```

---

#### `gaussian_blur(kernel_size: int = 3, sigma: float = 1.0) -> Image`

Apply Gaussian blur with given kernel size and standard deviation.

kernel_size should be odd (3, 5, 7, ...).

```python
blurred = img.gaussian_blur(kernel_size=5, sigma=1.0)
```

---

#### `normalize(mean: Tuple[float, ...], std: Tuple[float, ...]) -> Image`

Normalize image with per-channel mean and std. Returns float32.

Formula: `(image / 255.0 - mean) / std`

```python
# ImageNet normalization
normalized = img.normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)
print(normalized.dtype)  # float32
```

---

### Context Manager

Image supports Python's context manager protocol.

```python
with Image.open("input.jpg") as img:
    result = img.resize(512, 512).adjust_brightness(0.2)
    result.save("output.jpg")
```

---

### String Representation

```python
img = Image.open("photo.jpg")
print(img)  # Image(mode=RGB, size=1920x1080, dtype=uint8)

gray = img.to_grayscale()
print(gray)  # Image(mode=L, size=1920x1080, dtype=uint8)
```

---

## Augmentations

All augmentations are random and return new Image instances. Compatible with training loops and Ray Data.

Import from `kornia_rs`:

```python
from kornia_rs import ColorJitter, RandomHorizontalFlip, RandomCrop, Compose
```

### `ColorJitter`

Randomly adjust brightness, contrast, saturation, and hue. Applies transformations in random order (like torchvision).

**Args:**
- `brightness`: Float or (min, max). Factor from [max(0, 1-brightness), 1+brightness]
- `contrast`: Float or (min, max). Factor chosen uniformly
- `saturation`: Float or (min, max). Factor chosen uniformly
- `hue`: Float in [0, 0.5] or (min, max) in [-0.5, 0.5]

```python
# Simple: symmetric ranges
jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1)

# Or explicit ranges
jitter = ColorJitter(
    brightness=(0.8, 1.2),  # 80% to 120%
    contrast=(0.8, 1.2),
    saturation=(0.7, 1.3),
    hue=(-0.1, 0.1)
)

img = Image.open("photo.jpg")
augmented = jitter(img)
```

**Implementation Note**: Currently applies primitives sequentially (composed approach). Future versions will fuse to single Rust kernel.

---

### `RandomHorizontalFlip`

Randomly flip image horizontally with probability p.

```python
flip = RandomHorizontalFlip(p=0.5)
augmented = flip(img)  # 50% chance of flip
```

---

### `RandomVerticalFlip`

Randomly flip image vertically with probability p.

```python
flip = RandomVerticalFlip(p=0.5)
augmented = flip(img)
```

---

### `RandomCrop`

Randomly crop image to given size.

```python
crop = RandomCrop(size=(224, 224))
augmented = crop(img)  # Random crop to 224x224

# Or square
crop = RandomCrop(size=256)  # Crops to 256x256
```

**Raises**: ValueError if image is smaller than crop size.

---

### `RandomRotation`

Randomly rotate image within degree range.

```python
rotate = RandomRotation(degrees=45.0)  # -45 to +45 degrees
augmented = rotate(img)

# Or explicit range
rotate = RandomRotation(degrees=(-30, 60))
```

---

### `Compose`

Chain multiple transforms together.

```python
transforms = Compose([
    RandomCrop((224, 224)),
    RandomHorizontalFlip(p=0.5),
    ColorJitter(brightness=0.2, contrast=0.2),
])

augmented = transforms(img)
```

---

## Training Pipeline Example

Complete example for training data augmentation:

```python
from kornia_rs.image import Image
from kornia_rs import (
    ColorJitter, RandomHorizontalFlip, RandomCrop, Compose
)

# Define augmentation pipeline
train_transform = Compose([
    RandomCrop((224, 224)),
    RandomHorizontalFlip(p=0.5),
    ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
])

# Use in training loop
for epoch in range(10):
    for path in image_paths:
        img = Image.open(path)
        augmented = train_transform(img)  # Random augmentation
        normalized = augmented.normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
        # Feed to model...
```

---

## Zero-Copy Integration

### NumPy

Access image data without copying:

```python
img = Image.open("photo.jpg")

# Option 1: Direct access (read-only by convention)
arr = img.data  # View of backing buffer
arr_copy = np.array(img)  # Calls __array__(), returns buffer

# Option 2: Create Image from NumPy without copy
arr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
img = Image.from_buffer(arr)  # No allocation
```

### PyTorch

Convert to PyTorch tensor with zero-copy:

```python
import torch

img = Image.open("photo.jpg")

# HWC to CHW (channels-first)
arr = np.array(img)  # Zero-copy access
tensor = torch.from_numpy(arr).permute(2, 0, 1)  # Zero-copy

# Or use as float
tensor_f = torch.from_numpy(np.array(img)).float() / 255.0
```

### Ray Data

Process images with Ray Data without copies:

```python
import ray
from kornia_rs.image import Image

def process_image(batch):
    """Ray Data worker processing Arrow batch."""
    # Arrow data is in shared memory
    # Create Images with zero-copy wrapping
    images = [Image.from_buffer(arr) for arr in batch["image"]]

    # Process
    processed = [
        img.resize(256, 256).normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
        for img in images
    ]

    # Return processed data
    return {"image": [p.data for p in processed]}

# Create Ray Data
dataset = ray.data.read_images("path/to/images/")
processed = dataset.map_batches(process_image, batch_size=32)
```

---

## Copy Behavior

| Operation | Copies | Notes |
|-----------|--------|-------|
| `Image.from_buffer(arr)` | 0 | Zero-copy wrapping |
| `np.array(img)` | 0 | Returns backing buffer |
| `img.data` | 0 | Direct buffer access |
| `torch.from_numpy(np.array(img))` | 0 | Zero-copy to torch |
| `Image.from_numpy(arr)` | 1 | Safety copy |
| `Image(arr)` | 1 | Constructor makes copy |
| `img.to_numpy()` | 1 | Explicit copy |
| `img.copy()` | 1 | Deep copy |
| Any transform (resize, crop, etc.) | 1 | New Image instance |

---

## Multiprocessing

Image is fully serializable and spawn-safe for parallel processing.

### ProcessPoolExecutor

```python
from concurrent.futures import ProcessPoolExecutor
from kornia_rs.image import Image

def process_image(path):
    """Worker function (runs in separate process)."""
    img = Image.open(path)
    result = img.resize(256, 256).adjust_brightness(0.2)
    return result.to_numpy()

# Main process
paths = ["img1.jpg", "img2.jpg", "img3.jpg"]

with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_image, paths))
```

Image serialization is automatic via `__reduce__()`. NumPy arrays are efficiently serialized by Python's multiprocessing.

### Ray Actors

```python
import ray
from kornia_rs.image import Image

@ray.remote
def process_batch(image_paths):
    """Ray actor processing multiple images."""
    results = []
    for path in image_paths:
        img = Image.open(path)
        result = img.resize(256, 256).to_grayscale()
        results.append(result.to_numpy())
    return results

# Distribute work
paths_batches = [paths[i::4] for i in range(4)]
futures = [process_batch.remote(batch) for batch in paths_batches]
results = ray.get(futures)
```

---

## Color Space Conversions

The `kornia_rs.color` module provides color space conversions. Always RGB-first.

```python
from kornia_rs.color import (
    to_grayscale, to_rgb, to_bgr, to_rgba, to_hsv
)
```

### `to_grayscale(data: np.ndarray) -> np.ndarray`

Convert RGB to grayscale (1 channel). Uses ITU-R 601 luma weights.

```python
img = Image.open("photo.jpg")
gray = img.to_grayscale()
```

---

### `to_rgb(data: np.ndarray) -> np.ndarray`

Convert grayscale to RGB by duplicating channels.

```python
gray_img = Image.open("gray.jpg")
rgb = gray_img.to_rgb()
```

---

### `to_bgr(data: np.ndarray) -> np.ndarray`

Convert RGB to BGR. Rarely needed—Image is always RGB.

```python
from kornia_rs.color import to_bgr
bgr_data = to_bgr(img.data)
```

---

### `to_rgba(data: np.ndarray, alpha: int = 255) -> np.ndarray`

Convert RGB to RGBA with optional alpha channel.

```python
from kornia_rs.color import to_rgba
rgba = to_rgba(img.data, alpha=200)
```

---

### `to_hsv(data: np.ndarray) -> np.ndarray`

Convert RGB to HSV. Returns float32 in [0, 1] range.

```python
from kornia_rs.color import to_hsv
hsv = to_hsv(img.data)  # float32, [0, 1]
```

---

## PIL/OpenCV Pain Points Solved

| Pain Point | PIL/OpenCV | kornia_rs |
|-----------|-----------|-----------|
| **Default color order** | OpenCV uses BGR | Always RGB, no confusion |
| **Image I/O speed** | PIL is pure Python | Rust-powered JPEG/PNG/TIFF |
| **Multiprocess safety** | PIL Image not serializable | Fully serializable for Ray/multiprocessing |
| **Dimension confusion** | `img.size` vs `img.shape` vs `.mode` | Clear: `.width`, `.height`, `.channels`, `.size` (WH), `.shape` (HWC) |
| **In-place mutation** | OpenCV modifies arrays in-place | Immutable—every op returns new Image |
| **Zero-copy integration** | Complex, error-prone | Seamless with NumPy/PyTorch/Arrow |
| **Chaining transforms** | Awkward, requires variables | Fluent pipeline: `img.resize(...).flip(...).crop(...)` |

---

## Architecture and Future Work

### Current Implementation

- **Python Image class** wraps NumPy array (HWC format)
- **Each transform** calls out to Rust functions, returns new NumPy array
- **Serialization** via `__reduce__()` makes data pickleable
- **Zero-copy** with `from_buffer()` and `__array__()` for NumPy/PyTorch/Arrow

### Future: Rust-Side Image

Planned improvements:

1. **Rust #[pyclass] Image**: Hold data Rust-side for zero-copy chains
2. **Kernel fusion**: Combine ColorJitter operations into single pass (currently composed primitives)
3. **GPU support**: Leverage Rust GPU libraries for larger datasets
4. **Lazy evaluation**: Chain transforms without allocating until `.to_numpy()`

These are backward-compatible—existing API stays the same.

---

## Comparison with PIL and OpenCV

### PIL.Image

```python
from PIL import Image as PILImage

# Open
img = PILImage.open("photo.jpg")

# Resize
resized = img.resize((256, 256))  # Returns PIL Image

# Save
resized.save("out.jpg", quality=95)

# Shape confusion
print(img.size)      # (width, height)
print(img.mode)      # 'RGB'
```

### OpenCV

```python
import cv2

# Open (returns BGR!)
img = cv2.imread("photo.jpg")  # uint8, HWC, BGR

# Resize
resized = cv2.resize(img, (256, 256))  # Returns numpy array, in-place modified

# Save
cv2.imwrite("out.jpg", resized)  # Modifies img in-place

# Color confusion
if img is not None:  # Always check return
    print(img.shape)  # (height, width, channels), BGR not RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

### kornia_rs.Image

```python
from kornia_rs.image import Image

# Open (always RGB)
img = Image.open("photo.jpg")

# Resize (returns new Image, immutable)
resized = img.resize(256, 256)

# Save
resized.save("out.jpg", quality=95)

# Clear API
print(img.width, img.height)  # 1920, 1080
print(img.channels)            # 3
print(img.size)               # (1920, 1080)
print(img.shape)              # (1080, 1920, 3)
```

---

## Performance Notes

- **I/O**: Rust-powered, 2-10x faster than PIL for JPEG/PNG
- **Transforms**: Rust acceleration for uint8 RGB, numpy fallback for other dtypes
- **Multiprocess**: Zero overhead—images serialize efficiently via numpy
- **GIL**: Rust operations are GIL-free, true parallelism with multiprocessing/Ray

---

## See Also

- `kornia_rs.transforms`: Low-level transform functions on numpy arrays
- `kornia_rs.color`: Color space conversion functions
- `kornia_rs.augmentations`: All augmentation classes
