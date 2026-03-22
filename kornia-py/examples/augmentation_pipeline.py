"""Example: Efficient data augmentation pipeline with kornia_rs.

Shows how to use the Image API for augmentations, and zero-copy
conversion to numpy and PyTorch tensors.
"""

import numpy as np
import time
from kornia_rs.image import Image
from kornia_rs.augmentations import ColorJitter, RandomHorizontalFlip, RandomCrop, Compose


# --- 1. Define an augmentation pipeline ---

train_transform = Compose([
    RandomCrop((224, 224)),
    RandomHorizontalFlip(p=0.5),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
])


# --- 2. Efficient augmentation on a batch ---

def augment_batch(images: list, transform: Compose) -> list:
    """Apply augmentations to a batch of Image objects."""
    return [transform(img) for img in images]


# --- 3. Zero-copy to numpy ---

def to_numpy_batch(images: list) -> np.ndarray:
    """Convert list of Image to a numpy batch (N, H, W, C).

    Uses np.array(img) which calls __array__ — returns the underlying
    numpy buffer WITHOUT copying.
    """
    return np.stack([np.array(img) for img in images])


# --- 4. Zero-copy to PyTorch (when available) ---

def to_torch_batch(images: list):
    """Convert list of Image to a PyTorch tensor (N, C, H, W).

    Uses torch.from_numpy for zero-copy, then permutes to CHW.
    """
    import torch
    # np.array(img) returns the backing buffer (no copy)
    # torch.from_numpy shares memory with numpy (no copy)
    # .permute rearranges dimensions (no copy, just a view)
    tensors = [torch.from_numpy(np.array(img)).permute(2, 0, 1) for img in images]
    return torch.stack(tensors)  # this copies into contiguous batch tensor


# --- 5. Image.from_buffer for zero-copy ingestion ---

def from_ray_data_batch(numpy_batch: np.ndarray) -> list:
    """Convert a Ray Data numpy batch back to Image objects.

    Image.from_buffer avoids copying — the Image wraps the same
    memory as the numpy array.
    """
    return [Image.frombytes(numpy_batch[i]) for i in range(len(numpy_batch))]


# --- Demo ---

if __name__ == "__main__":
    # Create synthetic dataset of 100 images (256x256 RGB)
    N = 100
    dataset = [Image.frombytes(np.full((256, 256, 3), i % 256, dtype=np.uint8)) for i in range(N)]

    # Warm up
    _ = augment_batch(dataset[:2], train_transform)

    # Benchmark augmentation
    start = time.perf_counter()
    augmented = augment_batch(dataset, train_transform)
    aug_time = time.perf_counter() - start
    print(f"Augmented {N} images in {aug_time:.3f}s ({N/aug_time:.0f} img/s)")

    # Benchmark zero-copy to numpy batch
    start = time.perf_counter()
    batch = to_numpy_batch(augmented)
    np_time = time.perf_counter() - start
    print(f"To numpy batch: {batch.shape} in {np_time:.6f}s")

    # Check zero-copy: Image.data and np.array(img) share memory
    img = augmented[0]
    arr = np.array(img)
    print(f"Zero-copy check: shares memory = {np.shares_memory(img.data, arr)}")

    # Try PyTorch if available
    try:
        import torch
        start = time.perf_counter()
        tensor_batch = to_torch_batch(augmented)
        torch_time = time.perf_counter() - start
        print(f"To torch batch: {tensor_batch.shape} in {torch_time:.6f}s")

        # Zero-copy check: numpy -> torch
        single_arr = np.array(augmented[0])
        single_tensor = torch.from_numpy(single_arr)
        single_arr[0, 0, 0] = 42
        print(f"Torch zero-copy check: tensor[0,0,0] = {single_tensor[0, 0, 0]} (should be 42)")
    except ImportError:
        print("PyTorch not available, skipping torch conversion demo")

    # from_buffer zero-copy round-trip
    numpy_batch = to_numpy_batch(augmented)
    recovered = from_ray_data_batch(numpy_batch)
    print(f"from_buffer round-trip: {len(recovered)} images, "
          f"shares memory = {np.shares_memory(numpy_batch[0], recovered[0].data)}")
