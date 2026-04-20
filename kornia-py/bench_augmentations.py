"""Benchmark kornia-rs augmentations vs albumentations/OpenCV."""
import json
import time
import numpy as np
from kornia_rs.image import Image
from kornia_rs.augmentations import ColorJitter, RandomHorizontalFlip, RandomVerticalFlip, RandomCrop, RandomRotation

import albumentations as A
import cv2


def bench(name, fn, n=200, warmup=10):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    elapsed = (time.perf_counter() - t0) / n * 1000
    print(f"  {name:40s} {elapsed:8.3f} ms")
    return elapsed


def run_benchmarks():
    results = {}

    for label, (h, w) in [("640x480", (480, 640)), ("1920x1080", (1080, 1920))]:
        print(f"\n{'='*60}")
        print(f"Image size: {label} (HxW={h}x{w}, 3ch)")
        print(f"{'='*60}")

        data = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        img = Image.frombuffer(data)
        res = {}

        # Equivalent OpenCV jitter: convertScaleAbs for brightness+contrast,
        # HSV round-trip for saturation+hue. Approximate but comparable.
        def cv_colorjitter(x):
            y = cv2.convertScaleAbs(x, alpha=1.15, beta=0.15 * 255)
            hsv = cv2.cvtColor(y, cv2.COLOR_RGB2HSV).astype(np.int16)
            hsv[..., 0] = (hsv[..., 0] + 18) % 180
            hsv[..., 1] = np.clip(hsv[..., 1] * 1.15, 0, 255)
            return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # ColorJitter
        print("\n--- ColorJitter ---")
        jk = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        ja = A.ColorJitter(brightness=(0.7, 1.3), contrast=(0.7, 1.3),
                           saturation=(0.7, 1.3), hue=(-0.1, 0.1), p=1.0)
        res["colorjitter"] = {
            "kornia": bench("kornia-rs", lambda: jk(img)),
            "albumentations": bench("albumentations", lambda: ja(image=data)["image"]),
            "opencv": bench("opencv", lambda: cv_colorjitter(data)),
        }

        # Brightness
        print("\n--- Brightness ---")
        bk = ColorJitter(brightness=0.3)
        ba = A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0, p=1.0)
        res["brightness"] = {
            "kornia": bench("kornia-rs", lambda: bk(img)),
            "albumentations": bench("albumentations", lambda: ba(image=data)["image"]),
            "opencv": bench("opencv", lambda: cv2.convertScaleAbs(data, alpha=1.0, beta=0.3 * 255)),
        }

        # Horizontal Flip
        print("\n--- Horizontal Flip ---")
        fk = RandomHorizontalFlip(p=1.0)
        fa = A.HorizontalFlip(p=1.0)
        res["hflip"] = {
            "kornia": bench("kornia-rs", lambda: fk(img)),
            "albumentations": bench("albumentations", lambda: fa(image=data)["image"]),
            "opencv": bench("opencv", lambda: cv2.flip(data, 1)),
        }

        # Vertical Flip
        print("\n--- Vertical Flip ---")
        vk = RandomVerticalFlip(p=1.0)
        va = A.VerticalFlip(p=1.0)
        res["vflip"] = {
            "kornia": bench("kornia-rs", lambda: vk(img)),
            "albumentations": bench("albumentations", lambda: va(image=data)["image"]),
            "opencv": bench("opencv", lambda: cv2.flip(data, 0)),
        }

        # Random Crop 224x224 — apples-to-apples: all three pick random (x,y)
        # per call so cache behavior matches. The previous fixed-(0,0) slice for
        # opencv gave it a warm-cache advantage that didn't reflect real training.
        print("\n--- Crop 224x224 ---")
        import random
        ck = RandomCrop((224, 224))
        ca = A.RandomCrop(224, 224, p=1.0)

        def cv_randcrop():
            x = random.randrange(w - 224)
            y = random.randrange(h - 224)
            return data[y:y + 224, x:x + 224].copy()

        res["crop"] = {
            "kornia": bench("kornia-rs", lambda: ck(img)),
            "albumentations": bench("albumentations", lambda: ca(image=data)["image"]),
            "opencv": bench("opencv", cv_randcrop),
        }

        # Grayscale
        print("\n--- Grayscale ---")
        res["grayscale"] = {
            "kornia": bench("kornia-rs", lambda: img.to_grayscale()),
            "opencv": bench("opencv", lambda: cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)),
        }

        # Resize 50%
        print("\n--- Resize (half) ---")
        res["resize"] = {
            "kornia": bench("kornia-rs", lambda: img.resize(w // 2, h // 2)),
            "opencv": bench("opencv", lambda: cv2.resize(data, (w // 2, h // 2))),
        }

        # Gaussian Blur
        print("\n--- Gaussian Blur 5x5 ---")
        res["blur"] = {
            "kornia": bench("kornia-rs", lambda: img.gaussian_blur(5)),
            "opencv": bench("opencv", lambda: cv2.GaussianBlur(data, (5, 5), 0)),
        }

        # Rotation
        print("\n--- Rotation ±30° ---")
        rk = RandomRotation(30.0)
        ra = A.Rotate(limit=30, p=1.0)
        rot_m = cv2.getRotationMatrix2D((w / 2, h / 2), 30, 1.0)
        res["rotation"] = {
            "kornia": bench("kornia-rs", lambda: rk(img)),
            "albumentations": bench("albumentations", lambda: ra(image=data)["image"]),
            "opencv": bench("opencv", lambda: cv2.warpAffine(data, rot_m, (w, h))),
        }

        # Normalize — kornia returns f32 HWC; OpenCV equivalent: float32 convert + subtract/divide
        print("\n--- Normalize ---")
        mean_arr = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255
        std_arr = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255
        res["normalize"] = {
            "kornia": bench("kornia-rs", lambda: img.normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))),
            "opencv": bench("opencv", lambda: (data.astype(np.float32) - mean_arr) / std_arr),
        }

        results[label] = res

    return results


if __name__ == "__main__":
    results = run_benchmarks()
    print("\n" + json.dumps(results, indent=2))
