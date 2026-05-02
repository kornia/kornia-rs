"""Benchmark kornia-rs augmentations vs albumentations/OpenCV."""
import json
import numpy as np
import kornia_rs as K
from kornia_rs.image import Image
from kornia_rs.augmentations import ColorJitter, RandomHorizontalFlip, RandomVerticalFlip, RandomCrop, RandomRotation

import albumentations as A
import cv2

from _bench import bench as _bench_fn, compat_print


def bench(name, fn, n=None, warmup=None):
    return compat_print(name, _bench_fn(fn))


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
        # 2x3 rotation-around-center (cv2.getRotationMatrix2D convention,
        # +30° CCW, scale 1.0). Built directly so cv2 only appears as a
        # timing target via cv2.warpAffine, not as a setup helper.
        _a = np.deg2rad(30.0)
        _c, _s = float(np.cos(_a)), float(np.sin(_a))
        rot_m = np.array([
            [ _c, _s,  (1 - _c) * (w / 2) - _s * (h / 2)],
            [-_s, _c,  _s * (w / 2) + (1 - _c) * (h / 2)],
        ], dtype=np.float32)
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

        # Box Blur 5x5
        print("\n--- Box Blur 5x5 ---")
        res["box_blur"] = {
            "kornia": bench("kornia-rs", lambda: img.box_blur(5)),
            "opencv": bench("opencv", lambda: cv2.blur(data, (5, 5))),
        }

        # Warp Affine (non-rotation shear) — distinct from Rotation ±30°
        print("\n--- Warp Affine (shear) ---")
        affine_flat = [1.0, 0.3, 0.0, 0.0, 1.0, 0.0]
        affine_cv = np.array([[1.0, 0.3, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        res["warp_affine"] = {
            "kornia": bench("kornia-rs", lambda: K.imgproc.warp_affine(data, affine_flat, (h, w), "bilinear")),
            "opencv": bench("opencv", lambda: cv2.warpAffine(data, affine_cv, (w, h))),
        }

        # Warp Perspective
        print("\n--- Warp Perspective ---")
        persp_flat = [1.2, 0.1, -30.0, 0.05, 1.15, -20.0, 0.0005, 0.0002, 1.0]
        persp_cv = np.array(persp_flat, dtype=np.float32).reshape(3, 3)
        res["warp_perspective"] = {
            "kornia": bench("kornia-rs", lambda: K.imgproc.warp_perspective(data, persp_flat, (h, w), "bilinear")),
            "opencv": bench("opencv", lambda: cv2.warpPerspective(data, persp_cv, (w, h))),
        }

        results[label] = res

    return results


if __name__ == "__main__":
    results = run_benchmarks()
    print("\n" + json.dumps(results, indent=2))
