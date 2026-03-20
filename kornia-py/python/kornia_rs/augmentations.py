"""Random augmentations — torchvision-style callables.

All augmentations return new Image instances (immutable pattern) for
multiprocess safety with Ray Data and similar frameworks.
"""

from __future__ import annotations

import numpy as np
import random
from typing import Tuple, Union, List

from kornia_rs._image import Image
from kornia_rs import transforms


class ColorJitter:
    """Randomly change brightness, contrast, saturation, and hue.

    Uses composed primitives (applied in random order per call) for future
    kernel fusion compatibility.
    # TODO: fuse these ops into a single pass when kernel fusion is available

    Args:
        brightness: Jitter range. Float or (min, max). Factor chosen from
            [max(0, 1-brightness), 1+brightness].
        contrast: Same format. Factor chosen uniformly.
        saturation: Same format.
        hue: Float in [0, 0.5] or (min, max) in [-0.5, 0.5].
    """

    def __init__(
        self,
        brightness: Union[float, Tuple[float, float]] = 0,
        contrast: Union[float, Tuple[float, float]] = 0,
        saturation: Union[float, Tuple[float, float]] = 0,
        hue: Union[float, Tuple[float, float]] = 0,
    ):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation', center=1)
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5))

    @staticmethod
    def _check_input(value, name, center=1, bound=None):
        if isinstance(value, (int, float)):
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
            if name == 'hue':
                value = (-value, value)
            else:
                value = (max(0, center - value), center + value)
        if bound is not None:
            if value[0] < bound[0] or value[1] > bound[1]:
                raise ValueError(f"{name} values should be between {bound}")
        return value

    def __call__(self, img: Image) -> Image:
        """Apply random color jitter to an image."""
        data = img.data.copy()

        # Build list of transforms and shuffle order (like torchvision)
        # TODO: when kernel fusion is available, fuse into single pass
        ops = []

        if self.brightness[0] != 1 or self.brightness[1] != 1:
            factor = random.uniform(self.brightness[0], self.brightness[1])
            ops.append(('brightness', factor))

        if self.contrast[0] != 1 or self.contrast[1] != 1:
            factor = random.uniform(self.contrast[0], self.contrast[1])
            ops.append(('contrast', factor))

        if self.saturation[0] != 1 or self.saturation[1] != 1:
            factor = random.uniform(self.saturation[0], self.saturation[1])
            ops.append(('saturation', factor))

        if self.hue[0] != 0 or self.hue[1] != 0:
            factor = random.uniform(self.hue[0], self.hue[1])
            ops.append(('hue', factor))

        random.shuffle(ops)

        for op_name, factor in ops:
            if op_name == 'brightness':
                data = np.clip(data.astype(np.float32) * factor, 0, 255).astype(np.uint8)
            elif op_name == 'contrast':
                data = transforms.adjust_contrast(data, factor)
            elif op_name == 'saturation':
                data = transforms.adjust_saturation(data, factor)
            elif op_name == 'hue':
                data = transforms.adjust_hue(data, factor)

        return Image._wrap(data)

    def __repr__(self) -> str:
        return (f"ColorJitter(brightness={self.brightness}, contrast={self.contrast}, "
                f"saturation={self.saturation}, hue={self.hue})")


class RandomHorizontalFlip:
    """Randomly flip image horizontally with probability p."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: Image) -> Image:
        if random.random() < self.p:
            return img.flip_horizontal()
        return img.copy()

    def __repr__(self) -> str:
        return f"RandomHorizontalFlip(p={self.p})"


class RandomVerticalFlip:
    """Randomly flip image vertically with probability p."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: Image) -> Image:
        if random.random() < self.p:
            return img.flip_vertical()
        return img.copy()

    def __repr__(self) -> str:
        return f"RandomVerticalFlip(p={self.p})"


class RandomCrop:
    """Randomly crop image to given size."""

    def __init__(self, size: Union[int, Tuple[int, int]]):
        if isinstance(size, int):
            size = (size, size)
        self.height, self.width = size

    def __call__(self, img: Image) -> Image:
        if img.width < self.width or img.height < self.height:
            raise ValueError(
                f"Image ({img.width}x{img.height}) is smaller than "
                f"crop size ({self.width}x{self.height})"
            )
        x = random.randint(0, img.width - self.width)
        y = random.randint(0, img.height - self.height)
        return img.crop(x, y, self.width, self.height)

    def __repr__(self) -> str:
        return f"RandomCrop(size=({self.height}, {self.width}))"


class RandomRotation:
    """Randomly rotate image within degree range."""

    def __init__(self, degrees: Union[float, Tuple[float, float]]):
        if isinstance(degrees, (int, float)):
            degrees = (-degrees, degrees)
        self.degrees = degrees

    def __call__(self, img: Image) -> Image:
        angle = random.uniform(self.degrees[0], self.degrees[1])
        return img.rotate(angle)

    def __repr__(self) -> str:
        return f"RandomRotation(degrees={self.degrees})"


class Compose:
    """Compose several transforms together."""

    def __init__(self, transforms_list: List):
        self.transforms = transforms_list

    def __call__(self, img: Image) -> Image:
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self) -> str:
        lines = [f"  {t}" for t in self.transforms]
        return "Compose([\n" + ",\n".join(lines) + "\n])"
