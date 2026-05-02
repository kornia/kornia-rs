"""Tests for the Image API and augmentations."""

import io
import multiprocessing
import tempfile
import os
import numpy as np
import pytest

from kornia_rs.image import Image
from kornia_rs.augmentations import (
    ColorJitter, RandomHorizontalFlip, RandomVerticalFlip,
    RandomCrop, RandomRotation, Compose, set_seed,
)


def _make_test_image(width=8, height=6, channels=3, fill=42):
    """Create a test Image via frombuffer (zero-copy from numpy)."""
    data = np.full((height, width, channels), fill, dtype=np.uint8)
    return Image.frombuffer(data)


# --- Image class tests ---

class TestImage:
    def test_constructor_numpy(self):
        data = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        img = Image(data)
        assert img.width == 80
        assert img.height == 100
        assert img.channels == 3
        assert img.size == (80, 100)
        assert img.dtype == np.uint8

    def test_create_grayscale_2d(self):
        data = np.random.randint(0, 255, (50, 60), dtype=np.uint8)
        img = Image(data)
        assert img.channels == 1

    def test_copy(self):
        img = _make_test_image()
        copy = img.copy()
        assert img == copy
        copy.data[0, 0, 0] = 255
        assert img != copy

    def test_context_manager(self):
        with _make_test_image() as img:
            assert img.width == 8

    def test_repr(self):
        img = _make_test_image(width=100, height=50)
        r = repr(img)
        assert "RGB" in r
        assert "100x50" in r

    def test_to_numpy_is_copy(self):
        data = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        img = Image.frombuffer(data)
        out = img.to_numpy()
        assert np.array_equal(out, data)
        out[0, 0, 0] = 255 - out[0, 0, 0]
        assert not np.array_equal(out, img.data)

    def test_array_protocol(self):
        data = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        img = Image.frombuffer(data)
        arr = np.array(img)
        assert np.array_equal(arr, data)

    def test_eq(self):
        data = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        img1 = Image.frombuffer(data.copy())
        img2 = Image.frombuffer(data.copy())
        assert img1 == img2

    def test_invalid_dims(self):
        with pytest.raises((ValueError, TypeError)):
            Image.frombuffer(np.zeros((10,), dtype=np.uint8))

    def test_nbytes(self):
        img = _make_test_image(width=100, height=50, channels=3)
        assert img.nbytes == 100 * 50 * 3

    def test_len(self):
        img = _make_test_image(width=100, height=50)
        assert len(img) == 50  # height


class TestFrombuffer:
    """Test Image.frombuffer() — zero-copy from numpy arrays."""

    def test_3d_array(self):
        arr = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        img = Image.frombuffer(arr)
        assert img.width == 80
        assert img.height == 100
        assert img.channels == 3
        assert np.shares_memory(img.data, arr)

    def test_2d_array(self):
        arr = np.random.randint(0, 255, (50, 60), dtype=np.uint8)
        img = Image.frombuffer(arr)
        assert img.channels == 1
        assert np.shares_memory(img.data, arr)

    def test_mutation_visible_both_ways(self):
        arr = np.zeros((10, 10, 3), dtype=np.uint8)
        img = Image.frombuffer(arr)
        arr[0, 0, 0] = 42
        assert img.data[0, 0, 0] == 42
        img.data[5, 5, 1] = 99
        assert arr[5, 5, 1] == 99

    def test_with_mode(self):
        arr = np.zeros((10, 10, 3), dtype=np.uint8)
        img = Image.frombuffer(arr, mode="BGR")
        assert img.mode == "BGR"

    def test_rejects_raw_bytes(self):
        with pytest.raises(TypeError):
            Image.frombuffer(bytes([0] * 24))

    def test_rejects_1d(self):
        with pytest.raises((ValueError, TypeError)):
            Image.frombuffer(np.zeros((10,), dtype=np.uint8))


class TestFrombytes:
    """Test Image.frombytes() — copies raw bytes into a new buffer."""

    def test_bytes(self):
        width, height, channels = 10, 5, 3
        raw = bytes([128] * (width * height * channels))
        img = Image.frombytes(raw, width, height, channels)
        assert img.width == width
        assert img.height == height
        assert img.channels == channels
        assert np.all(img.data == 128)

    def test_bytearray(self):
        ba = bytearray([100] * (8 * 6 * 3))
        img = Image.frombytes(ba, 8, 6, 3)
        assert img.width == 8
        assert np.all(img.data == 100)

    def test_memoryview(self):
        arr = np.full((6, 8, 3), 77, dtype=np.uint8)
        mv = memoryview(arr)
        img = Image.frombytes(mv, 8, 6, 3)
        assert img.width == 8
        assert img.height == 6
        assert np.all(img.data == 77)

    def test_owns_data(self):
        """frombytes creates its own buffer — can't share with immutable bytes."""
        raw = bytes([55] * (4 * 4 * 3))
        img = Image.frombytes(raw, 4, 4, 3)
        assert np.all(img.data == 55)

    def test_wrong_size_raises(self):
        raw = bytes([0] * 10)
        with pytest.raises(ValueError):
            Image.frombytes(raw, 4, 4, 3)

    def test_with_mode(self):
        raw = bytes([0] * (4 * 4 * 1))
        img = Image.frombytes(raw, 4, 4, 1, mode="L")
        assert img.mode == "L"

    def test_requires_dimensions(self):
        """Raw bytes without width/height should raise TypeError."""
        raw = bytes([0] * 24)
        with pytest.raises(TypeError):
            Image.frombytes(raw)


class TestImageSerialize:
    """Test serialization for Ray Data / multiprocessing compatibility."""

    def test_reduce(self):
        img = _make_test_image(fill=42)
        constructor, args = img.__reduce__()
        assert constructor is Image
        reconstructed = constructor(*args)
        assert img == reconstructed

    def test_getstate_setstate(self):
        img = _make_test_image(fill=42)
        constructor, args = img.__reduce__()
        img2 = constructor(*args)
        assert np.array_equal(img.data, img2.data)


# --- Zero-copy and memory tests ---

class TestZeroCopy:
    """Verify zero-copy behavior and memory ownership semantics."""

    def test_constructor_shares_memory(self):
        arr = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        img = Image(arr)
        assert np.shares_memory(img.data, arr)

    def test_constructor_2d_shares_memory(self):
        arr = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        img = Image(arr)
        assert img.channels == 1
        assert np.shares_memory(img.data, arr)

    def test_constructor_mutation_roundtrip(self):
        arr = np.zeros((10, 10, 3), dtype=np.uint8)
        img = Image(arr)
        arr[3, 3, 0] = 200
        assert img.data[3, 3, 0] == 200
        img.data[7, 7, 2] = 150
        assert arr[7, 7, 2] == 150

    def test_data_getter_shares_memory(self):
        arr = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
        img = Image(arr)
        d1 = img.data
        d2 = img.data
        assert np.shares_memory(d1, d2)
        assert np.shares_memory(d1, arr)

    def test_asarray_shares_memory(self):
        arr = np.random.randint(0, 255, (30, 30, 3), dtype=np.uint8)
        img = Image(arr)
        out = np.asarray(img)
        assert np.shares_memory(out, arr)

    def test_array_copy_isolates(self):
        arr = np.random.randint(0, 255, (30, 30, 3), dtype=np.uint8)
        img = Image(arr)
        out = np.array(img, copy=True)
        assert not np.shares_memory(out, arr)

    def test_copy_does_not_share_memory(self):
        arr = np.full((20, 20, 3), 50, dtype=np.uint8)
        img = Image(arr)
        img2 = img.copy()
        assert not np.shares_memory(img.data, img2.data)
        img2.data[0, 0, 0] = 255
        assert img.data[0, 0, 0] == 50

    def test_to_numpy_does_not_share_memory(self):
        arr = np.full((20, 20, 3), 50, dtype=np.uint8)
        img = Image(arr)
        out = img.to_numpy()
        assert not np.shares_memory(out, arr)
        out[0, 0, 0] = 255
        assert img.data[0, 0, 0] == 50


class TestImmutableTransforms:
    """Verify ALL transforms return new Images with independent memory."""

    def setup_method(self):
        self.original_data = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        self.img = Image.frombuffer(self.original_data.copy())
        self.snapshot = self.img.data.copy()

    def _assert_original_unchanged(self):
        assert np.array_equal(self.img.data, self.snapshot), \
            "Transform mutated the source Image!"

    def _assert_isolated(self, result):
        self._assert_original_unchanged()
        assert not np.shares_memory(result.data, self.img.data), \
            "Transform result shares memory with source!"

    def test_resize_isolates(self):
        result = self.img.resize(50, 40)
        self._assert_isolated(result)

    def test_flip_horizontal_isolates(self):
        result = self.img.flip_horizontal()
        self._assert_isolated(result)

    def test_flip_vertical_isolates(self):
        result = self.img.flip_vertical()
        self._assert_isolated(result)

    def test_crop_isolates(self):
        result = self.img.crop(10, 10, 30, 30)
        self._assert_isolated(result)

    def test_gaussian_blur_isolates(self):
        result = self.img.gaussian_blur(3, 1.0)
        self._assert_isolated(result)

    def test_box_blur_isolates(self):
        result = self.img.box_blur(3)
        self._assert_isolated(result)

    def test_adjust_brightness_isolates(self):
        result = self.img.adjust_brightness(0.1)
        self._assert_isolated(result)

    def test_adjust_contrast_isolates(self):
        result = self.img.adjust_contrast(1.5)
        self._assert_isolated(result)

    def test_adjust_saturation_isolates(self):
        result = self.img.adjust_saturation(1.5)
        self._assert_isolated(result)

    def test_adjust_hue_isolates(self):
        result = self.img.adjust_hue(0.1)
        self._assert_isolated(result)

    def test_rotate_isolates(self):
        result = self.img.rotate(45.0)
        self._assert_isolated(result)

    def test_to_grayscale_isolates(self):
        result = self.img.to_grayscale()
        self._assert_isolated(result)

    def test_to_rgb_from_gray_isolates(self):
        gray = self.img.to_grayscale()
        snapshot = gray.data.copy()
        result = gray.to_rgb()
        assert np.array_equal(gray.data, snapshot)
        assert not np.shares_memory(result.data, gray.data)

    def test_chain_isolates(self):
        result = (
            self.img
            .resize(50, 40)
            .flip_horizontal()
            .crop(5, 5, 20, 20)
            .adjust_brightness(0.1)
            .gaussian_blur(3, 1.0)
        )
        self._assert_isolated(result)
        assert result.shape == (20, 20, 3)

    def test_augmentation_colorjitter_isolates(self):
        jitter = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        result = jitter(self.img)
        self._assert_isolated(result)

    def test_augmentation_random_hflip_isolates(self):
        flip = RandomHorizontalFlip(p=1.0)
        result = flip(self.img)
        self._assert_isolated(result)

    def test_augmentation_random_vflip_isolates(self):
        flip = RandomVerticalFlip(p=1.0)
        result = flip(self.img)
        self._assert_isolated(result)

    def test_augmentation_random_crop_isolates(self):
        crop = RandomCrop((50, 50))
        result = crop(self.img)
        self._assert_isolated(result)

    def test_augmentation_random_rotation_isolates(self):
        rot = RandomRotation(30.0)
        result = rot(self.img)
        self._assert_isolated(result)

    def test_compose_isolates(self):
        transform = Compose([
            RandomHorizontalFlip(p=1.0),
            ColorJitter(brightness=0.2),
        ])
        result = transform(self.img)
        self._assert_isolated(result)


class TestScopedLifetime:
    """Verify Image data survives or is properly isolated across scopes."""

    def test_data_survives_image_going_out_of_scope(self):
        img = _make_test_image(fill=42)
        arr = img.data
        del img
        assert arr[0, 0, 0] == 42
        assert arr.shape == (6, 8, 3)

    def test_frombuffer_data_lifetime(self):
        arr = np.full((10, 10, 3), 77, dtype=np.uint8)
        img = Image.frombuffer(arr)
        assert img.data[0, 0, 0] == 77
        arr[0, 0, 0] = 88
        assert img.data[0, 0, 0] == 88

    def test_transform_result_independent_of_source_scope(self):
        def create_and_transform():
            img = _make_test_image(width=100, height=80, fill=128)
            return img.resize(50, 40).flip_horizontal()

        result = create_and_transform()
        assert result.width == 50
        assert result.height == 40
        assert result.data[0, 0, 0] is not None

    def test_augmentation_result_independent(self):
        def augment():
            img = _make_test_image(width=50, height=50, fill=100)
            flip = RandomHorizontalFlip(p=1.0)
            return flip(img)

        result = augment()
        assert result.width == 50
        assert result.data.sum() > 0

    def test_chained_intermediate_gc(self):
        import gc
        import weakref

        img = _make_test_image(width=100, height=80, fill=128)
        intermediate = img.resize(50, 40)
        weak_ref = weakref.ref(intermediate)
        result = intermediate.flip_horizontal()
        del intermediate
        gc.collect()

        assert result.width == 50
        assert result.height == 40
        assert result.data.shape == (40, 50, 3)


# --- Transform tests ---

class TestTransforms:
    def setup_method(self):
        self.img = _make_test_image(width=100, height=80, channels=3, fill=128)

    def test_resize(self):
        resized = self.img.resize(50, 40)
        assert resized.width == 50
        assert resized.height == 40

    def test_flip_horizontal(self):
        data = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        img = Image.frombuffer(data)
        flipped = img.flip_horizontal()
        assert np.array_equal(flipped.data[0, 0], data[0, 1])
        assert np.array_equal(flipped.data[0, 1], data[0, 0])

    def test_flip_vertical(self):
        data = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        img = Image.frombuffer(data)
        flipped = img.flip_vertical()
        assert np.array_equal(flipped.data[0], data[1])
        assert np.array_equal(flipped.data[1], data[0])

    def test_crop(self):
        cropped = self.img.crop(10, 20, 30, 40)
        assert cropped.width == 30
        assert cropped.height == 40

    def test_gaussian_blur(self):
        blurred = self.img.gaussian_blur(kernel_size=3, sigma=1.0)
        assert blurred.shape == self.img.shape

    def test_adjust_brightness(self):
        bright = self.img.adjust_brightness(0.1)
        assert bright.shape == self.img.shape

    def test_normalize(self):
        normalized = self.img.normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        assert normalized.dtype == np.float32

    def test_chain(self):
        result = self.img.resize(50, 40).flip_horizontal().crop(0, 0, 25, 20)
        assert result.width == 25
        assert result.height == 20


# --- Color tests ---

class TestColor:
    def test_to_grayscale(self):
        img = _make_test_image(width=10, height=10, channels=3, fill=128)
        gray = img.to_grayscale()
        assert gray.channels == 1

    def test_to_rgb_from_gray(self):
        gray = _make_test_image(width=10, height=10, channels=1, fill=128)
        rgb = gray.to_rgb()
        assert rgb.channels == 3

    def test_grayscale_roundtrip(self):
        img = _make_test_image(width=10, height=10, channels=3, fill=100)
        gray = img.to_grayscale()
        rgb = gray.to_rgb()
        assert rgb.channels == 3

    def test_already_grayscale(self):
        gray = _make_test_image(width=10, height=10, channels=1, fill=128)
        result = gray.to_grayscale()
        assert result == gray


# --- Augmentation tests ---

class TestColorJitter:
    def test_basic(self):
        img = _make_test_image(width=100, height=80, channels=3, fill=128)
        jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        result = jitter(img)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_no_change(self):
        img = _make_test_image(width=100, height=80, channels=3, fill=128)
        jitter = ColorJitter()
        result = jitter(img)
        assert result.shape == img.shape

    def test_brightness_only(self):
        img = _make_test_image(width=50, height=50, channels=3, fill=128)
        jitter = ColorJitter(brightness=0.5)
        result = jitter(img)
        assert result.shape == img.shape

    def test_repr(self):
        jitter = ColorJitter(brightness=0.2, contrast=0.3)
        assert "ColorJitter" in repr(jitter)


class TestRandomFlips:
    def test_horizontal_flip_always(self):
        img = Image.frombuffer(np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8))
        flip = RandomHorizontalFlip(p=1.0)
        result = flip(img)
        assert result.shape == img.shape

    def test_vertical_flip_always(self):
        img = _make_test_image()
        flip = RandomVerticalFlip(p=1.0)
        result = flip(img)
        assert result.shape == img.shape

    def test_no_flip(self):
        img = _make_test_image(fill=42)
        flip = RandomHorizontalFlip(p=0.0)
        result = flip(img)
        assert result == img


class TestRandomCrop:
    def test_basic(self):
        img = _make_test_image(width=100, height=80)
        crop = RandomCrop((50, 50))
        result = crop(img)
        assert result.width == 50
        assert result.height == 50

    def test_too_large(self):
        img = _make_test_image(width=10, height=10)
        crop = RandomCrop((20, 20))
        with pytest.raises(ValueError):
            crop(img)


class TestRandomRotation:
    def test_basic(self):
        img = _make_test_image(width=100, height=80, channels=3, fill=128)
        rot = RandomRotation(30.0)
        result = rot(img)
        assert result.shape == img.shape

    def test_repr(self):
        rot = RandomRotation(45.0)
        r = repr(rot)
        assert "RandomRotation" in r
        assert "-45" in r
        assert "45" in r

    def test_does_not_mutate(self):
        img = _make_test_image(width=100, height=80, channels=3, fill=128)
        snapshot = img.data.copy()
        rot = RandomRotation(30.0)
        result = rot(img)
        assert np.array_equal(img.data, snapshot)
        assert not np.shares_memory(result.data, img.data)


class TestCompose:
    def test_basic(self):
        img = _make_test_image(width=100, height=80, channels=3, fill=128)
        transform = Compose([
            RandomHorizontalFlip(p=0.5),
            ColorJitter(brightness=0.1),
        ])
        result = transform(img)
        assert result.shape == img.shape

    def test_repr(self):
        transform = Compose([RandomHorizontalFlip(), ColorJitter()])
        assert "Compose" in repr(transform)


class TestSetSeed:
    """Test set_seed for reproducible augmentations."""

    def teardown_method(self):
        set_seed(None)

    def test_seed_reproducible_colorjitter(self):
        img = _make_test_image(width=50, height=50, channels=3, fill=128)
        jitter = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)

        set_seed(42)
        result1 = jitter(img)

        set_seed(42)
        result2 = jitter(img)

        assert np.array_equal(result1.data, result2.data)

    def test_seed_reproducible_random_flip(self):
        img = _make_test_image(width=50, height=50, channels=3, fill=128)
        flip = RandomHorizontalFlip(p=0.5)

        set_seed(123)
        results1 = [np.array(flip(img).data) for _ in range(10)]

        set_seed(123)
        results2 = [np.array(flip(img).data) for _ in range(10)]

        for r1, r2 in zip(results1, results2):
            assert np.array_equal(r1, r2)

    def test_seed_reproducible_random_crop(self):
        img_data = np.random.RandomState(0).randint(0, 255, (80, 100, 3), dtype=np.uint8)
        img = Image(img_data)
        crop = RandomCrop((50, 50))

        set_seed(99)
        result1 = crop(img)

        set_seed(99)
        result2 = crop(img)

        assert np.array_equal(result1.data, result2.data)

    def test_seed_reproducible_compose(self):
        img_data = np.random.RandomState(0).randint(0, 255, (80, 100, 3), dtype=np.uint8)
        img = Image(img_data)
        transform = Compose([
            RandomHorizontalFlip(p=0.5),
            RandomCrop((50, 60)),
            ColorJitter(brightness=0.2, contrast=0.2),
        ])

        set_seed(77)
        result1 = transform(img)

        set_seed(77)
        result2 = transform(img)

        assert np.array_equal(result1.data, result2.data)

    def test_different_seeds_differ(self):
        img_data = np.random.RandomState(0).randint(0, 255, (80, 100, 3), dtype=np.uint8)
        img = Image(img_data)
        jitter = ColorJitter(brightness=0.5, contrast=0.5)

        set_seed(1)
        result1 = jitter(img)

        set_seed(2)
        result2 = jitter(img)

        assert not np.array_equal(result1.data, result2.data)

    def test_reset_seed_none(self):
        set_seed(42)
        set_seed(None)
        img = _make_test_image(width=50, height=50, channels=3, fill=128)
        flip = RandomHorizontalFlip(p=0.5)
        result = flip(img)
        assert result.shape == img.shape

    def test_seed_reproducible_random_rotation(self):
        img_data = np.random.RandomState(0).randint(0, 255, (80, 100, 3), dtype=np.uint8)
        img = Image(img_data)
        rot = RandomRotation(30.0)

        set_seed(55)
        result1 = rot(img)

        set_seed(55)
        result2 = rot(img)

        assert np.array_equal(result1.data, result2.data)


# --- Cached params tests ---

class TestCachedParams:
    """Test sample(), last_params, and passing params to __call__."""

    def setup_method(self):
        self.img = _make_test_image(width=50, height=40, channels=3, fill=128)

    def test_color_jitter_sample_and_apply(self):
        jitter = ColorJitter(brightness=0.5, contrast=0.3, saturation=0.2, hue=0.1)
        params = jitter.sample()
        assert "brightness" in params
        assert "contrast" in params
        assert "saturation" in params
        assert "hue" in params
        assert "order" in params
        # Apply same params to two copies
        r1 = jitter(self.img, params=params)
        r2 = jitter(self.img, params=params)
        assert np.array_equal(r1.data, r2.data)

    def test_color_jitter_last_params(self):
        jitter = ColorJitter(brightness=0.5)
        assert jitter.last_params is None
        jitter(self.img)
        assert jitter.last_params is not None
        assert "brightness" in jitter.last_params

    def test_hflip_sample_and_apply(self):
        flip = RandomHorizontalFlip(p=0.5)
        params = flip.sample()
        assert "flip" in params
        r1 = flip(self.img, params=params)
        r2 = flip(self.img, params=params)
        assert np.array_equal(r1.data, r2.data)

    def test_hflip_last_params(self):
        flip = RandomHorizontalFlip(p=1.0)
        assert flip.last_params is None
        flip(self.img)
        assert flip.last_params is not None
        assert flip.last_params["flip"] is True

    def test_vflip_sample_and_apply(self):
        flip = RandomVerticalFlip(p=0.5)
        params = flip.sample()
        assert "flip" in params
        r1 = flip(self.img, params=params)
        r2 = flip(self.img, params=params)
        assert np.array_equal(r1.data, r2.data)

    def test_vflip_last_params(self):
        flip = RandomVerticalFlip(p=1.0)
        assert flip.last_params is None
        flip(self.img)
        assert flip.last_params is not None
        assert flip.last_params["flip"] is True

    def test_random_crop_sample_and_apply(self):
        crop = RandomCrop((20, 30))
        params = crop.sample(self.img)
        assert "x" in params
        assert "y" in params
        r1 = crop(self.img, params=params)
        r2 = crop(self.img, params=params)
        assert np.array_equal(r1.data, r2.data)
        assert r1.width == 30
        assert r1.height == 20

    def test_random_crop_last_params(self):
        crop = RandomCrop((20, 30))
        assert crop.last_params is None
        crop(self.img)
        assert crop.last_params is not None
        assert "x" in crop.last_params
        assert "y" in crop.last_params

    def test_random_rotation_sample_and_apply(self):
        rot = RandomRotation(30.0)
        params = rot.sample()
        assert "angle" in params
        r1 = rot(self.img, params=params)
        r2 = rot(self.img, params=params)
        assert np.array_equal(r1.data, r2.data)

    def test_random_rotation_last_params(self):
        rot = RandomRotation(30.0)
        assert rot.last_params is None
        rot(self.img)
        assert rot.last_params is not None
        assert "angle" in rot.last_params

    def test_apply_same_augmentation_to_image_and_mask(self):
        """Realistic use case: apply same transform to image and its mask."""
        img = _make_test_image(width=50, height=40, channels=3, fill=100)
        mask = _make_test_image(width=50, height=40, channels=3, fill=200)

        flip = RandomHorizontalFlip(p=0.5)
        params = flip.sample()
        img_out = flip(img, params=params)
        mask_out = flip(mask, params=params)

        if params["flip"]:
            assert np.array_equal(img_out.data, img.flip_horizontal().data)
            assert np.array_equal(mask_out.data, mask.flip_horizontal().data)
        else:
            assert np.array_equal(img_out.data, img.data)
            assert np.array_equal(mask_out.data, mask.data)


# --- Decode tests ---

class TestDecode:
    """Test Image.decode() for JPEG and PNG byte streams."""

    def _make_jpeg_bytes(self, width=16, height=16, channels=3, fill=100):
        arr = np.full((height, width, channels), fill, dtype=np.uint8)
        img = Image.frombuffer(arr)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            tmp_path = f.name
        try:
            img.save(tmp_path)
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            os.unlink(tmp_path)

    def _make_png_bytes(self, width=16, height=16, channels=3, fill=100):
        arr = np.full((height, width, channels), fill, dtype=np.uint8)
        img = Image.frombuffer(arr)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name
        try:
            img.save(tmp_path)
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            os.unlink(tmp_path)

    def test_decode_jpeg(self):
        jpeg_bytes = self._make_jpeg_bytes(width=16, height=16, channels=3)
        img = Image.decode(jpeg_bytes)
        assert img.channels == 3
        assert img.width == 16
        assert img.height == 16
        assert img.dtype == np.uint8

    def test_decode_png(self):
        png_bytes = self._make_png_bytes(width=16, height=16, channels=3)
        img = Image.decode(png_bytes)
        assert img.channels == 3
        assert img.width == 16
        assert img.height == 16
        assert img.dtype == np.uint8

    def test_decode_invalid_bytes(self):
        garbage = bytes([0xDE, 0xAD, 0xBE, 0xEF] * 16)
        with pytest.raises(Exception):
            Image.decode(garbage)


class TestEncode:
    """Test Image.encode() in-memory and Image.encode_png_u16() for depth maps.

    `encode` is the in-memory complement of `save` — same backend, no path on
    disk. `encode_png_u16` is the static-method bridge for 16-bit data until
    Image natively supports uint16 storage; PNG-16 is the only safe lossless
    format for depth (JPEG smears object-edge discontinuities).
    """

    def test_encode_jpeg_returns_bytes(self):
        arr = np.random.randint(0, 255, (32, 64, 3), dtype=np.uint8)
        img = Image.frombuffer(arr)
        out = img.encode("jpeg")
        assert isinstance(out, bytes)
        # JPEG SOI marker.
        assert out[:3] == b"\xff\xd8\xff"

    def test_encode_jpeg_quality_is_honoured(self):
        # High-frequency content so the quality knob actually moves the bytes.
        arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        img = Image.frombuffer(arr)
        small = img.encode("jpeg", quality=20)
        big = img.encode("jpeg", quality=95)
        assert len(big) > len(small)

    def test_encode_jpg_alias(self):
        # "jpg" must work as an alias of "jpeg" — matches `save` extension behavior.
        arr = np.zeros((8, 8, 3), dtype=np.uint8)
        img = Image.frombuffer(arr)
        a = img.encode("jpg")
        b = img.encode("jpeg")
        # Same encoder, same input → identical bytes.
        assert a == b

    def test_encode_png_rgb(self):
        arr = np.random.randint(0, 255, (32, 64, 3), dtype=np.uint8)
        img = Image.frombuffer(arr)
        out = img.encode("png")
        assert isinstance(out, bytes)
        # PNG signature (8 bytes).
        assert out[:8] == b"\x89PNG\r\n\x1a\n"
        # Round-trip via decode → exact match (PNG is lossless).
        decoded = Image.decode(out)
        assert decoded.width == 64
        assert decoded.height == 32
        assert decoded.channels == 3
        assert np.array_equal(np.array(decoded), arr)

    def test_encode_png_grayscale(self):
        arr = np.full((16, 16, 1), 128, dtype=np.uint8)
        img = Image.frombuffer(arr)
        out = img.encode("png")
        assert out[:8] == b"\x89PNG\r\n\x1a\n"

    def test_encode_png_rgba(self):
        arr = np.zeros((16, 16, 4), dtype=np.uint8)
        arr[:, :, 3] = 255  # opaque alpha
        img = Image.frombuffer(arr)
        out = img.encode("png")
        assert out[:8] == b"\x89PNG\r\n\x1a\n"

    def test_encode_unsupported_format_raises(self):
        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        img = Image.frombuffer(arr)
        with pytest.raises(ValueError, match="Unsupported"):
            img.encode("bmp")

    def test_encode_jpeg_rejects_grayscale(self):
        # JPEG path requires 3-channel; mirror the `save` constraint.
        arr = np.zeros((8, 8, 1), dtype=np.uint8)
        img = Image.frombuffer(arr)
        with pytest.raises(ValueError):
            img.encode("jpeg")

    # --- u16 (depth) — full Image lifecycle, not a static bridge ---

    def test_u16_image_construction_2d(self):
        # 2D uint16 → 1-channel u16 Image (depth-map shape).
        depth = np.full((32, 64), 1500, dtype=np.uint16)
        img = Image.fromarray(depth)
        assert img.dtype == np.uint16
        assert img.mode == "I;16"
        assert img.width == 64 and img.height == 32 and img.channels == 1

    def test_u16_image_construction_3d(self):
        depth = np.full((16, 16, 1), 1500, dtype=np.uint16)
        img = Image.fromarray(depth)
        assert img.dtype == np.uint16
        assert img.shape == (16, 16, 1)

    def test_u16_image_encode_png(self):
        # Realistic depth: smooth gradient + sharp object — what PNG-16 preserves.
        h, w = 48, 64
        depth = np.fromfunction(
            lambda y, x: 1000 + (x.astype(np.uint16) * 8) + (y.astype(np.uint16) * 4),
            (h, w),
            dtype=np.uint16,
        )
        depth[10:20, 20:40] = 500
        img = Image.fromarray(depth)
        out = img.encode("png")
        assert out[:8] == b"\x89PNG\r\n\x1a\n"

        # Round-trip: decode produces an equal u16 Image (PNG-16 is lossless).
        back = Image.decode(out, mode="L")
        assert back.dtype == np.uint16
        assert back.mode == "I;16"
        assert np.array_equal(np.array(back).reshape(h, w), depth)

    def test_u16_image_save_to_bytesio(self):
        # PIL parity: save accepts a file-like (BytesIO) target.
        depth = np.full((8, 8, 1), 9000, dtype=np.uint16)
        img = Image.fromarray(depth)
        buf = io.BytesIO()
        img.save(buf, format="png")
        out = buf.getvalue()
        assert out[:8] == b"\x89PNG\r\n\x1a\n"

    def test_u16_image_jpeg_rejected(self):
        depth = np.full((8, 8, 1), 1500, dtype=np.uint16)
        img = Image.fromarray(depth)
        with pytest.raises(ValueError, match="16-bit"):
            img.encode("jpeg")

    def test_u16_image_imgproc_raises_clear_error(self):
        # Math-heavy methods raise NotImplementedError on u16 with a hint;
        # dtype-trivial ops (flip_*, crop) work natively on u16.
        depth = np.full((8, 8, 1), 1500, dtype=np.uint16)
        img = Image.fromarray(depth)
        with pytest.raises(NotImplementedError, match="uint16"):
            img.resize(4, 4)
        with pytest.raises(NotImplementedError, match="uint16"):
            img.gaussian_blur()
        with pytest.raises(NotImplementedError, match="uint16"):
            img.adjust_brightness(0.5)


class TestSaveToBytesIO:
    """PIL-parity: ``Image.save(BytesIO, format=...)`` works for u8 too."""

    def test_save_to_bytesio_jpeg(self):
        arr = np.zeros((8, 8, 3), dtype=np.uint8)
        img = Image.frombuffer(arr)
        buf = io.BytesIO()
        img.save(buf, format="jpeg")
        out = buf.getvalue()
        assert out[:3] == b"\xff\xd8\xff"

    def test_save_to_bytesio_png(self):
        arr = np.full((8, 8, 3), 64, dtype=np.uint8)
        img = Image.frombuffer(arr)
        buf = io.BytesIO()
        img.save(buf, format="png")
        out = buf.getvalue()
        assert out[:8] == b"\x89PNG\r\n\x1a\n"

    def test_save_to_bytesio_requires_format(self):
        # No path extension to fall back on → format= is mandatory.
        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        img = Image.frombuffer(arr)
        buf = io.BytesIO()
        with pytest.raises(ValueError, match="format="):
            img.save(buf)


class TestFromarrayAlias:
    """``Image.fromarray`` is the PIL idiom; verify it's a true alias of frombuffer."""

    def test_fromarray_u8(self):
        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        a = Image.fromarray(arr)
        b = Image.frombuffer(arr)
        assert a.mode == b.mode
        assert a.dtype == b.dtype
        assert a.shape == b.shape

    def test_fromarray_u16(self):
        arr = np.full((4, 4), 1234, dtype=np.uint16)
        img = Image.fromarray(arr)
        assert img.dtype == np.uint16
        assert img.mode == "I;16"


# --- Multiprocessing tests ---

def _worker_process_image(data_tuple):
    """Worker function that runs in a separate process."""
    from kornia_rs.image import Image
    data, width, height = data_tuple
    img = Image.frombuffer(data)
    result = img.resize(width, height).flip_horizontal()
    return result.data


class TestMultiprocessing:
    """Verify Image works safely across process boundaries."""

    def test_spawn_pool(self):
        data = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        tasks = [(data.copy(), 50, 40) for _ in range(2)]

        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(2) as pool:
            results = pool.map(_worker_process_image, tasks)

        assert len(results) == 2
        for r in results:
            assert r.shape == (40, 50, 3)
            assert r.dtype == np.uint8

    def test_concurrent_futures(self):
        from concurrent.futures import ProcessPoolExecutor
        data = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        tasks = [(data.copy(), 50, 40) for _ in range(2)]

        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as executor:
            results = list(executor.map(_worker_process_image, tasks))

        assert len(results) == 2
        for r in results:
            assert r.shape == (40, 50, 3)
