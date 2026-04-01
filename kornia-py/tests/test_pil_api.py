"""Tests for the PIL-like Image API and augmentations."""

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
    """Create a test Image using frombytes."""
    data = np.full((height, width, channels), fill, dtype=np.uint8)
    return Image.frombytes(data)


# --- Image class tests ---

class TestImage:
    def test_frombytes_numpy(self):
        data = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        img = Image.frombytes(data)
        assert img.width == 80
        assert img.height == 100
        assert img.channels == 3
        assert img.size == (80, 100)
        assert img.dtype == np.uint8

    def test_create_grayscale_2d(self):
        data = np.random.randint(0, 255, (50, 60), dtype=np.uint8)
        img = Image.frombytes(data)
        assert img.channels == 1

    def test_frombytes_raw_bytes(self):
        width, height, channels = 10, 5, 3
        raw = bytes([128] * (width * height * channels))
        img = Image.frombytes(raw, width=width, height=height, channels=channels)
        assert img.width == width
        assert img.height == height
        assert img.channels == channels
        assert np.all(img.data == 128)

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
        img = Image.frombytes(data)
        out = img.to_numpy()
        assert np.array_equal(out, data)
        out[0, 0, 0] = 255 - out[0, 0, 0]
        assert not np.array_equal(out, img.data)

    def test_array_protocol(self):
        data = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        img = Image.frombytes(data)
        arr = np.array(img)
        assert np.array_equal(arr, data)

    def test_eq(self):
        data = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        img1 = Image.frombytes(data.copy())
        img2 = Image.frombytes(data.copy())
        assert img1 == img2

    def test_invalid_dims(self):
        with pytest.raises(ValueError):
            Image.frombytes(np.zeros((10,), dtype=np.uint8))

    def test_nbytes(self):
        img = _make_test_image(width=100, height=50, channels=3)
        assert img.nbytes == 100 * 50 * 3

    def test_len(self):
        img = _make_test_image(width=100, height=50)
        assert len(img) == 50  # height

    def test_frombytes_memoryview(self):
        """frombytes from a memoryview should be zero-copy."""
        arr = np.full((6, 8, 3), 77, dtype=np.uint8)
        mv = memoryview(arr)
        img = Image.frombytes(mv, width=8, height=6, channels=3)
        assert img.width == 8
        assert img.height == 6
        assert img.channels == 3
        assert np.all(img.data == 77)

    def test_frombytes_requires_dims(self):
        """Raw bytes without width/height should raise ValueError."""
        raw = bytes([0] * 24)
        with pytest.raises((ValueError, TypeError)):
            Image.frombytes(raw)


class TestImageSerialize:
    """Test serialization for Ray Data / multiprocessing compatibility."""

    def test_reduce(self):
        """Image.__reduce__ returns (Image, (data,)) for serialization."""
        img = _make_test_image(fill=42)
        constructor, args = img.__reduce__()
        assert constructor is Image
        reconstructed = constructor(*args)
        assert img == reconstructed

    def test_getstate_setstate(self):
        """Test serialization roundtrip via reduce/reconstruct."""
        img = _make_test_image(fill=42)
        constructor, args = img.__reduce__()
        img2 = constructor(*args)
        assert np.array_equal(img.data, img2.data)


# --- Zero-copy and memory tests ---

class TestZeroCopy:
    """Verify zero-copy behavior and memory ownership semantics."""

    def test_frombytes_shares_memory(self):
        """frombytes(numpy_arr) must NOT copy — same memory as source array."""
        arr = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        img = Image.frombytes(arr)
        assert np.shares_memory(img.data, arr)

    def test_frombytes_mutation_visible(self):
        """Mutating the source array is visible through the Image (shared memory)."""
        arr = np.zeros((10, 10, 3), dtype=np.uint8)
        img = Image.frombytes(arr)
        arr[0, 0, 0] = 42
        assert img.data[0, 0, 0] == 42

    def test_frombytes_mutation_from_image_visible(self):
        """Mutating via img.data is visible in the original array."""
        arr = np.zeros((10, 10, 3), dtype=np.uint8)
        img = Image.frombytes(arr)
        img.data[5, 5, 1] = 99
        assert arr[5, 5, 1] == 99

    def test_data_property_is_backing_buffer(self):
        """img.data returns the backing buffer (zero-copy access)."""
        arr = np.full((20, 20, 3), 100, dtype=np.uint8)
        img = Image.frombytes(arr)
        assert np.shares_memory(img.data, arr)

    def test_data_mutation_visible(self):
        """Mutations via img.data are visible (it's the real buffer)."""
        arr = np.full((20, 20, 3), 100, dtype=np.uint8)
        img = Image.frombytes(arr)
        img.data[0, 0, 0] = 42
        assert arr[0, 0, 0] == 42

    def test_to_numpy_isolates(self):
        """to_numpy() must copy — mutations must NOT affect original Image."""
        img = _make_test_image(fill=50)
        arr = img.to_numpy()
        arr[0, 0, 0] = 255
        assert img.data[0, 0, 0] == 50

    def test_copy_isolates(self):
        """copy() must produce a fully independent Image."""
        img = _make_test_image(fill=50)
        img2 = img.copy()
        img2.data[0, 0, 0] = 255
        assert img.data[0, 0, 0] == 50

    def test_frombytes_raw_bytes_owns_data(self):
        """frombytes from raw bytes (not numpy) creates its own buffer."""
        width, height, channels = 4, 4, 3
        raw = bytes([55] * (width * height * channels))
        img = Image.frombytes(raw, width=width, height=height, channels=channels)
        assert img.width == width
        assert img.height == height
        assert np.all(img.data == 55)
        # Raw bytes input: Image owns its copy, mutations to 'raw' won't matter
        assert img.data[0, 0, 0] == 55


class TestMemoryZeroCopy:
    """Exhaustive zero-copy and memory isolation tests."""

    # --- Constructor zero-copy ---

    def test_image_constructor_shares_memory(self):
        """Image(numpy_array) must be zero-copy for 3D arrays."""
        arr = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        img = Image(arr)
        assert np.shares_memory(img.data, arr)

    def test_image_constructor_2d_shares_memory(self):
        """Image(2D_array) expands dims via view — should share memory."""
        arr = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        img = Image(arr)
        assert img.channels == 1
        assert np.shares_memory(img.data, arr)

    def test_image_constructor_mutation_roundtrip(self):
        """Mutating source array must be visible through Image (zero-copy proof)."""
        arr = np.zeros((10, 10, 3), dtype=np.uint8)
        img = Image(arr)
        arr[3, 3, 0] = 200
        assert img.data[3, 3, 0] == 200
        img.data[7, 7, 2] = 150
        assert arr[7, 7, 2] == 150

    def test_frombytes_numpy_shares_memory(self):
        """frombytes(numpy_arr) must be zero-copy."""
        arr = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        img = Image.frombytes(arr)
        assert np.shares_memory(img.data, arr)

    def test_frombytes_2d_shares_memory(self):
        """frombytes(2D_array) should share memory."""
        arr = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        img = Image.frombytes(arr)
        assert img.channels == 1
        assert np.shares_memory(img.data, arr)

    def test_frombytes_raw_bytes_owns_copy(self):
        """frombytes(raw_bytes) creates its own buffer (can't share with immutable bytes)."""
        raw = bytes([55] * (4 * 4 * 3))
        img = Image.frombytes(raw, width=4, height=4, channels=3)
        assert np.all(img.data == 55)

    def test_frombytes_bytearray_roundtrip(self):
        """frombytes(bytearray) — verify data is accessible."""
        ba = bytearray([100] * (8 * 6 * 3))
        img = Image.frombytes(ba, width=8, height=6, channels=3)
        assert img.width == 8
        assert np.all(img.data == 100)

    # --- __array__ protocol ---

    def test_array_protocol_no_copy_shares_memory(self):
        """np.asarray(img) should share memory (zero-copy)."""
        arr = np.random.randint(0, 255, (30, 30, 3), dtype=np.uint8)
        img = Image(arr)
        out = np.asarray(img)
        assert np.shares_memory(out, arr)

    def test_array_protocol_copy_isolates(self):
        """np.array(img, copy=True) must NOT share memory."""
        arr = np.random.randint(0, 255, (30, 30, 3), dtype=np.uint8)
        img = Image(arr)
        out = np.array(img, copy=True)
        assert not np.shares_memory(out, arr)

    # --- data property ---

    def test_data_getter_shares_memory(self):
        """img.data must return the backing buffer, not a copy."""
        arr = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
        img = Image(arr)
        d1 = img.data
        d2 = img.data
        assert np.shares_memory(d1, d2)
        assert np.shares_memory(d1, arr)

    # --- copy/to_numpy isolation ---

    def test_copy_does_not_share_memory(self):
        """copy() must produce fully independent memory."""
        arr = np.full((20, 20, 3), 50, dtype=np.uint8)
        img = Image(arr)
        img2 = img.copy()
        assert not np.shares_memory(img.data, img2.data)
        img2.data[0, 0, 0] = 255
        assert img.data[0, 0, 0] == 50

    def test_to_numpy_does_not_share_memory(self):
        """to_numpy() must return an independent copy."""
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
        self.img = Image.frombytes(self.original_data.copy())
        self.snapshot = self.img.data.copy()

    def _assert_original_unchanged(self):
        assert np.array_equal(self.img.data, self.snapshot), \
            "Transform mutated the source Image!"

    def _assert_isolated(self, result):
        """Assert result has independent memory from source."""
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
        """A full chain of transforms must not touch the original."""
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
        """ColorJitter must not mutate the source Image."""
        jitter = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        result = jitter(self.img)
        self._assert_isolated(result)

    def test_augmentation_random_hflip_isolates(self):
        """RandomHorizontalFlip must not mutate source."""
        flip = RandomHorizontalFlip(p=1.0)
        result = flip(self.img)
        self._assert_isolated(result)

    def test_augmentation_random_vflip_isolates(self):
        """RandomVerticalFlip must not mutate source."""
        flip = RandomVerticalFlip(p=1.0)
        result = flip(self.img)
        self._assert_isolated(result)

    def test_augmentation_random_crop_isolates(self):
        """RandomCrop must not mutate source."""
        crop = RandomCrop((50, 50))
        result = crop(self.img)
        self._assert_isolated(result)

    def test_augmentation_random_rotation_isolates(self):
        """RandomRotation must not mutate source."""
        rot = RandomRotation(30.0)
        result = rot(self.img)
        self._assert_isolated(result)

    def test_compose_isolates(self):
        """Compose pipeline must not mutate the source Image."""
        transform = Compose([
            RandomHorizontalFlip(p=1.0),
            ColorJitter(brightness=0.2),
        ])
        result = transform(self.img)
        self._assert_isolated(result)


class TestScopedLifetime:
    """Verify Image data survives or is properly isolated across scopes."""

    def test_data_survives_image_going_out_of_scope(self):
        """np.array(img) should keep data alive even after Image is deleted."""
        img = _make_test_image(fill=42)
        arr = img.data  # get backing buffer
        del img  # Image goes out of scope
        # arr should still be valid (numpy owns the memory)
        assert arr[0, 0, 0] == 42
        assert arr.shape == (6, 8, 3)

    def test_frombytes_data_lifetime(self):
        """frombytes Image depends on source array lifetime."""
        arr = np.full((10, 10, 3), 77, dtype=np.uint8)
        img = Image.frombytes(arr)
        assert img.data[0, 0, 0] == 77
        # arr still alive, so img.data is valid
        arr[0, 0, 0] = 88
        assert img.data[0, 0, 0] == 88

    def test_transform_result_independent_of_source_scope(self):
        """Transform results must be valid even if source Image is deleted."""
        def create_and_transform():
            img = _make_test_image(width=100, height=80, fill=128)
            return img.resize(50, 40).flip_horizontal()

        result = create_and_transform()
        # Original img is out of scope, result should still work
        assert result.width == 50
        assert result.height == 40
        assert result.data[0, 0, 0] is not None  # data accessible

    def test_augmentation_result_independent(self):
        """Augmentation results must be valid after source is deleted."""
        from kornia_rs.augmentations import RandomHorizontalFlip

        def augment():
            img = _make_test_image(width=50, height=50, fill=100)
            flip = RandomHorizontalFlip(p=1.0)
            return flip(img)

        result = augment()
        assert result.width == 50
        assert result.data.sum() > 0

    def test_chained_intermediate_gc(self):
        """Intermediate Images in a chain should be GC-able."""
        import gc
        import weakref

        img = _make_test_image(width=100, height=80, fill=128)

        # Create intermediate and track it
        intermediate = img.resize(50, 40)
        weak_ref = weakref.ref(intermediate)

        # Chain further, dropping intermediate reference
        result = intermediate.flip_horizontal()
        del intermediate
        gc.collect()

        # Intermediate may or may not be collected (GC is non-deterministic)
        # But result must be fully valid regardless
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
        img = Image.frombytes(data)
        flipped = img.flip_horizontal()
        assert np.array_equal(flipped.data[0, 0], data[0, 1])
        assert np.array_equal(flipped.data[0, 1], data[0, 0])

    def test_flip_vertical(self):
        data = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        img = Image.frombytes(data)
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
        img = Image.frombytes(np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8))
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
        """Reset to non-deterministic after each test."""
        set_seed(None)

    def test_seed_reproducible_colorjitter(self):
        """Same seed produces identical ColorJitter results."""
        img = _make_test_image(width=50, height=50, channels=3, fill=128)
        jitter = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)

        set_seed(42)
        result1 = jitter(img)

        set_seed(42)
        result2 = jitter(img)

        assert np.array_equal(result1.data, result2.data)

    def test_seed_reproducible_random_flip(self):
        """Same seed produces identical flip decisions."""
        img = _make_test_image(width=50, height=50, channels=3, fill=128)
        flip = RandomHorizontalFlip(p=0.5)

        set_seed(123)
        results1 = [np.array(flip(img).data) for _ in range(10)]

        set_seed(123)
        results2 = [np.array(flip(img).data) for _ in range(10)]

        for r1, r2 in zip(results1, results2):
            assert np.array_equal(r1, r2)

    def test_seed_reproducible_random_crop(self):
        """Same seed produces identical crop positions."""
        img = _make_test_image(width=100, height=80, channels=3, fill=128)
        # Use non-uniform image so different crops give different data
        img_data = np.random.RandomState(0).randint(0, 255, (80, 100, 3), dtype=np.uint8)
        img = Image(img_data)
        crop = RandomCrop((50, 50))

        set_seed(99)
        result1 = crop(img)

        set_seed(99)
        result2 = crop(img)

        assert np.array_equal(result1.data, result2.data)

    def test_seed_reproducible_compose(self):
        """Same seed produces identical Compose pipeline results."""
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
        """Different seeds produce different results (with high probability)."""
        img_data = np.random.RandomState(0).randint(0, 255, (80, 100, 3), dtype=np.uint8)
        img = Image(img_data)
        jitter = ColorJitter(brightness=0.5, contrast=0.5)

        set_seed(1)
        result1 = jitter(img)

        set_seed(2)
        result2 = jitter(img)

        assert not np.array_equal(result1.data, result2.data)

    def test_reset_seed_none(self):
        """set_seed(None) resets to non-deterministic mode."""
        set_seed(42)
        set_seed(None)
        # Just verify it doesn't crash — non-deterministic means we can't
        # assert exact values, just that it runs
        img = _make_test_image(width=50, height=50, channels=3, fill=128)
        flip = RandomHorizontalFlip(p=0.5)
        result = flip(img)
        assert result.shape == img.shape

    def test_seed_reproducible_random_rotation(self):
        """Same seed produces identical rotation results."""
        img_data = np.random.RandomState(0).randint(0, 255, (80, 100, 3), dtype=np.uint8)
        img = Image(img_data)
        rot = RandomRotation(30.0)

        set_seed(55)
        result1 = rot(img)

        set_seed(55)
        result2 = rot(img)

        assert np.array_equal(result1.data, result2.data)


# --- Decode tests ---

class TestDecode:
    """Test Image.decode() for JPEG and PNG byte streams."""

    def _make_jpeg_bytes(self, width=16, height=16, channels=3, fill=100):
        """Save a small image to a temp file and read back as JPEG bytes."""
        arr = np.full((height, width, channels), fill, dtype=np.uint8)
        img = Image.frombytes(arr)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            tmp_path = f.name
        try:
            img.save(tmp_path)
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            os.unlink(tmp_path)

    def _make_png_bytes(self, width=16, height=16, channels=3, fill=100):
        """Save a small image to a temp file and read back as PNG bytes."""
        arr = np.full((height, width, channels), fill, dtype=np.uint8)
        img = Image.frombytes(arr)
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


# --- Multiprocessing tests ---

def _worker_process_image(data_tuple):
    """Worker function that runs in a separate process."""
    from kornia_rs.image import Image
    data, width, height = data_tuple
    img = Image.frombytes(data)
    result = img.resize(width, height).flip_horizontal()
    return result.data


class TestMultiprocessing:
    """Verify Image works safely across process boundaries."""

    def test_spawn_pool(self):
        """Image can be processed in a multiprocessing Pool with spawn context."""
        data = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        tasks = [(data.copy(), 50, 40) for _ in range(2)]

        # Use 'spawn' — safer for PyO3 extensions (avoids fork+GIL issues)
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(2) as pool:
            results = pool.map(_worker_process_image, tasks)

        assert len(results) == 2
        for r in results:
            assert r.shape == (40, 50, 3)
            assert r.dtype == np.uint8

    def test_concurrent_futures(self):
        """Image works with concurrent.futures ProcessPoolExecutor."""
        from concurrent.futures import ProcessPoolExecutor
        data = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        tasks = [(data.copy(), 50, 40) for _ in range(2)]

        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as executor:
            results = list(executor.map(_worker_process_image, tasks))

        assert len(results) == 2
        for r in results:
            assert r.shape == (40, 50, 3)
