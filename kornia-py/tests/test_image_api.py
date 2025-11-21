"""Tests for the high-level Image API"""
from __future__ import annotations
import tempfile
from pathlib import Path
import kornia_rs as K
import numpy as np
import pytest

# TODO: inject this from elsewhere
DATA_DIR = Path(__file__).parents[2] / "tests" / "data"


class TestImageOpen:
    """Test Image.open() functionality"""

    def test_open_jpeg_rgb(self):
        """Test opening a JPEG image as RGB"""
        img_path = DATA_DIR / "dog.jpeg"
        img = K.Image.open(str(img_path))
        
        assert img.mode == "RGB"
        assert img.width == 258
        assert img.height == 195
        assert img.size.width == 258
        assert img.size.height == 195

    def test_open_png(self):
        """Test opening a PNG image"""
        img_path = DATA_DIR / "dog-rgb8.png"
        img = K.Image.open(str(img_path))
        
        assert img.mode == "RGB"
        assert isinstance(img.size, K.ImageSize)


class TestImageNew:
    """Test Image.new() functionality"""

    def test_new_rgb_default(self):
        """Test creating a new RGB image with default color"""
        img = K.Image.new("RGB", (640, 480))
        
        assert img.mode == "RGB"
        assert img.width == 640
        assert img.height == 480
        assert img.size.width == 640
        assert img.size.height == 480

    def test_new_rgb_with_tuple_color(self):
        """Test creating a new RGB image with tuple color"""
        img = K.Image.new("RGB", (100, 100), color=(255, 128, 64))
        
        assert img.mode == "RGB"
        assert img.size.width == 100
        assert img.size.height == 100
        
        # Check a pixel has the right color
        pixel = img.getpixel((50, 50))
        assert pixel == (255, 128, 64)

    def test_new_rgb_with_int_color(self):
        """Test creating a new RGB image with integer color"""
        img = K.Image.new("RGB", (100, 100), color=128)
        
        pixel = img.getpixel((10, 10))
        assert pixel == (128, 128, 128)

    def test_new_rgba_with_tuple_color(self):
        """Test creating a new RGBA image"""
        img = K.Image.new("RGBA", (50, 50), color=(255, 0, 0, 128))
        
        assert img.mode == "RGBA"
        assert img.size.width == 50
        assert img.size.height == 50
        
        pixel = img.getpixel((10, 10))
        assert pixel == (255, 0, 0, 128)

    def test_new_grayscale(self):
        """Test creating a new grayscale image"""
        img = K.Image.new("L", (80, 60), color=128)
        
        assert img.mode == "L"
        assert img.size.width == 80
        assert img.size.height == 60
        
        pixel = img.getpixel((10, 10))
        assert pixel == 128

    def test_new_invalid_mode(self):
        """Test creating image with invalid mode raises error"""
        with pytest.raises(ValueError):
            K.Image.new("INVALID", (100, 100))


class TestImageSave:
    """Test Image.save() functionality"""

    def test_save_rgb_as_jpeg(self):
        """Test saving RGB image as JPEG"""
        img = K.Image.new("RGB", (100, 100), color=(128, 64, 32))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jpg"
            img.save(str(path), quality=95)
            assert path.exists()
            
            # Load back and verify
            loaded = K.Image.open(str(path))
            assert loaded.mode == "RGB"
            assert loaded.size.width == 100
            assert loaded.size.height == 100

    def test_save_rgb_as_png(self):
        """Test saving RGB image as PNG"""
        img = K.Image.new("RGB", (100, 100), color=(255, 0, 0))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.png"
            img.save(str(path))
            assert path.exists()

    def test_save_grayscale_as_jpeg(self):
        """Test saving grayscale image as JPEG"""
        img = K.Image.new("L", (100, 100), color=200)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jpg"
            img.save(str(path))
            assert path.exists()

    def test_save_rgba_as_png(self):
        """Test saving RGBA image as PNG"""
        img = K.Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.png"
            img.save(str(path))
            assert path.exists()

    def test_save_rgba_as_jpeg_fails(self):
        """Test that saving RGBA as JPEG raises error"""
        img = K.Image.new("RGBA", (100, 100))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jpg"
            with pytest.raises(ValueError, match="JPEG.*does not support RGBA"):
                img.save(str(path))


class TestImageResize:
    """Test Image.resize() functionality"""

    def test_resize_rgb(self):
        """Test resizing RGB image"""
        img = K.Image.new("RGB", (200, 200))
        resized = img.resize((100, 100))
        
        assert resized.mode == "RGB"
        assert resized.size.width == 100
        assert resized.size.height == 100

    def test_resize_rgb_different_aspect(self):
        """Test resizing RGB image to different aspect ratio"""
        img = K.Image.new("RGB", (200, 100))
        resized = img.resize((50, 150))
        
        assert resized.size.width == 50
        assert resized.size.height == 150

    def test_resize_grayscale(self):
        """Test resizing grayscale image"""
        img = K.Image.new("L", (200, 200))
        resized = img.resize((80, 80))
        
        assert resized.mode == "L"
        assert resized.size.width == 80
        assert resized.size.height == 80

    def test_resize_with_interpolation(self):
        """Test resizing with different interpolation modes"""
        img = K.Image.new("RGB", (100, 100))
        
        bilinear = img.resize((50, 50), interpolation="bilinear")
        assert bilinear.size.width == 50
        
        nearest = img.resize((50, 50), interpolation="nearest")
        assert nearest.size.width == 50


class TestImagePixelAccess:
    """Test pixel get/set operations"""

    def test_getpixel_rgb(self):
        """Test getting pixel from RGB image"""
        img = K.Image.new("RGB", (100, 100), color=(255, 128, 64))
        pixel = img.getpixel((50, 50))
        
        assert isinstance(pixel, tuple)
        assert len(pixel) == 3
        assert pixel == (255, 128, 64)

    def test_getpixel_rgba(self):
        """Test getting pixel from RGBA image"""
        img = K.Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        pixel = img.getpixel((50, 50))
        
        assert isinstance(pixel, tuple)
        assert len(pixel) == 4
        assert pixel == (255, 0, 0, 128)

    def test_getpixel_grayscale(self):
        """Test getting pixel from grayscale image"""
        img = K.Image.new("L", (100, 100), color=200)
        pixel = img.getpixel((50, 50))
        
        assert isinstance(pixel, int)
        assert pixel == 200

    def test_putpixel_rgb_tuple(self):
        """Test setting pixel in RGB image with tuple"""
        img = K.Image.new("RGB", (100, 100))
        img.putpixel((50, 50), (255, 0, 0))
        
        pixel = img.getpixel((50, 50))
        assert pixel == (255, 0, 0)

    def test_putpixel_rgb_int(self):
        """Test setting pixel in RGB image with integer"""
        img = K.Image.new("RGB", (100, 100))
        img.putpixel((50, 50), 200)
        
        pixel = img.getpixel((50, 50))
        assert pixel == (200, 200, 200)

    def test_putpixel_grayscale(self):
        """Test setting pixel in grayscale image"""
        img = K.Image.new("L", (100, 100))
        img.putpixel((50, 50), 150)
        
        pixel = img.getpixel((50, 50))
        assert pixel == 150

    def test_getpixel_out_of_bounds(self):
        """Test getting pixel out of bounds raises error"""
        img = K.Image.new("RGB", (100, 100))
        
        with pytest.raises(IndexError):
            img.getpixel((200, 50))

    def test_putpixel_out_of_bounds(self):
        """Test setting pixel out of bounds raises error"""
        img = K.Image.new("RGB", (100, 100))
        
        with pytest.raises(IndexError):
            img.putpixel((200, 50), (255, 0, 0))


class TestImageCopy:
    """Test Image.copy() functionality"""

    def test_copy_creates_independent_image(self):
        """Test that copy creates an independent image"""
        img = K.Image.new("RGB", (100, 100), color=(255, 0, 0))
        img_copy = img.copy()
        
        # Modify the copy
        img_copy.putpixel((50, 50), (0, 255, 0))
        
        # Original should be unchanged
        assert img.getpixel((50, 50)) == (255, 0, 0)
        assert img_copy.getpixel((50, 50)) == (0, 255, 0)


class TestImageNumpyConversion:
    """Test NumPy array conversions"""

    def test_to_numpy_rgb(self):
        """Test converting RGB image to NumPy array"""
        img = K.Image.new("RGB", (100, 80), color=(128, 64, 32))
        arr = img.to_numpy()
        
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (80, 100, 3)
        assert arr.dtype == np.uint8
        # Check pixel value
        assert tuple(arr[40, 50]) == (128, 64, 32)

    def test_from_numpy_rgb(self):
        """Test creating image from NumPy array"""
        arr = np.zeros((50, 60, 3), dtype=np.uint8)
        arr[:, :] = [200, 150, 100]
        
        img = K.Image.from_numpy(arr)
        
        assert img.mode == "RGB"
        assert img.size.width == 60
        assert img.size.height == 50
        
        pixel = img.getpixel((30, 25))
        assert pixel == (200, 150, 100)

    def test_numpy_roundtrip(self):
        """Test NumPy conversion roundtrip"""
        original = K.Image.new("RGB", (100, 80), color=(128, 64, 32))
        arr = original.to_numpy()
        reconstructed = K.Image.from_numpy(arr)
        
        assert reconstructed.mode == original.mode
        assert reconstructed.size.width == original.size.width
        assert reconstructed.size.height == original.size.height
        
        # Check pixel values match
        assert reconstructed.getpixel((50, 40)) == original.getpixel((50, 40))


class TestImageProperties:
    """Test Image properties"""

    def test_repr(self):
        """Test string representation"""
        img = K.Image.new("RGB", (640, 480))
        repr_str = repr(img)
        
        assert "Image" in repr_str
        assert "RGB" in repr_str
        assert "640" in repr_str
        assert "480" in repr_str

    def test_size_is_imagesize(self):
        """Test that size property returns ImageSize"""
        img = K.Image.new("RGB", (640, 480))
        size = img.size
        
        assert isinstance(size, K.ImageSize)
        assert size.width == 640
        assert size.height == 480
