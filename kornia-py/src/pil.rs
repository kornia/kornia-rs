use kornia_image::{allocator::CpuAllocator, Image as KorniaImage, ImageSize, color_spaces::Rgb8};
use kornia_io::{jpeg, png, tiff, jpegturbo};
use pyo3::prelude::*;

use crate::image::{FromPyImage, ToPyImage, PyImage};

/// A PIL-like Image class for kornia-rs
///
/// This class provides a Pillow (PIL) compatible API for image manipulation
/// using kornia-rs as the backend. It focuses on safety and zero-copy semantics
/// wherever possible.
///
/// # Examples
///
/// ```python
/// import kornia_rs as K
///
/// # Load an image
/// img = K.Image.open("path/to/image.jpg")
///
/// # Access image properties
/// width, height = img.size
/// print(f"Mode: {img.mode}")
///
/// # Resize the image
/// resized = img.resize((128, 128))
///
/// # Save the image
/// img.save("output.jpg")
/// ```
#[pyclass(name = "Image")]
pub struct PyImage3 {
    // Internal storage - we store as u8/3 channel by default
    // For now, we'll keep it simple with RGB u8
    inner: KorniaImage<u8, 3, CpuAllocator>,
}

#[pymethods]
impl PyImage3 {
    /// Create a new image from a file path.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the image file
    ///
    /// # Returns
    ///
    /// A new Image instance
    ///
    /// # Examples
    ///
    /// ```python
    /// import kornia_rs as K
    /// img = K.Image.open("dog.jpeg")
    /// print(img.size)  # (width, height)
    /// ```
    #[staticmethod]
    pub fn open(path: &str) -> PyResult<Self> {
        let rgb8 = kornia_io::functional::read_image_any_rgb8(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to open image: {}", e)))?;
        
        Ok(PyImage3 { inner: rgb8.into_inner() })
    }

    /// Create a new image with the given mode and size, filled with the specified color.
    ///
    /// # Arguments
    ///
    /// * `mode` - Image mode (e.g., "RGB", "L", "RGBA")
    /// * `size` - Tuple of (width, height)
    /// * `color` - Fill color (int for grayscale, tuple for RGB/RGBA), defaults to 0
    ///
    /// # Returns
    ///
    /// A new Image instance
    ///
    /// # Examples
    ///
    /// ```python
    /// import kornia_rs as K
    /// # Create a black RGB image
    /// img = K.Image.new("RGB", (640, 480))
    /// # Create a red RGB image
    /// img = K.Image.new("RGB", (640, 480), color=(255, 0, 0))
    /// ```
    #[staticmethod]
    #[pyo3(signature = (mode, size, color=None))]
    pub fn new(mode: &str, size: (usize, usize), color: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let image_size = ImageSize {
            width: size.0,
            height: size.1,
        };

        match mode {
            "RGB" => {
                let fill_value = if let Some(c) = color {
                    if let Ok(tuple) = c.extract::<(u8, u8, u8)>() {
                        // For RGB, we need to fill with pattern [R, G, B, R, G, B, ...]
                        let mut data = Vec::with_capacity(size.0 * size.1 * 3);
                        for _ in 0..(size.0 * size.1) {
                            data.push(tuple.0);
                            data.push(tuple.1);
                            data.push(tuple.2);
                        }
                        return Ok(PyImage3 {
                            inner: KorniaImage::new(image_size, data, CpuAllocator)
                                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to create image: {}", e)))?,
                        });
                    } else if let Ok(val) = c.extract::<u8>() {
                        val
                    } else {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "Color must be an integer or tuple of (R, G, B)"
                        ));
                    }
                } else {
                    0
                };

                let image = KorniaImage::from_size_val(image_size, fill_value, CpuAllocator)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to create image: {}", e)))?;
                
                Ok(PyImage3 { inner: image })
            }
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                format!("Unsupported mode: {}. Currently only 'RGB' is supported.", mode)
            )),
        }
    }

    /// Save the image to a file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path where to save the image
    /// * `quality` - JPEG quality (1-100), only for JPEG format
    ///
    /// # Examples
    ///
    /// ```python
    /// import kornia_rs as K
    /// img = K.Image.open("input.jpg")
    /// img.save("output.jpg", quality=95)
    /// img.save("output.png")
    /// ```
    #[pyo3(signature = (path, quality=None))]
    pub fn save(&self, path: &str, quality: Option<u8>) -> PyResult<()> {
        // Determine format from extension
        let path_lower = path.to_lowercase();
        
        if path_lower.ends_with(".jpg") || path_lower.ends_with(".jpeg") {
            let quality = quality.unwrap_or(95);
            jpeg::write_image_jpeg_rgb8(path, &self.inner, quality)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to save JPEG: {}", e)))?;
        } else if path_lower.ends_with(".png") {
            png::write_image_png_rgb8(path, &self.inner)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to save PNG: {}", e)))?;
        } else if path_lower.ends_with(".tif") || path_lower.ends_with(".tiff") {
            tiff::write_image_tiff_rgb8(path, &self.inner)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to save TIFF: {}", e)))?;
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Unsupported file format. Supported: .jpg, .jpeg, .png, .tif, .tiff")
            ));
        }
        
        Ok(())
    }

    /// Get the size of the image as (width, height).
    ///
    /// # Returns
    ///
    /// Tuple of (width, height)
    #[getter]
    pub fn size(&self) -> (usize, usize) {
        (self.inner.width(), self.inner.height())
    }

    /// Get the image mode.
    ///
    /// # Returns
    ///
    /// Image mode as string (e.g., "RGB")
    #[getter]
    pub fn mode(&self) -> &str {
        // For now, we only support RGB
        "RGB"
    }

    /// Get the width of the image.
    ///
    /// # Returns
    ///
    /// Width in pixels
    #[getter]
    pub fn width(&self) -> usize {
        self.inner.width()
    }

    /// Get the height of the image.
    ///
    /// # Returns
    ///
    /// Height in pixels
    #[getter]
    pub fn height(&self) -> usize {
        self.inner.height()
    }

    /// Resize the image to the given size.
    ///
    /// # Arguments
    ///
    /// * `size` - Target size as (width, height)
    /// * `interpolation` - Interpolation mode ("nearest", "bilinear"), defaults to "bilinear"
    ///
    /// # Returns
    ///
    /// A new resized Image
    ///
    /// # Examples
    ///
    /// ```python
    /// import kornia_rs as K
    /// img = K.Image.open("image.jpg")
    /// small = img.resize((128, 128))
    /// ```
    #[pyo3(signature = (size, interpolation="bilinear"))]
    pub fn resize(&self, size: (usize, usize), interpolation: &str) -> PyResult<Self> {
        let new_size = ImageSize {
            width: size.0,
            height: size.1,
        };

        let interp = match interpolation {
            "nearest" => kornia_imgproc::interpolation::InterpolationMode::Nearest,
            "bilinear" => kornia_imgproc::interpolation::InterpolationMode::Bilinear,
            _ => return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Unsupported interpolation mode: {}. Use 'nearest' or 'bilinear'.", interpolation)
            )),
        };

        let mut resized = KorniaImage::from_size_val(new_size, 0u8, CpuAllocator)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to create output image: {}", e)))?;

        kornia_imgproc::resize::resize_fast_rgb(&self.inner, &mut resized, interp)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to resize: {}", e)))?;

        Ok(PyImage3 { inner: resized })
    }

    /// Create a copy of the image.
    ///
    /// # Returns
    ///
    /// A new Image with copied data
    ///
    /// # Examples
    ///
    /// ```python
    /// import kornia_rs as K
    /// img = K.Image.open("image.jpg")
    /// img_copy = img.copy()
    /// ```
    pub fn copy(&self) -> PyResult<Self> {
        Ok(PyImage3 {
            inner: self.inner.clone(),
        })
    }

    /// Get the pixel value at coordinates (x, y).
    ///
    /// # Arguments
    ///
    /// * `xy` - Tuple of (x, y) coordinates
    ///
    /// # Returns
    ///
    /// Pixel value as tuple (R, G, B) for RGB mode
    ///
    /// # Examples
    ///
    /// ```python
    /// import kornia_rs as K
    /// img = K.Image.open("image.jpg")
    /// r, g, b = img.getpixel((10, 20))
    /// ```
    pub fn getpixel(&self, xy: (usize, usize)) -> PyResult<(u8, u8, u8)> {
        let (x, y) = xy;
        
        if x >= self.inner.width() || y >= self.inner.height() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                format!("Pixel coordinates ({}, {}) out of bounds for image of size ({}, {})",
                    x, y, self.inner.width(), self.inner.height())
            ));
        }

        let r = *self.inner.get_pixel(x, y, 0)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to get pixel: {}", e)))?;
        let g = *self.inner.get_pixel(x, y, 1)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to get pixel: {}", e)))?;
        let b = *self.inner.get_pixel(x, y, 2)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to get pixel: {}", e)))?;

        Ok((r, g, b))
    }

    /// Set the pixel value at coordinates (x, y).
    ///
    /// # Arguments
    ///
    /// * `xy` - Tuple of (x, y) coordinates
    /// * `value` - Pixel value as tuple (R, G, B) for RGB mode or integer
    ///
    /// # Examples
    ///
    /// ```python
    /// import kornia_rs as K
    /// img = K.Image.open("image.jpg")
    /// img.putpixel((10, 20), (255, 0, 0))  # Set pixel to red
    /// ```
    pub fn putpixel(&mut self, xy: (usize, usize), value: &Bound<'_, PyAny>) -> PyResult<()> {
        let (x, y) = xy;
        
        if x >= self.inner.width() || y >= self.inner.height() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                format!("Pixel coordinates ({}, {}) out of bounds for image of size ({}, {})",
                    x, y, self.inner.width(), self.inner.height())
            ));
        }

        if let Ok(tuple) = value.extract::<(u8, u8, u8)>() {
            self.inner.set_pixel(x, y, 0, tuple.0)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to set pixel: {}", e)))?;
            self.inner.set_pixel(x, y, 1, tuple.1)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to set pixel: {}", e)))?;
            self.inner.set_pixel(x, y, 2, tuple.2)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to set pixel: {}", e)))?;
        } else if let Ok(val) = value.extract::<u8>() {
            self.inner.set_pixel(x, y, 0, val)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to set pixel: {}", e)))?;
            self.inner.set_pixel(x, y, 1, val)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to set pixel: {}", e)))?;
            self.inner.set_pixel(x, y, 2, val)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to set pixel: {}", e)))?;
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Value must be an integer or tuple of (R, G, B)"
            ));
        }

        Ok(())
    }

    /// Convert the image to a NumPy array (zero-copy when possible).
    ///
    /// # Returns
    ///
    /// NumPy array of shape (height, width, 3) with dtype uint8
    ///
    /// # Examples
    ///
    /// ```python
    /// import kornia_rs as K
    /// img = K.Image.open("image.jpg")
    /// arr = img.to_numpy()
    /// print(arr.shape)  # (height, width, 3)
    /// ```
    pub fn to_numpy(&self) -> PyResult<PyImage> {
        self.inner.clone().to_pyimage()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to convert to numpy: {}", e)))
    }

    /// Create an Image from a NumPy array.
    ///
    /// # Arguments
    ///
    /// * `array` - NumPy array of shape (height, width, 3) with dtype uint8
    ///
    /// # Returns
    ///
    /// A new Image instance
    ///
    /// # Examples
    ///
    /// ```python
    /// import kornia_rs as K
    /// import numpy as np
    /// arr = np.zeros((480, 640, 3), dtype=np.uint8)
    /// img = K.Image.from_numpy(arr)
    /// ```
    #[staticmethod]
    pub fn from_numpy(array: PyImage) -> PyResult<Self> {
        let image = KorniaImage::<u8, 3, CpuAllocator>::from_pyimage(array)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to create image from numpy: {}", e)))?;
        
        Ok(PyImage3 { inner: image })
    }

    /// String representation of the Image.
    fn __repr__(&self) -> String {
        format!(
            "Image(mode='{}', size=({}, {}))",
            self.mode(),
            self.inner.width(),
            self.inner.height()
        )
    }

    /// String representation of the Image.
    fn __str__(&self) -> String {
        self.__repr__()
    }
}
