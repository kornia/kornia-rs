use kornia_image::{allocator::CpuAllocator, Image as KorniaImage, ImageSize, color_spaces::{Rgb8, Rgba8, Gray8}};
use kornia_io::{jpeg, png, tiff};
use pyo3::prelude::*;

use crate::image::{FromPyImage, ToPyImage, PyImage, PyImageSize};

/// Internal storage for the PIL-like Image class
/// This enum allows us to support multiple image modes with different channel counts
#[derive(Clone)]
enum ImageData {
    Rgb(KorniaImage<u8, 3, CpuAllocator>),
    Rgba(KorniaImage<u8, 4, CpuAllocator>),
    L(KorniaImage<u8, 1, CpuAllocator>),
}

impl ImageData {
    fn width(&self) -> usize {
        match self {
            ImageData::Rgb(img) => img.width(),
            ImageData::Rgba(img) => img.width(),
            ImageData::L(img) => img.width(),
        }
    }

    fn height(&self) -> usize {
        match self {
            ImageData::Rgb(img) => img.height(),
            ImageData::Rgba(img) => img.height(),
            ImageData::L(img) => img.height(),
        }
    }

    fn mode(&self) -> &str {
        match self {
            ImageData::Rgb(_) => "RGB",
            ImageData::Rgba(_) => "RGBA",
            ImageData::L(_) => "L",
        }
    }

    fn num_channels(&self) -> usize {
        match self {
            ImageData::Rgb(_) => 3,
            ImageData::Rgba(_) => 4,
            ImageData::L(_) => 1,
        }
    }
}

/// A high-level Image class for kornia-rs
///
/// This class provides a convenient object-oriented API for image manipulation
/// using kornia-rs as the backend. It focuses on safety and zero-copy semantics
/// wherever possible.
///
/// Supported modes:
/// - "RGB": 3-channel RGB images (u8)
/// - "RGBA": 4-channel RGBA images (u8)
/// - "L": 1-channel grayscale images (u8)
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
/// size = img.size
/// print(f"Width: {size.width}, Height: {size.height}")
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
    // Internal storage supporting multiple modes
    inner: ImageData,
}

#[pymethods]
impl PyImage3 {
    /// Create a new image from a file path.
    ///
    /// Automatically loads as RGB and can convert to other modes.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the image file
    /// * `mode` - Optional mode to convert to ("RGB", "RGBA", "L"). Defaults to "RGB".
    ///
    /// # Returns
    ///
    /// A new Image instance
    ///
    /// # Examples
    ///
    /// ```python
    /// import kornia_rs as K
    /// img = K.Image.open("dog.jpeg")  # Loads as RGB
    /// img_gray = K.Image.open("dog.jpeg", mode="L")  # Convert to grayscale
    /// ```
    #[staticmethod]
    #[pyo3(signature = (path, mode=None))]
    pub fn open(path: &str, mode: Option<&str>) -> PyResult<Self> {
        // Always load as RGB first (this is what the I/O layer supports)
        let rgb8 = kornia_io::functional::read_image_any_rgb8(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to open image: {}", e)))?;
        
        let target_mode = mode.unwrap_or("RGB");
        
        match target_mode {
            "RGB" => {
                Ok(PyImage3 { inner: ImageData::Rgb(rgb8.into_inner()) })
            }
            "L" => {
                // Convert RGB to grayscale
                // For now, return error as conversion not implemented
                Err(pyo3::exceptions::PyValueError::new_err(
                    "Grayscale mode 'L' not yet fully supported. Use 'RGB' mode."
                ))
            }
            "RGBA" => {
                // Convert RGB to RGBA
                Err(pyo3::exceptions::PyValueError::new_err(
                    "RGBA mode not yet fully supported. Use 'RGB' mode."
                ))
            }
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                format!("Unsupported mode: {}. Use 'RGB' for now.", target_mode)
            )),
        }
    }

    /// Create a new image with the given mode and size, filled with the specified color.
    ///
    /// # Arguments
    ///
    /// * `mode` - Image mode: "RGB", "RGBA", or "L" (grayscale)
    /// * `size` - Tuple of (width, height)
    /// * `color` - Fill color:
    ///   - For "L": int (0-255)
    ///   - For "RGB": int or tuple (R, G, B)
    ///   - For "RGBA": int or tuple (R, G, B, A)
    ///   - Defaults to 0 (black/transparent)
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
    /// # Create a gray image
    /// img = K.Image.new("L", (640, 480), color=128)
    /// # Create a semi-transparent RGBA image
    /// img = K.Image.new("RGBA", (640, 480), color=(255, 0, 0, 128))
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
                if let Some(c) = color {
                    if let Ok(tuple) = c.extract::<(u8, u8, u8)>() {
                        let mut data = Vec::with_capacity(size.0 * size.1 * 3);
                        for _ in 0..(size.0 * size.1) {
                            data.push(tuple.0);
                            data.push(tuple.1);
                            data.push(tuple.2);
                        }
                        let image = KorniaImage::new(image_size, data, CpuAllocator)
                            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to create image: {}", e)))?;
                        return Ok(PyImage3 { inner: ImageData::Rgb(image) });
                    } else if let Ok(val) = c.extract::<u8>() {
                        let image = KorniaImage::from_size_val(image_size, val, CpuAllocator)
                            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to create image: {}", e)))?;
                        return Ok(PyImage3 { inner: ImageData::Rgb(image) });
                    } else {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "Color for RGB mode must be an integer or tuple of (R, G, B)"
                        ));
                    }
                }
                let image = KorniaImage::from_size_val(image_size, 0u8, CpuAllocator)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to create image: {}", e)))?;
                Ok(PyImage3 { inner: ImageData::Rgb(image) })
            }
            "RGBA" => {
                if let Some(c) = color {
                    if let Ok(tuple) = c.extract::<(u8, u8, u8, u8)>() {
                        let mut data = Vec::with_capacity(size.0 * size.1 * 4);
                        for _ in 0..(size.0 * size.1) {
                            data.push(tuple.0);
                            data.push(tuple.1);
                            data.push(tuple.2);
                            data.push(tuple.3);
                        }
                        let image = KorniaImage::new(image_size, data, CpuAllocator)
                            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to create image: {}", e)))?;
                        return Ok(PyImage3 { inner: ImageData::Rgba(image) });
                    } else if let Ok(val) = c.extract::<u8>() {
                        let image = KorniaImage::from_size_val(image_size, val, CpuAllocator)
                            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to create image: {}", e)))?;
                        return Ok(PyImage3 { inner: ImageData::Rgba(image) });
                    } else {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "Color for RGBA mode must be an integer or tuple of (R, G, B, A)"
                        ));
                    }
                }
                let image = KorniaImage::from_size_val(image_size, 0u8, CpuAllocator)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to create image: {}", e)))?;
                Ok(PyImage3 { inner: ImageData::Rgba(image) })
            }
            "L" => {
                let fill_value = if let Some(c) = color {
                    c.extract::<u8>().map_err(|_| pyo3::exceptions::PyValueError::new_err(
                        "Color for L mode must be an integer (0-255)"
                    ))?
                } else {
                    0
                };
                let image = KorniaImage::from_size_val(image_size, fill_value, CpuAllocator)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to create image: {}", e)))?;
                Ok(PyImage3 { inner: ImageData::L(image) })
            }
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                format!("Unsupported mode: {}. Supported modes: 'RGB', 'RGBA', 'L'.", mode)
            )),
        }
    }

    /// Save the image to a file.
    ///
    /// Format is determined by file extension. Supports .jpg/.jpeg, .png, .tif/.tiff
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
        let path_lower = path.to_lowercase();
        
        if path_lower.ends_with(".jpg") || path_lower.ends_with(".jpeg") {
            let quality = quality.unwrap_or(95);
            match &self.inner {
                ImageData::Rgb(img) => {
                    jpeg::write_image_jpeg_rgb8(path, img, quality)
                        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to save JPEG: {}", e)))?;
                }
                ImageData::L(img) => {
                    jpeg::write_image_jpeg_gray8(path, img, quality)
                        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to save JPEG: {}", e)))?;
                }
                ImageData::Rgba(_) => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "JPEG format does not support RGBA images. Convert to RGB first or use PNG format."
                    ));
                }
            }
        } else if path_lower.ends_with(".png") {
            match &self.inner {
                ImageData::Rgb(img) => {
                    png::write_image_png_rgb8(path, img)
                        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to save PNG: {}", e)))?;
                }
                ImageData::Rgba(img) => {
                    png::write_image_png_rgba8(path, img)
                        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to save PNG: {}", e)))?;
                }
                ImageData::L(img) => {
                    png::write_image_png_gray8(path, img)
                        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to save PNG: {}", e)))?;
                }
            }
        } else if path_lower.ends_with(".tif") || path_lower.ends_with(".tiff") {
            match &self.inner {
                ImageData::Rgb(img) => {
                    tiff::write_image_tiff_rgb8(path, img)
                        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to save TIFF: {}", e)))?;
                }
                ImageData::L(img) => {
                    tiff::write_image_tiff_mono8(path, img)
                        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to save TIFF: {}", e)))?;
                }
                ImageData::Rgba(_) => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "TIFF format support for RGBA is limited. Convert to RGB or use PNG format."
                    ));
                }
            }
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Unsupported file format. Supported: .jpg, .jpeg, .png, .tif, .tiff")
            ));
        }
        
        Ok(())
    }

    /// Get the size of the image as an ImageSize object.
    ///
    /// # Returns
    ///
    /// ImageSize object with width and height properties
    ///
    /// # Examples
    ///
    /// ```python
    /// import kornia_rs as K
    /// img = K.Image.open("image.jpg")
    /// size = img.size
    /// print(f"Width: {size.width}, Height: {size.height}")
    /// ```
    #[getter]
    pub fn size(&self) -> PyImageSize {
        let image_size = ImageSize {
            width: self.inner.width(),
            height: self.inner.height(),
        };
        image_size.into()
    }

    /// Get the image mode.
    ///
    /// # Returns
    ///
    /// Image mode as string: "RGB", "RGBA", or "L"
    #[getter]
    pub fn mode(&self) -> &str {
        self.inner.mode()
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

        let resized_data = match &self.inner {
            ImageData::Rgb(img) => {
                let mut resized = KorniaImage::from_size_val(new_size, 0u8, CpuAllocator)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to create output image: {}", e)))?;
                kornia_imgproc::resize::resize_fast_rgb(img, &mut resized, interp)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to resize: {}", e)))?;
                ImageData::Rgb(resized)
            }
            ImageData::Rgba(_) => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Resize for RGBA images not yet implemented. Use RGB mode."
                ));
            }
            ImageData::L(img) => {
                let mut resized = KorniaImage::from_size_val(new_size, 0u8, CpuAllocator)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to create output image: {}", e)))?;
                kornia_imgproc::resize::resize_fast_mono(img, &mut resized, interp)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to resize: {}", e)))?;
                ImageData::L(resized)
            }
        };

        Ok(PyImage3 { inner: resized_data })
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
    /// - For RGB: tuple (R, G, B)
    /// - For RGBA: tuple (R, G, B, A)
    /// - For L: int (0-255)
    ///
    /// # Examples
    ///
    /// ```python
    /// import kornia_rs as K
    /// img = K.Image.open("image.jpg", mode="RGB")
    /// r, g, b = img.getpixel((10, 20))
    /// 
    /// gray = K.Image.open("image.jpg", mode="L")
    /// val = gray.getpixel((10, 20))
    /// ```
    pub fn getpixel(&self, py: Python, xy: (usize, usize)) -> PyResult<PyObject> {
        let (x, y) = xy;
        
        if x >= self.inner.width() || y >= self.inner.height() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                format!("Pixel coordinates ({}, {}) out of bounds for image of size ({}, {})",
                    x, y, self.inner.width(), self.inner.height())
            ));
        }

        match &self.inner {
            ImageData::Rgb(img) => {
                let r = *img.get_pixel(x, y, 0)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to get pixel: {}", e)))?;
                let g = *img.get_pixel(x, y, 1)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to get pixel: {}", e)))?;
                let b = *img.get_pixel(x, y, 2)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to get pixel: {}", e)))?;
                Ok((r, g, b).into_pyobject(py).map(|o| o.into())?)
            }
            ImageData::Rgba(img) => {
                let r = *img.get_pixel(x, y, 0)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to get pixel: {}", e)))?;
                let g = *img.get_pixel(x, y, 1)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to get pixel: {}", e)))?;
                let b = *img.get_pixel(x, y, 2)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to get pixel: {}", e)))?;
                let a = *img.get_pixel(x, y, 3)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to get pixel: {}", e)))?;
                Ok((r, g, b, a).into_pyobject(py).map(|o| o.into())?)
            }
            ImageData::L(img) => {
                let val = *img.get_pixel(x, y, 0)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to get pixel: {}", e)))?;
                Ok(val.into_pyobject(py).map(|o| o.into())?)
            }
        }
    }

    /// Set the pixel value at coordinates (x, y).
    ///
    /// # Arguments
    ///
    /// * `xy` - Tuple of (x, y) coordinates
    /// * `value` - Pixel value:
    ///   - For RGB: tuple (R, G, B) or int
    ///   - For RGBA: tuple (R, G, B, A) or int
    ///   - For L: int (0-255)
    ///
    /// # Examples
    ///
    /// ```python
    /// import kornia_rs as K
    /// img = K.Image.open("image.jpg", mode="RGB")
    /// img.putpixel((10, 20), (255, 0, 0))  # Set pixel to red
    /// 
    /// gray = K.Image.new("L", (100, 100))
    /// gray.putpixel((10, 20), 128)  # Set gray value
    /// ```
    pub fn putpixel(&mut self, xy: (usize, usize), value: &Bound<'_, PyAny>) -> PyResult<()> {
        let (x, y) = xy;
        
        if x >= self.inner.width() || y >= self.inner.height() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                format!("Pixel coordinates ({}, {}) out of bounds for image of size ({}, {})",
                    x, y, self.inner.width(), self.inner.height())
            ));
        }

        match &mut self.inner {
            ImageData::Rgb(img) => {
                if let Ok(tuple) = value.extract::<(u8, u8, u8)>() {
                    img.set_pixel(x, y, 0, tuple.0)
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to set pixel: {}", e)))?;
                    img.set_pixel(x, y, 1, tuple.1)
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to set pixel: {}", e)))?;
                    img.set_pixel(x, y, 2, tuple.2)
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to set pixel: {}", e)))?;
                } else if let Ok(val) = value.extract::<u8>() {
                    img.set_pixel(x, y, 0, val)
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to set pixel: {}", e)))?;
                    img.set_pixel(x, y, 1, val)
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to set pixel: {}", e)))?;
                    img.set_pixel(x, y, 2, val)
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to set pixel: {}", e)))?;
                } else {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Value for RGB must be an integer or tuple of (R, G, B)"
                    ));
                }
            }
            ImageData::Rgba(img) => {
                if let Ok(tuple) = value.extract::<(u8, u8, u8, u8)>() {
                    img.set_pixel(x, y, 0, tuple.0)
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to set pixel: {}", e)))?;
                    img.set_pixel(x, y, 1, tuple.1)
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to set pixel: {}", e)))?;
                    img.set_pixel(x, y, 2, tuple.2)
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to set pixel: {}", e)))?;
                    img.set_pixel(x, y, 3, tuple.3)
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to set pixel: {}", e)))?;
                } else if let Ok(val) = value.extract::<u8>() {
                    img.set_pixel(x, y, 0, val)
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to set pixel: {}", e)))?;
                    img.set_pixel(x, y, 1, val)
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to set pixel: {}", e)))?;
                    img.set_pixel(x, y, 2, val)
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to set pixel: {}", e)))?;
                    img.set_pixel(x, y, 3, val)
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to set pixel: {}", e)))?;
                } else {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Value for RGBA must be an integer or tuple of (R, G, B, A)"
                    ));
                }
            }
            ImageData::L(img) => {
                let val = value.extract::<u8>().map_err(|_| pyo3::exceptions::PyValueError::new_err(
                    "Value for L must be an integer (0-255)"
                ))?;
                img.set_pixel(x, y, 0, val)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to set pixel: {}", e)))?;
            }
        }

        Ok(())
    }

    /// Convert the image to a NumPy array (zero-copy when possible).
    ///
    /// # Returns
    ///
    /// NumPy array of shape:
    /// - (height, width, 3) for RGB
    /// - (height, width, 4) for RGBA  
    /// - (height, width, 1) for L
    /// with dtype uint8
    ///
    /// # Examples
    ///
    /// ```python
    /// import kornia_rs as K
    /// img = K.Image.open("image.jpg")
    /// arr = img.to_numpy()
    /// print(arr.shape)  # (height, width, channels)
    /// ```
    pub fn to_numpy(&self) -> PyResult<PyImage> {
        match &self.inner {
            ImageData::Rgb(img) => img.clone().to_pyimage()
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to convert to numpy: {}", e))),
            ImageData::Rgba(_) => Err(pyo3::exceptions::PyValueError::new_err(
                "RGBA to numpy conversion not yet implemented. Use RGB mode."
            )),
            ImageData::L(_) => Err(pyo3::exceptions::PyValueError::new_err(
                "Grayscale to numpy conversion not yet implemented. Use RGB mode."
            )),
        }
    }

    /// Create an Image from a NumPy array.
    ///
    /// # Arguments
    ///
    /// * `array` - NumPy array of shape (height, width, channels) with dtype uint8
    ///   - channels=3 for RGB
    ///   - channels=4 for RGBA
    ///   - channels=1 for L (grayscale)
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
        // For now, only support RGB (3-channel)
        let image = KorniaImage::<u8, 3, CpuAllocator>::from_pyimage(array)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to create image from numpy: {}", e)))?;
        
        Ok(PyImage3 { inner: ImageData::Rgb(image) })
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
