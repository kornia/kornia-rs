use numpy::{PyArray, PyArray3, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::image::PyImage;

fn mode_from_channels(channels: usize) -> String {
    match channels {
        1 => "L".to_string(),
        3 => "RGB".to_string(),
        4 => "RGBA".to_string(),
        c => format!("{}ch", c),
    }
}

/// A high-level image object backed by numpy arrays and Rust operations.
///
/// Always stores data as HWC (height, width, channels) numpy arrays.
/// RGB color order by default.
///
/// Thread-safe and serialization-friendly for use with Ray Data,
/// multiprocessing, and other parallel execution frameworks.
#[pyclass(name = "Image", module = "kornia_rs", weakref)]
pub struct PyImageApi {
    data: Py<PyArray3<u8>>,
    mode: String,
}

impl PyImageApi {
    /// Internal: wrap a Py<PyArray3<u8>> with auto-detected mode
    pub fn wrap(py: Python<'_>, data: Py<PyArray3<u8>>, mode: Option<String>) -> Self {
        let channels = data.bind(py).shape()[2];
        let mode = mode.unwrap_or_else(|| mode_from_channels(channels));
        Self { data, mode }
    }

    /// Internal: wrap a PyImage from other bindings
    pub fn from_pyimage(py: Python<'_>, image: PyImage, mode: Option<String>) -> Self {
        Self::wrap(py, image, mode)
    }
}

#[pymethods]
impl PyImageApi {
    /// Create an Image from a numpy array.
    #[new]
    #[pyo3(signature = (data, mode=None))]
    fn new(py: Python<'_>, data: &Bound<'_, PyAny>, mode: Option<String>) -> PyResult<Self> {
        let shape: Vec<usize> = data.getattr("shape")?.extract()?;
        let arr = if shape.len() == 2 {
            let np = py.import("numpy")?;
            np.call_method1("expand_dims", (data, 2i32))?
                .extract::<Py<PyArray3<u8>>>()?
        } else if shape.len() == 3 {
            data.extract::<Py<PyArray3<u8>>>()?
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Expected 2D or 3D array, got {}D",
                shape.len()
            )));
        };
        Ok(Self::wrap(py, arr, mode))
    }

    // --- Static constructors ---

    /// Create an Image from raw pixel data or numpy array. Zero-copy for numpy.
    #[staticmethod]
    #[pyo3(signature = (data, width=None, height=None, channels=3, mode=None))]
    fn frombytes(
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        width: Option<usize>,
        height: Option<usize>,
        channels: Option<usize>,
        mode: Option<String>,
    ) -> PyResult<Self> {
        let np = py.import("numpy")?;

        // If it's a numpy array, use directly (zero-copy)
        if let Ok(shape_attr) = data.getattr("shape") {
            if let Ok(shape) = shape_attr.extract::<Vec<usize>>() {
                if !shape.is_empty() {
                    // Convert to numpy array if not already (handles memoryview etc.)
                    let as_array = np.call_method1("asarray", (data,))?;
                    let arr = if shape.len() == 2 {
                        np.call_method1("expand_dims", (&as_array, 2i32))?
                            .extract::<Py<PyArray3<u8>>>()?
                    } else if shape.len() == 3 {
                        as_array.extract::<Py<PyArray3<u8>>>()?
                    } else {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Expected 2D or 3D array, got {}D",
                            shape.len()
                        )));
                    };
                    return Ok(Self::wrap(py, arr, mode));
                }
            }
        }

        // Raw bytes/bytearray/memoryview — need explicit dimensions
        let w = width.ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "width and height are required for raw bytes input",
            )
        })?;
        let h = height.ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "width and height are required for raw bytes input",
            )
        })?;
        let c = channels.unwrap_or(3);

        let arr = np
            .call_method1("frombuffer", (data, np.getattr("uint8")?))?
            .call_method1("reshape", ((h, w, c),))?
            .extract::<Py<PyArray3<u8>>>()?;

        Ok(Self::wrap(py, arr, mode))
    }

    /// Load an image from file (JPEG, PNG, TIFF). Returns RGB.
    #[staticmethod]
    fn load(py: Python<'_>, path: &str) -> PyResult<Self> {
        let path_str = pyo3::types::PyString::new(py, path);
        let arr_any = crate::io::functional::read_image(path_str.into_any())?;
        let arr: Py<PyArray3<u8>> = arr_any.extract(py)?;
        Ok(Self::wrap(py, arr, None))
    }

    /// Decode encoded image bytes (JPEG, PNG) into an Image.
    #[staticmethod]
    #[pyo3(signature = (data, mode="RGB"))]
    fn decode(py: Python<'_>, data: &[u8], mode: &str) -> PyResult<Self> {
        let native_mode = match mode {
            "RGB" => "rgb",
            "RGBA" => "rgba",
            "L" => "mono",
            _ => "rgb",
        };

        let arr = if data.len() >= 2 && data[0] == 0xff && data[1] == 0xd8 {
            // JPEG
            match crate::io::jpegturbo::decode_image_jpegturbo(data, native_mode) {
                Ok(a) => a,
                Err(_) => crate::io::jpeg::decode_image_jpeg(data)?,
            }
        } else if data.len() >= 4 && &data[0..4] == b"\x89PNG" {
            // PNG: parse dimensions from IHDR
            if data.len() < 24 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Data too short to be a valid PNG",
                ));
            }
            let width = u32::from_be_bytes([data[16], data[17], data[18], data[19]]) as usize;
            let height = u32::from_be_bytes([data[20], data[21], data[22], data[23]]) as usize;
            crate::io::png::decode_image_png_u8(data, (height, width), native_mode)?
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unsupported image format: not JPEG or PNG",
            ));
        };

        Ok(Self::wrap(py, arr, Some(mode.to_string())))
    }

    // --- Properties ---

    #[getter]
    pub fn width(&self, py: Python<'_>) -> usize {
        self.data.bind(py).shape()[1]
    }

    #[getter]
    pub fn height(&self, py: Python<'_>) -> usize {
        self.data.bind(py).shape()[0]
    }

    #[getter]
    fn channels(&self, py: Python<'_>) -> usize {
        self.data.bind(py).shape()[2]
    }

    #[getter]
    pub fn mode(&self) -> &str {
        &self.mode
    }

    #[getter]
    fn size(&self, py: Python<'_>) -> (usize, usize) {
        let s = self.data.bind(py).shape();
        (s[1], s[0])
    }

    #[getter]
    fn shape(&self, py: Python<'_>) -> (usize, usize, usize) {
        let s = self.data.bind(py).shape();
        (s[0], s[1], s[2])
    }

    #[getter]
    fn dtype(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(py.import("numpy")?.getattr("uint8")?.unbind())
    }

    #[getter]
    pub fn data(&self, py: Python<'_>) -> Py<PyArray3<u8>> {
        self.data.clone_ref(py)
    }

    #[getter]
    fn nbytes(&self, py: Python<'_>) -> usize {
        let s = self.data.bind(py).shape();
        s[0] * s[1] * s[2]
    }

    // --- IO ---

    /// Save image to file. Format detected from extension.
    #[pyo3(signature = (path, quality=95))]
    fn save(&self, py: Python<'_>, path: &str, quality: u8) -> PyResult<()> {
        let c = self.data.bind(py).shape()[2];
        if c != 3 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "save requires 3-channel RGB image, got {} channels",
                c
            )));
        }

        let ext = path.rsplit('.').next().unwrap_or("").to_lowercase();
        match ext.as_str() {
            "jpg" | "jpeg" => crate::io::jpeg::write_image_jpeg(
                path,
                self.data.clone_ref(py),
                "rgb",
                quality,
            ),
            "png" => {
                crate::io::png::write_image_png_u8(path, self.data.clone_ref(py), "rgb")
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unsupported format: .{}. Supported: .jpg, .jpeg, .png",
                ext
            ))),
        }
    }

    /// Return a copy of the underlying numpy array.
    fn to_numpy(&self, py: Python<'_>) -> PyResult<Py<PyArray3<u8>>> {
        let copy = self.data.bind(py).call_method0("copy")?;
        Ok(copy.extract()?)
    }

    /// Return a deep copy of this image.
    pub fn copy(&self, py: Python<'_>) -> PyResult<Self> {
        let copy: Py<PyArray3<u8>> = self.data.bind(py).call_method0("copy")?.extract()?;
        Ok(Self {
            data: copy,
            mode: self.mode.clone(),
        })
    }

    // --- Chainable transforms ---

    /// Resize image to (width, height).
    #[pyo3(signature = (width, height, interpolation="bilinear"))]
    fn resize(
        &self,
        py: Python<'_>,
        width: usize,
        height: usize,
        interpolation: &str,
    ) -> PyResult<Self> {
        let c = self.data.bind(py).shape()[2];
        if c == 3 {
            let result =
                crate::resize::resize(self.data.clone_ref(py), (height, width), interpolation)?;
            Ok(Self::wrap(py, result, Some(self.mode.clone())))
        } else {
            // Numpy fallback: nearest neighbor
            let np = py.import("numpy")?;
            let arr = self.data.bind(py);
            let s = arr.shape();
            let (h, w) = (s[0], s[1]);

            let row_idx = np
                .call_method1("arange", (height,))?
                .call_method1("__mul__", (h as f64 / height as f64,))?
                .call_method1("astype", ("int",))?;
            let col_idx = np
                .call_method1("arange", (width,))?
                .call_method1("__mul__", (w as f64 / width as f64,))?
                .call_method1("astype", ("int",))?;

            let ix = np.call_method1("ix_", (&row_idx, &col_idx))?;
            let result = arr
                .get_item(ix)?
                .extract::<Py<PyArray3<u8>>>()?;
            Ok(Self::wrap(py, result, Some(self.mode.clone())))
        }
    }

    /// Flip image horizontally.
    pub fn flip_horizontal(&self, py: Python<'_>) -> PyResult<Self> {
        let c = self.data.bind(py).shape()[2];
        if c == 3 {
            let result = crate::flip::horizontal_flip(self.data.clone_ref(py))?;
            Ok(Self::wrap(py, result, Some(self.mode.clone())))
        } else {
            let np = py.import("numpy")?;
            let flipped = np.call_method1("flip", (self.data.bind(py), 1i32))?;
            let result = np
                .call_method1("ascontiguousarray", (&flipped,))?
                .extract::<Py<PyArray3<u8>>>()?;
            Ok(Self::wrap(py, result, Some(self.mode.clone())))
        }
    }

    /// Flip image vertically.
    pub fn flip_vertical(&self, py: Python<'_>) -> PyResult<Self> {
        let c = self.data.bind(py).shape()[2];
        if c == 3 {
            let result = crate::flip::vertical_flip(self.data.clone_ref(py))?;
            Ok(Self::wrap(py, result, Some(self.mode.clone())))
        } else {
            let np = py.import("numpy")?;
            let flipped = np.call_method1("flip", (self.data.bind(py), 0i32))?;
            let result = np
                .call_method1("ascontiguousarray", (&flipped,))?
                .extract::<Py<PyArray3<u8>>>()?;
            Ok(Self::wrap(py, result, Some(self.mode.clone())))
        }
    }

    /// Crop image at (x, y) with given width and height.
    pub fn crop(
        &self,
        py: Python<'_>,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
    ) -> PyResult<Self> {
        let c = self.data.bind(py).shape()[2];
        if c == 3 {
            let result = crate::crop::crop(self.data.clone_ref(py), x, y, width, height)?;
            Ok(Self::wrap(py, result, Some(self.mode.clone())))
        } else {
            let np = py.import("numpy")?;
            let arr = self.data.bind(py);
            let sliced = arr.call_method1(
                "__getitem__",
                (pyo3::types::PyTuple::new(
                    py,
                    &[
                        py.import("builtins")?
                            .call_method1("slice", (y, y + height))?,
                        py.import("builtins")?
                            .call_method1("slice", (x, x + width))?,
                    ],
                )?,),
            )?;
            let result = np
                .call_method1("ascontiguousarray", (&sliced,))?
                .extract::<Py<PyArray3<u8>>>()?;
            Ok(Self::wrap(py, result, Some(self.mode.clone())))
        }
    }

    /// Apply Gaussian blur.
    #[pyo3(signature = (kernel_size=3, sigma=1.0))]
    fn gaussian_blur(&self, py: Python<'_>, kernel_size: usize, sigma: f32) -> PyResult<Self> {
        let c = self.data.bind(py).shape()[2];
        if c == 3 {
            let result = crate::blur::gaussian_blur(
                self.data.clone_ref(py),
                (kernel_size, kernel_size),
                (sigma, sigma),
            )?;
            Ok(Self::wrap(py, result, Some(self.mode.clone())))
        } else {
            self.copy(py)
        }
    }

    /// Apply box blur.
    #[pyo3(signature = (kernel_size=3))]
    fn box_blur(&self, py: Python<'_>, kernel_size: usize) -> PyResult<Self> {
        let c = self.data.bind(py).shape()[2];
        if c == 3 {
            let result =
                crate::blur::box_blur(self.data.clone_ref(py), (kernel_size, kernel_size))?;
            Ok(Self::wrap(py, result, Some(self.mode.clone())))
        } else {
            self.copy(py)
        }
    }

    /// Adjust brightness. Factor is additive in [0,1] range.
    fn adjust_brightness(&self, py: Python<'_>, factor: f32) -> PyResult<Self> {
        let c = self.data.bind(py).shape()[2];
        if c == 3 {
            let result =
                crate::brightness::adjust_brightness_py(self.data.clone_ref(py), factor)?;
            Ok(Self::wrap(py, result, Some(self.mode.clone())))
        } else {
            let np = py.import("numpy")?;
            let arr = self.data.bind(py);
            let result = np
                .call_method1(
                    "clip",
                    (
                        np.call_method1(
                            "add",
                            (
                                arr.call_method1("astype", ("float32",))?,
                                factor * 255.0,
                            ),
                        )?,
                        0i32,
                        255i32,
                    ),
                )?
                .call_method1("astype", ("uint8",))?
                .extract::<Py<PyArray3<u8>>>()?;
            Ok(Self::wrap(py, result, Some(self.mode.clone())))
        }
    }

    /// Adjust contrast. factor=1.0 is identity, >1 increases contrast.
    fn adjust_contrast(&self, py: Python<'_>, factor: f64) -> PyResult<Self> {
        let np = py.import("numpy")?;
        let arr = self.data.bind(py);
        let mean: f64 = arr.call_method0("mean")?.extract()?;

        let float_data = arr.call_method1("astype", ("float32",))?;
        let centered = np.call_method1("subtract", (&float_data, mean))?;
        let scaled = np.call_method1("multiply", (&centered, factor))?;
        let shifted = np.call_method1("add", (&scaled, mean))?;
        let clipped = np.call_method1("clip", (&shifted, 0.0f64, 255.0f64))?;
        let result = clipped
            .call_method1("astype", ("uint8",))?
            .extract::<Py<PyArray3<u8>>>()?;
        Ok(Self::wrap(py, result, Some(self.mode.clone())))
    }

    /// Adjust saturation. factor=1.0 is identity, 0.0 is grayscale.
    fn adjust_saturation(&self, py: Python<'_>, factor: f64) -> PyResult<Self> {
        let c = self.data.bind(py).shape()[2];
        if c != 3 {
            return self.copy(py);
        }
        let np = py.import("numpy")?;
        let arr = self.data.bind(py);

        let float_data = arr.call_method1("astype", ("float32",))?;
        let weights = PyArray::from_vec(py, vec![0.299f64, 0.587, 0.114]);
        let gray = np.call_method1("dot", (&float_data, weights))?;
        let gray = np.call_method1("expand_dims", (&gray, 2i32))?;
        let diff = np.call_method1("subtract", (&float_data, &gray))?;
        let scaled = np.call_method1("multiply", (&diff, factor))?;
        let result_f = np.call_method1("add", (&gray, &scaled))?;
        let clipped = np.call_method1("clip", (&result_f, 0.0f64, 255.0f64))?;
        let result = clipped
            .call_method1("astype", ("uint8",))?
            .extract::<Py<PyArray3<u8>>>()?;
        Ok(Self::wrap(py, result, Some(self.mode.clone())))
    }

    /// Adjust hue. factor is in [-0.5, 0.5], fraction of hue wheel.
    pub fn adjust_hue(&self, py: Python<'_>, factor: f64) -> PyResult<Self> {
        let c = self.data.bind(py).shape()[2];
        if c != 3 || factor == 0.0 {
            return self.copy(py);
        }

        let locals = PyDict::new(py);
        locals.set_item("np", py.import("numpy")?)?;
        locals.set_item("data", self.data.bind(py))?;
        locals.set_item("factor", factor)?;

        py.run(
            cr#"
img_f = data.astype('float32') / 255.0
maxc = img_f.max(axis=2)
minc = img_f.min(axis=2)
diff = maxc - minc
h = np.zeros_like(maxc)
s = np.zeros_like(maxc)
v = maxc
mask = diff > 0
r_mask = mask & (img_f[..., 0] == maxc)
g_mask = mask & (img_f[..., 1] == maxc) & ~r_mask
b_mask = mask & ~r_mask & ~g_mask
h[r_mask] = ((img_f[..., 1] - img_f[..., 2])[r_mask] / diff[r_mask]) % 6
h[g_mask] = ((img_f[..., 2] - img_f[..., 0])[g_mask] / diff[g_mask]) + 2
h[b_mask] = ((img_f[..., 0] - img_f[..., 1])[b_mask] / diff[b_mask]) + 4
h = h / 6.0
s[mask] = diff[mask] / maxc[mask]
h = (h + factor) % 1.0
h6 = h * 6.0
i = np.floor(h6).astype(int) % 6
f = h6 - np.floor(h6)
p = v * (1 - s)
q = v * (1 - s * f)
t = v * (1 - s * (1 - f))
_result = np.zeros_like(img_f)
for idx, (r, g, b) in enumerate([(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]):
    m = (i == idx)
    _result[..., 0][m] = r[m]
    _result[..., 1][m] = g[m]
    _result[..., 2][m] = b[m]
_result = np.clip(_result * 255, 0, 255).astype(np.uint8)
"#,
            None,
            Some(&locals),
        )?;

        let result = locals
            .get_item("_result")?
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("adjust_hue failed")
            })?
            .extract::<Py<PyArray3<u8>>>()?;
        Ok(Self::wrap(py, result, Some(self.mode.clone())))
    }

    /// Normalize image to float32 using mean and std per channel.
    fn normalize(
        &self,
        py: Python<'_>,
        mean: (f32, f32, f32),
        std: (f32, f32, f32),
    ) -> PyResult<Py<PyArray3<f32>>> {
        let c = self.data.bind(py).shape()[2];
        if c == 3 {
            crate::normalize::normalize_mean_std(
                self.data.clone_ref(py),
                [mean.0, mean.1, mean.2],
                [std.0, std.1, std.2],
            )
        } else {
            let np = py.import("numpy")?;
            let arr = self.data.bind(py);
            let img_f = np.call_method1(
                "divide",
                (arr.call_method1("astype", ("float32",))?, 255.0f64),
            )?;
            let mean_arr = PyArray::from_vec(py, vec![mean.0, mean.1, mean.2])
                .call_method1("reshape", ((1i32, 1i32, 3i32),))?;
            let std_arr = PyArray::from_vec(py, vec![std.0, std.1, std.2])
                .call_method1("reshape", ((1i32, 1i32, 3i32),))?;
            let normed = np.call_method1(
                "divide",
                (np.call_method1("subtract", (&img_f, mean_arr))?, std_arr),
            )?;
            Ok(normed
                .call_method1("astype", ("float32",))?
                .extract::<Py<PyArray3<f32>>>()?)
        }
    }

    /// Convert RGB image to grayscale (1 channel).
    fn to_grayscale(&self, py: Python<'_>) -> PyResult<Self> {
        let c = self.data.bind(py).shape()[2];
        if c == 1 {
            return self.copy(py);
        }
        if c == 3 {
            let result = crate::color::gray_from_rgb(self.data.clone_ref(py))?;
            Ok(Self::wrap(py, result, Some("L".to_string())))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Cannot convert {}-channel image to grayscale",
                c
            )))
        }
    }

    /// Convert grayscale to RGB (3 channels).
    fn to_rgb(&self, py: Python<'_>) -> PyResult<Self> {
        let c = self.data.bind(py).shape()[2];
        if c == 3 {
            return self.copy(py);
        }
        if c == 1 {
            let result = crate::color::rgb_from_gray(self.data.clone_ref(py))?;
            Ok(Self::wrap(py, result, Some("RGB".to_string())))
        } else if c == 4 {
            let result = crate::color::rgb_from_rgba(self.data.clone_ref(py), None)?;
            Ok(Self::wrap(py, result, Some("RGB".to_string())))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Cannot convert {}-channel image to RGB",
                c
            )))
        }
    }

    /// Rotate image by angle degrees (counter-clockwise).
    pub fn rotate(&self, py: Python<'_>, angle: f64) -> PyResult<Self> {
        let c = self.data.bind(py).shape()[2];
        if c == 3 {
            let s = self.data.bind(py).shape();
            let (h, w) = (s[0] as f64, s[1] as f64);
            let (cx, cy) = (w / 2.0, h / 2.0);
            let rad = angle.to_radians();
            let cos_a = rad.cos() as f32;
            let sin_a = rad.sin() as f32;
            let tx = (cx - cos_a as f64 * cx + sin_a as f64 * cy) as f32;
            let ty = (cy - sin_a as f64 * cx - cos_a as f64 * cy) as f32;
            let m = [cos_a, -sin_a, tx, sin_a, cos_a, ty];
            let result = crate::warp::warp_affine(
                self.data.clone_ref(py),
                m,
                (s[0], s[1]),
                "bilinear",
            )?;
            Ok(Self::wrap(py, result, Some(self.mode.clone())))
        } else {
            // Numpy fallback: 90-degree multiples
            let np = py.import("numpy")?;
            let k = ((angle / 90.0).round() as i32).rem_euclid(4);
            if k == 0 {
                return self.copy(py);
            }
            let rotated = np.call_method1("rot90", (self.data.bind(py), k))?;
            let result = np
                .call_method1("ascontiguousarray", (&rotated,))?
                .extract::<Py<PyArray3<u8>>>()?;
            Ok(Self::wrap(py, result, Some(self.mode.clone())))
        }
    }

    // --- Serialization for multiprocess (Ray Data, etc.) ---

    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let cls = py
            .import("kornia_rs.image")?
            .getattr("Image")?
            .unbind();
        let args = pyo3::types::PyTuple::new(
            py,
            &[
                self.data.bind(py).as_any(),
                pyo3::types::PyString::new(py, &self.mode).as_any(),
            ],
        )?
        .unbind();
        Ok((cls, args.into_any()))
    }

    fn __getstate__(&self, py: Python<'_>) -> (Py<PyArray3<u8>>, String) {
        (self.data.clone_ref(py), self.mode.clone())
    }

    fn __setstate__(&mut self, _py: Python<'_>, state: (Py<PyArray3<u8>>, String)) {
        self.data = state.0;
        self.mode = state.1;
    }

    // --- Dunder methods ---

    fn __repr__(&self, py: Python<'_>) -> String {
        let s = self.data.bind(py).shape();
        format!(
            "Image(mode={}, size={}x{}, dtype=uint8)",
            self.mode, s[1], s[0]
        )
    }

    fn __eq__(&self, py: Python<'_>, other: &Self) -> PyResult<bool> {
        if self.mode != other.mode {
            return Ok(false);
        }
        let np = py.import("numpy")?;
        let equal: bool = np
            .call_method1("array_equal", (self.data.bind(py), other.data.bind(py)))?
            .extract()?;
        Ok(equal)
    }

    #[pyo3(signature = (dtype=None, copy=None))]
    fn __array__(
        &self,
        py: Python<'_>,
        dtype: Option<&str>,
        copy: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let arr = self.data.bind(py);
        if let Some(dt) = dtype {
            Ok(arr
                .call_method1("astype", (dt,))?
                .call_method0("copy")?
                .unbind())
        } else if copy.unwrap_or(false) {
            Ok(arr.call_method0("copy")?.unbind())
        } else {
            Ok(self.data.clone_ref(py).into_any())
        }
    }

    fn __len__(&self, py: Python<'_>) -> usize {
        self.data.bind(py).shape()[0]
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __exit__(
        &self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_val: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> bool {
        false
    }
}
