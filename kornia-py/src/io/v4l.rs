//! Python bindings for native V4L2 webcam capture (`kornia_io::v4l`).
//!
//! Exposes a [`PyV4lCapture`] class (`kornia_rs.capture.V4lCapture`) that grabs
//! frames from a `/dev/video*` device and returns [`PyV4lFrame`] objects. A frame
//! mirrors the Rust `EncodedFrame`: it holds a **zero-copy view** into the kernel
//! mmap buffer (a JPEG for ``MJPG``, packed 4:2:2 for ``YUYV``) plus metadata, and
//! decodes to an RGB ``kornia_rs.image.Image`` only on demand.
//!
//! # Zero-copy
//!
//! `grab()` performs **no copy**: the frame borrows the kernel mmap buffer (kept
//! valid via the same `Arc<MmapInfo>` refcount used in Rust). `raw` exposes those
//! bytes through the Python buffer protocol (`memoryview(frame)` /
//! `np.frombuffer(frame, np.uint8)`) with no copy either. Because a live frame pins
//! its kernel buffer (it will not be re-queued until dropped), keep at most
//! `buffer_size` frames alive at once — otherwise the next `grab()` raises
//! `RuntimeError` (buffers exhausted). To retain a frame indefinitely, copy it
//! explicitly (`bytes(frame.raw)` or `np.array(frame.image)`).
//!
//! Decoding (YUYV/MJPG → RGB) is done here in the Python layer, which already
//! depends on `kornia-imgproc` and `kornia-io`, keeping `kornia-io` itself free of
//! any image-processing dependency.
//!
//! Linux-only; gated behind the `v4l` cargo feature.
use std::os::raw::c_int;
use std::str::FromStr;

use pyo3::exceptions::{PyBufferError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyMemoryView, PyTuple};

use kornia_image::{ColorSpace, Image, ImageSize};
use kornia_imgproc::color::{convert_yuyv_to_rgb_u8, YuvToRgbMode};
use kornia_io::v4l::{MmapBuffer, PixelFormat, V4LCameraConfig, V4lVideoCapture};

use crate::backing::{AlignedBytes, Dtype};
use crate::image::{PyImageApi, PyImageSize};

/// Map a `V4L2Error` to a Python exception.
fn v4l_err(e: kornia_io::v4l::V4L2Error) -> PyErr {
    PyRuntimeError::new_err(format!("v4l capture error: {e}"))
}

/// A single captured frame, mirroring the Rust `kornia_io::v4l::EncodedFrame`.
///
/// Holds a **zero-copy** view of the raw sensor bytes (a JPEG for ``MJPG``, packed
/// YUYV for ``YUYV``) and decodes to RGB lazily on ``image``. Supports the buffer
/// protocol, so `memoryview(frame)` / `np.frombuffer(frame, np.uint8)` view the raw
/// bytes without copying.
#[pyclass(name = "V4lFrame", module = "kornia_rs.capture")]
pub struct PyV4lFrame {
    /// Zero-copy view of the kernel mmap buffer (holds the `Arc<MmapInfo>` keepalive).
    buffer: MmapBuffer,
    format: PixelFormat,
    height: usize,
    width: usize,
    /// Capture timestamp in seconds (kernel monotonic clock).
    #[pyo3(get)]
    timestamp: f64,
    /// Monotonically increasing frame sequence number from the driver.
    #[pyo3(get)]
    sequence: u32,
    /// Lazily-decoded RGB image, computed on first ``image`` access.
    image_cache: PyOnceLock<Py<PyImageApi>>,
}

impl PyV4lFrame {
    /// Decode the raw bytes into an owned RGB [`PyImageApi`].
    ///
    /// Reads the mmap buffer zero-copy; the RGB output is a genuine decode (YUYV→RGB
    /// or JPEG decompress), not a copy of the frame.
    fn decode(&self) -> PyResult<PyImageApi> {
        let size = ImageSize {
            width: self.width,
            height: self.height,
        };
        let mut rgb = Image::<u8, 3>::from_size_val(size, 0u8)
            .map_err(|e| PyRuntimeError::new_err(format!("failed to allocate RGB buffer: {e}")))?;

        match self.format {
            PixelFormat::YUYV => {
                convert_yuyv_to_rgb_u8(self.buffer.as_slice(), &mut rgb, YuvToRgbMode::Bt601Full)
                    .map_err(|e| PyRuntimeError::new_err(format!("YUYV decode failed: {e}")))?;
            }
            PixelFormat::MJPG => {
                kornia_io::jpeg::decode_image_jpeg_rgb8(self.buffer.as_slice(), &mut rgb)
                    .map_err(|e| PyRuntimeError::new_err(format!("JPEG decode failed: {e}")))?;
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "cannot decode pixel format {other} to RGB (supported: YUYV, MJPG)"
                )));
            }
        }

        let bytes = AlignedBytes::from_slice(rgb.as_slice());
        Ok(PyImageApi::from_owned_bytes(
            bytes,
            Dtype::U8,
            [self.height, self.width, 3],
            ColorSpace::Rgb,
            "RGB".to_string(),
        ))
    }
}

#[pymethods]
impl PyV4lFrame {
    /// The decoded RGB image (``kornia_rs.image.Image``, HxWx3 uint8).
    ///
    /// Decoded lazily on first access and cached, so reading it repeatedly is cheap.
    /// Skip this entirely if you only want the raw/encoded bytes.
    #[getter]
    fn image(&self, py: Python<'_>) -> PyResult<Py<PyImageApi>> {
        let cached = self
            .image_cache
            .get_or_try_init(py, || Py::new(py, self.decode()?))?;
        Ok(cached.clone_ref(py))
    }

    /// A zero-copy ``memoryview`` of the raw sensor bytes.
    ///
    /// For ``MJPG`` this is a complete JPEG; for ``YUYV`` it is packed 4:2:2 data.
    /// No copy is made — the view borrows the kernel buffer (read-only) and keeps
    /// this frame alive for its lifetime.
    #[getter]
    fn raw<'py>(slf: Bound<'py, Self>) -> PyResult<Bound<'py, PyMemoryView>> {
        PyMemoryView::from(slf.as_any())
    }

    /// The sensor pixel format (e.g. ``"YUYV"`` or ``"MJPG"``).
    #[getter]
    fn pixel_format(&self) -> String {
        self.format.to_string()
    }

    /// ``True`` if the frame holds a compressed/encoded format (e.g. MJPG/JPEG),
    /// ``False`` for raw uncompressed formats (e.g. YUYV).
    #[getter]
    fn is_encoded(&self) -> bool {
        self.format.bytes_per_pixel().is_none()
    }

    /// The frame size as an ``ImageSize``.
    fn size(&self) -> PyResult<PyImageSize> {
        PyImageSize::new(self.width, self.height)
    }

    fn __repr__(&self) -> String {
        format!(
            "V4lFrame(seq={}, ts={:.3}s, format={}, encoded={}, {} bytes)",
            self.sequence,
            self.timestamp,
            self.format,
            self.is_encoded(),
            self.buffer.len(),
        )
    }

    /// PEP 3118 buffer protocol: expose the raw mmap bytes read-only, zero-copy.
    ///
    /// `PyBuffer_FillInfo` installs this frame as the buffer's exporter (INCREF),
    /// so the kernel buffer stays alive for the memoryview's lifetime.
    unsafe fn __getbuffer__(
        slf: PyRefMut<'_, Self>,
        view: *mut pyo3::ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(PyBufferError::new_err("null view"));
        }
        let bytes = slf.buffer.as_slice();
        let ret = unsafe {
            pyo3::ffi::PyBuffer_FillInfo(
                view,
                slf.as_ptr(),
                bytes.as_ptr() as *mut std::ffi::c_void,
                bytes.len() as pyo3::ffi::Py_ssize_t,
                1, // readonly
                flags,
            )
        };
        if ret != 0 {
            return Err(PyErr::fetch(slf.py()));
        }
        Ok(())
    }

    unsafe fn __releasebuffer__(&self, _view: *mut pyo3::ffi::Py_buffer) {
        // PyBuffer_FillInfo allocated nothing; the exporter refcount is released by
        // the interpreter's PyBuffer_Release.
    }
}

/// Native V4L2 webcam capture.
///
/// Grabs frames from a `/dev/video*` device. Each [`grab`](Self::grab) returns a
/// [`PyV4lFrame`] holding a zero-copy view of the raw sensor bytes + metadata;
/// decode to RGB on demand via ``frame.image``.
#[pyclass(name = "V4lCapture", module = "kornia_rs.capture")]
pub struct PyV4lCapture {
    inner: V4lVideoCapture,
    size: ImageSize,
}

#[pymethods]
impl PyV4lCapture {
    /// Open a V4L2 capture device.
    ///
    /// Args:
    ///     camera_id: index of ``/dev/video{camera_id}`` (ignored if ``device`` given).
    ///     width / height: requested frame size (the driver may clamp it).
    ///     fps: requested frame rate.
    ///     pixel_format: sensor format, one of ``"YUYV"``, ``"MJPG"``.
    ///     buffer_size: number of kernel capture buffers (also the max number of
    ///         live frames you may hold before ``grab`` raises buffers-exhausted).
    ///     device: explicit device path (overrides ``camera_id``).
    #[new]
    #[pyo3(signature = (
        camera_id = 0,
        width = 640,
        height = 480,
        fps = 30,
        pixel_format = "YUYV",
        buffer_size = 4,
        device = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        camera_id: u32,
        width: usize,
        height: usize,
        fps: u32,
        pixel_format: &str,
        buffer_size: u32,
        device: Option<String>,
    ) -> PyResult<Self> {
        let format = PixelFormat::from_str(pixel_format)
            .map_err(|e| PyValueError::new_err(format!("invalid pixel_format: {e}")))?;

        let device_path = device.unwrap_or_else(|| format!("/dev/video{camera_id}"));

        let config = V4LCameraConfig {
            device_path,
            size: ImageSize { width, height },
            fps,
            format,
            buffer_size,
        };

        let inner = V4lVideoCapture::new(config).map_err(v4l_err)?;
        let size = inner.size();

        Ok(Self { inner, size })
    }

    /// Grab the next frame as a [`PyV4lFrame`] (zero-copy; decode lazily).
    ///
    /// Returns ``None`` if no frame was available (e.g. a configured timeout
    /// elapsed). Blocks until a frame is ready; the GIL is released while waiting.
    fn grab(&mut self, py: Python<'_>) -> PyResult<Option<PyV4lFrame>> {
        // Release the GIL during the blocking capture. No copy: the MmapBuffer is
        // moved out of the EncodedFrame into the Python frame.
        let out = py
            .detach(|| -> Result<Option<(MmapBuffer, PixelFormat, f64, u32)>, String> {
                let Some(frame) = self.inner.grab_frame().map_err(|e| e.to_string())? else {
                    return Ok(None);
                };
                let ts = frame.timestamp.sec as f64 + frame.timestamp.usec as f64 / 1_000_000.0;
                Ok(Some((frame.buffer, frame.pixel_format, ts, frame.sequence)))
            })
            .map_err(PyRuntimeError::new_err)?;

        let Some((buffer, format, timestamp, sequence)) = out else {
            return Ok(None);
        };

        Ok(Some(PyV4lFrame {
            buffer,
            format,
            height: self.size.height,
            width: self.size.width,
            timestamp,
            sequence,
            image_cache: PyOnceLock::new(),
        }))
    }

    /// The frame size the driver actually negotiated (may differ from requested).
    fn size(&self) -> PyResult<PyImageSize> {
        PyImageSize::new(self.size.width, self.size.height)
    }

    /// The negotiated sensor pixel format as a string (e.g. ``"YUYV"``).
    fn pixel_format(&self) -> String {
        self.inner.pixel_format().to_string()
    }

    /// Set the per-frame dequeue timeout in milliseconds (``None`` blocks).
    #[pyo3(signature = (timeout_ms = None))]
    fn set_timeout(&mut self, timeout_ms: Option<u32>) {
        self.inner.set_timeout(timeout_ms);
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (*_args))]
    fn __exit__(&self, _args: &Bound<'_, PyTuple>) -> bool {
        // Capture is stopped when the object is dropped; nothing to do here.
        false
    }

    fn __repr__(&self) -> String {
        format!(
            "V4lCapture(size={}x{}, format={})",
            self.size.width,
            self.size.height,
            self.inner.pixel_format()
        )
    }
}
