/// module for camera controls
pub mod camera_control;

mod pixel_format;
mod stream;

// re-export the camera control and pixel format types
pub use pixel_format::PixelFormat;
pub use stream::MmapBuffer;

use crate::v4l::camera_control::{CameraControlTrait, ControlType};
use kornia_image::ImageSize;
use v4l::{
    buffer::Type, control::Value, video::capture::Parameters, video::Capture, Device, Timestamp,
};

use std::any::Any;
use std::sync::Arc;

use kornia_tensor::{
    allocator::ForeignAllocator,
    resource::{MemoryDomain, MemoryResource},
};

/// A proper [`MemoryResource`] for a V4L2 mmap'd frame buffer.
///
/// Wraps a [`MmapBuffer`] which internally holds an `Arc<MmapInfo>`. When this
/// resource is dropped the Arc's reference count decrements; when it reaches zero
/// `MmapInfo::Drop` calls `munmap` — exactly once.
///
/// Zero-copy: the pointer exposed through `as_ptr` is the kernel mmap address;
/// no data is copied into or out of kornia-managed heap memory.
pub struct V4lResource {
    /// The mmap'd buffer; Drop decrements the Arc<MmapInfo> reference count.
    buffer: MmapBuffer,
}

// SAFETY: V4L2 mmap memory is page-mapped by the kernel and thread-safe for
// concurrent reads. MmapBuffer is Send + Sync (verified by its own unsafe impls).
unsafe impl Send for V4lResource {}
unsafe impl Sync for V4lResource {}

impl MemoryResource for V4lResource {
    /// Returns the host pointer to the mmap'd frame data.
    ///
    /// `MmapBuffer` exposes a `*const u8` via `as_slice`; we cast to `*mut u8`
    /// as the `MemoryResource` contract requires. Callers must not write through
    /// this pointer — V4L2 capture buffers are logically read-only.
    fn as_ptr(&self) -> *mut u8 {
        // NonNull<u8> stored inside MmapBuffer — extract via as_slice then cast.
        self.buffer.as_slice().as_ptr() as *mut u8
    }

    /// Length of the mmap'd region in bytes (the "used bytes" of the frame).
    fn len_bytes(&self) -> usize {
        self.buffer.len()
    }

    /// V4L2 mmap frames live in host-accessible memory.
    fn domain(&self) -> MemoryDomain {
        MemoryDomain::Host
    }

    /// Downcast hook.
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Construct a borrowed [`Image`] backed by a V4L2 mmap'd frame buffer.
// image_from_v4l_buffer is exercised by tests and available for caller use; the
// compiler sees it as dead code when compiled without tests because grab_frame still
// returns a raw EncodedFrame.  Allow dead_code rather than making it pub.
#[allow(dead_code)]
///
/// # Arguments
///
/// * `size`   - image dimensions (`{ width, height }`).
/// * `buffer` - the zero-copy `MmapBuffer` returned by [`MmapStream::next_frame`];
///   ownership is transferred into a [`V4lResource`] keepalive that is
///   `Arc`-shared with the tensor's `ForeignResource`.
///
/// # Returns
///
/// An `Image<u8, 3, ForeignAllocator>` whose memory is the V4L2 mmap region.
/// The kernel buffer remains live for exactly the lifetime of the returned `Image`;
/// dropping the `Image` releases the mmap region exactly once (via `Arc<MmapInfo>`).
///
/// # Safety
///
/// The caller must ensure `buffer` is not aliased as mutable elsewhere while the
/// returned `Image` is alive. V4L2 capture buffers are single-owner (the stream
/// dequeues them and re-queues only after the caller is done), so this is satisfied
/// by normal capture usage.
pub(crate) fn image_from_v4l_buffer(
    size: kornia_image::ImageSize,
    buffer: MmapBuffer,
) -> Result<kornia_image::Image<u8, 3, ForeignAllocator>, kornia_image::ImageError> {
    use kornia_tensor::storage::TensorStorage;
    use kornia_tensor::Tensor;

    // Capture pointer and length BEFORE moving buffer into V4lResource.
    let data_ptr: *const u8 = buffer.as_slice().as_ptr();
    let data_len: usize = buffer.len();

    // Move the MmapBuffer into a V4lResource; its implicit Drop releases the mmap.
    let resource = V4lResource { buffer };
    let keepalive: Arc<dyn Any + Send + Sync> = Arc::new(resource);

    // Build a TensorStorage that borrows the mmap memory and holds the keepalive.
    // SAFETY:
    //   - data_ptr is non-null (kernel mmap always returns a valid page-aligned address).
    //   - data_len equals the "used bytes" of the frame (captured above).
    //   - The memory is host-accessible (MemoryDomain::Host).
    //   - The keepalive (Arc<V4lResource>) holds the MmapBuffer (and Arc<MmapInfo>) alive.
    let storage: TensorStorage<u8, ForeignAllocator> = unsafe {
        TensorStorage::from_borrowed(
            data_ptr,
            data_len,
            ForeignAllocator,
            MemoryDomain::Host,
            0,
            keepalive,
        )
    };

    // Row-major strides for shape [H, W, 3]: strides = [W*3, 3, 1].
    let shape = [size.height, size.width, 3_usize];
    let strides = [size.width * 3, 3, 1];
    let tensor = Tensor {
        storage,
        shape,
        strides,
    };

    // TryFrom<Tensor3<T, A>> for Image<T, C, A> validates that shape[2] == C (== 3).
    kornia_image::Image::try_from(tensor)
}

/// Error types for the v4l2 module.
#[derive(Debug, thiserror::Error)]
pub enum V4L2Error {
    /// Failed to create image
    #[error(transparent)]
    ImageError(#[from] kornia_image::ImageError),

    /// Failed to set parameters
    #[error(transparent)]
    IoError(#[from] std::io::Error),
}

/// Configuration for V4L video capture.
pub struct V4LCameraConfig {
    /// The camera device path
    pub device_path: String,
    /// The desired image size
    pub size: ImageSize,
    /// The desired frames per second
    pub fps: u32,
    /// The desired pixel format
    pub format: PixelFormat,
    /// The number of buffers to use
    pub buffer_size: u32,
}

impl Default for V4LCameraConfig {
    fn default() -> Self {
        Self {
            device_path: "/dev/video0".to_string(),
            size: ImageSize {
                width: 640,
                height: 480,
            },
            fps: 30,
            format: PixelFormat::default(),
            buffer_size: 4,
        }
    }
}

/// V4L video capture.
pub struct V4lVideoCapture {
    stream: stream::MmapStream,
    pixel_format: PixelFormat,
    device: Device,
    size: ImageSize,
}

/// Represents a captured frame from a V4L camera.
///
/// The frame can contain either compressed data (e.g., JPEG) or uncompressed data
/// (e.g., YUYV, UYVY). The buffer uses zero-copy semantics via `MmapBuffer`, which
/// keeps the mmap'd memory alive via `Arc<MmapInfo>`.
pub struct EncodedFrame {
    /// The buffer of the frame (zero-copy via mmap)
    pub buffer: MmapBuffer,
    /// The image size of the frame
    pub size: ImageSize,
    /// The pixel format of the frame
    pub pixel_format: PixelFormat,
    /// The timestamp of the frame
    pub timestamp: Timestamp,
    /// The sequence number of the frame
    pub sequence: u32,
}

impl V4lVideoCapture {
    /// Create a new V4L video capture.
    pub fn new(config: V4LCameraConfig) -> Result<Self, V4L2Error> {
        let device = Device::with_path(&config.device_path)?;

        // Set the format
        let mut format = device.format()?;
        format.width = config.size.width as u32;
        format.height = config.size.height as u32;
        format.fourcc = config.format.to_fourcc();

        device.set_format(&format)?;

        // Verify the format was actually set (camera might not support it)
        let actual_format = device.format()?;
        if actual_format.fourcc != format.fourcc {
            eprintln!(
                "Warning: Requested format {} not supported, using {}",
                config.format,
                PixelFormat::from_fourcc(actual_format.fourcc)
            );
        }

        // Set the frame rate
        let params = Parameters::with_fps(config.fps);
        device.set_params(&params)?;

        // Create the stream
        let mut stream =
            stream::MmapStream::with_buffers(&device, Type::VideoCapture, config.buffer_size)?;
        stream.next_frame()?;

        Ok(Self {
            stream,
            pixel_format: PixelFormat::from_fourcc(actual_format.fourcc),
            device,
            size: config.size,
        })
    }

    /// Get the current pixel format
    #[inline]
    pub fn pixel_format(&self) -> PixelFormat {
        self.pixel_format
    }

    /// Set a camera control
    pub fn set_control<T: CameraControlTrait>(&mut self, control: T) -> Result<(), V4L2Error> {
        self.device
            .set_control(v4l::Control {
                id: control.control_id(),
                value: match control.value() {
                    ControlType::Integer(value) => Value::Integer(value),
                    ControlType::Boolean(value) => Value::Boolean(value),
                },
            })
            .map_err(V4L2Error::IoError)
    }

    /// Grab a frame from the camera
    pub fn grab_frame(&mut self) -> Result<Option<EncodedFrame>, V4L2Error> {
        let Ok((buffer, metadata)) = self.stream.next_frame() else {
            return Ok(None);
        };

        let frame = EncodedFrame {
            buffer,
            size: self.size,
            pixel_format: self.pixel_format,
            timestamp: metadata.timestamp,
            sequence: metadata.sequence,
        };

        Ok(Some(frame))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v4l::stream::MmapInfo;

    /// Unit test: build a `V4lResource` over an anonymous mmap region, read through
    /// an `Image` backed by it, then drop — verifying that `munmap` fires exactly
    /// once and the drop completes without fault.
    ///
    /// Uses `libc::mmap(MAP_ANONYMOUS | MAP_PRIVATE)` so no real camera is required.
    ///
    /// The "exactly once" guarantee is structural: `V4lResource` holds a `MmapBuffer`
    /// which holds an `Arc<MmapInfo>`.  Moving the buffer into the resource transfers
    /// the sole remaining `Arc` reference; when `V4lResource` drops, the `Arc` hits
    /// zero and `MmapInfo::Drop` calls `munmap` — once.  A second `munmap` on the
    /// same address would return `EINVAL` (Linux) or fault, which `MmapInfo::Drop`
    /// already prints as an error; the drop-counter below verifies the count.
    #[test]
    fn v4l_resource_anonymous_mmap_drop_once() {
        // We verify "exactly once" via Arc<MmapInfo> reference counting: after image
        // drop, the Weak reference must not upgrade (Arc gone → munmap fired).
        let page_size = 4096_usize;
        let total_bytes = page_size; // one page for the test

        // mmap an anonymous, private page — no file descriptor required.
        let raw_ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                total_bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert_ne!(
            raw_ptr,
            libc::MAP_FAILED,
            "anonymous mmap failed: {}",
            std::io::Error::last_os_error()
        );

        // Write a known pattern so we can verify read-through.
        unsafe {
            let bytes = std::slice::from_raw_parts_mut(raw_ptr as *mut u8, total_bytes);
            for (i, b) in bytes.iter_mut().enumerate() {
                *b = (i % 251) as u8; // 251 is prime — avoids wrap-around repetition
            }
        }

        // Wrap in Arc<MmapInfo> exactly as MmapStream does.
        let mmap_info = Arc::new(MmapInfo {
            ptr: raw_ptr as *mut u8,
            length: total_bytes,
            offset: 0,
        });

        // Keep a weak-count handle to observe the Arc going to zero.
        let arc_weak = Arc::downgrade(&mmap_info);

        // Build a MmapBuffer covering exactly H*W*3 bytes (12×4×3 = 144 ≤ 4096).
        const W: usize = 12;
        const H: usize = 4;
        const FRAME_BYTES: usize = W * H * 3;
        assert!(FRAME_BYTES <= total_bytes, "frame fits in one page");

        let buffer =
            unsafe { MmapBuffer::new(raw_ptr as *const u8, FRAME_BYTES, mmap_info.clone()) };

        // Drop the Arc we kept for observation — now only buffer (and arc_weak) hold it.
        drop(mmap_info);
        // Arc strong count == 1 (buffer's _mmap_info); weak count == 1 (arc_weak).
        assert!(
            arc_weak.upgrade().is_some(),
            "MmapInfo still alive via buffer"
        );

        // Build an Image via image_from_v4l_buffer — this moves `buffer` into V4lResource
        // and wraps it in Arc<dyn Any>; the V4lResource is then the sole holder of buffer
        // which is the sole holder of Arc<MmapInfo>.
        let size = ImageSize {
            width: W,
            height: H,
        };
        let image = image_from_v4l_buffer(size, buffer)
            .expect("image_from_v4l_buffer must succeed for a valid mmap region");

        // Verify dimensions and pixel data read-through.
        assert_eq!(image.width(), W);
        assert_eq!(image.height(), H);
        assert_eq!(image.num_channels(), 3);

        let slice = image.as_slice();
        assert_eq!(slice.len(), FRAME_BYTES);
        // Spot-check the pattern written above.
        for (i, &byte) in slice.iter().enumerate() {
            assert_eq!(byte, (i % 251) as u8, "pixel mismatch at index {i}");
        }

        // Arc<MmapInfo> still alive (held by V4lResource inside the Image).
        assert!(
            arc_weak.upgrade().is_some(),
            "MmapInfo must still be alive while Image is alive"
        );

        // Drop the Image → Image::Drop → Tensor::Drop → TensorStorage::Drop →
        // ForeignResource::Drop → Arc<V4lResource>::Drop (count → 0) →
        // V4lResource::Drop → MmapBuffer::Drop → Arc<MmapInfo>::Drop (count → 0) →
        // MmapInfo::Drop → munmap().  This is the path under test.
        drop(image);

        // After drop, the Arc<MmapInfo> must be gone (count == 0, munmap fired).
        assert!(
            arc_weak.upgrade().is_none(),
            "MmapInfo must have been released (munmap'd) when Image was dropped"
        );
    }

    /// Verify that `V4lResource` implements `MemoryResource` correctly for a synthetic
    /// non-null pointer (no camera required).
    #[test]
    fn v4l_resource_memory_resource_accessors() {
        // Use a stack-allocated array as backing memory (not mmap — no munmap needed).
        // We give MmapInfo a null ptr so its Drop is a no-op, and verify the resource fields.
        let data: [u8; 9] = [1, 2, 3, 4, 5, 6, 7, 8, 9]; // 3×3×1 = 9 bytes

        let mmap_info = Arc::new(MmapInfo {
            ptr: std::ptr::null_mut(), // null → Drop is a no-op (safe)
            length: 9,
            offset: 0,
        });

        let buffer = unsafe { MmapBuffer::new(data.as_ptr(), 9, mmap_info) };
        let resource = V4lResource { buffer };

        // MemoryResource trait checks.
        assert!(!resource.as_ptr().is_null(), "as_ptr must be non-null");
        assert_eq!(resource.len_bytes(), 9);
        assert!(
            matches!(resource.domain(), MemoryDomain::Host),
            "V4L2 frames are host-accessible"
        );
        // as_any downcast works.
        assert!(
            resource.as_any().downcast_ref::<V4lResource>().is_some(),
            "as_any downcast must succeed"
        );
    }
}
