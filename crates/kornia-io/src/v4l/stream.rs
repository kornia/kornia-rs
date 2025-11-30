use std::{io, mem, ptr::NonNull, sync::Arc};
use v4l::{
    buffer::{Metadata, Type},
    device::{Device, Handle},
    io::traits::Stream as StreamTrait,
    memory::Memory,
    v4l2,
    v4l_sys::*,
};

/// A memory-mapped buffer that provides safe access to mmap'd memory.
///
/// This buffer wraps a pointer to mmap'd memory and ensures proper cleanup
/// via the `MmapInfo` struct. The buffer uses `Arc<MmapInfo>` to keep the mmap'd
/// memory alive, allowing zero-copy access to the data.
///
/// # Zero-Copy Semantics
///
/// The buffer holds an `Arc<MmapInfo>` which keeps the mmap'd memory alive.
/// Multiple `MmapBuffer` instances can share the same mmap'd memory via reference
/// counting, enabling zero-copy buffer sharing.
///
/// # Safety
///
/// The mmap'd memory remains valid as long as at least one `Arc<MmapInfo>` reference
/// exists. When the last reference is dropped, the memory is unmapped.
#[derive(Clone)]
pub struct MmapBuffer {
    /// Non-null pointer to the mmap'd memory
    ptr: NonNull<u8>,
    /// Length of the buffer in bytes (may be less than full mmap size for used bytes)
    length: usize,
    /// Reference to the mmap info for proper cleanup tracking
    /// This ensures the memory is not unmapped while the buffer is still in use
    _mmap_info: Arc<MmapInfo>,
}

// Safety: MmapBuffer is Send because:
// - The mmap'd memory is read-only from the buffer's perspective
// - Arc<MmapInfo> is Send and Sync, providing thread-safe reference counting
// - Multiple threads can safely read from the same mmap'd memory concurrently
// - The pointer and length are Copy types that can be safely moved between threads
unsafe impl Send for MmapBuffer {}

// Safety: MmapBuffer is Sync because:
// - The mmap'd memory is read-only from the buffer's perspective
// - Arc<MmapInfo> is Send and Sync, providing thread-safe reference counting
// - Multiple threads can safely read from the same mmap'd memory concurrently
// - The pointer and length are Copy types that can be safely shared between threads
unsafe impl Sync for MmapBuffer {}

impl MmapBuffer {
    /// Create a new MmapBuffer from a raw pointer and length.
    ///
    /// # Safety
    ///
    /// - `ptr` must be a valid, non-null pointer to `length` bytes of memory
    /// - The memory must remain valid for the lifetime of the buffer
    /// - `mmap_info` must be the same `MmapInfo` that was used to create the mmap
    pub(crate) unsafe fn new(ptr: *const u8, length: usize, mmap_info: Arc<MmapInfo>) -> Self {
        Self {
            ptr: NonNull::new_unchecked(ptr as *mut u8),
            length,
            _mmap_info: mmap_info,
        }
    }

    /// Create a new MmapBuffer that points to a subset of the mmap'd memory.
    ///
    /// This is used to create a buffer that only covers the used bytes, while
    /// still keeping the full mmap alive via the shared `Arc<MmapInfo>`.
    ///
    /// The method safely bounds-checks `used_bytes` against the current buffer length,
    /// ensuring memory safety regardless of the input value.
    pub(crate) fn with_length(&self, used_bytes: usize) -> Self {
        let len = used_bytes.min(self.length);
        Self {
            ptr: self.ptr,
            length: len,
            _mmap_info: self._mmap_info.clone(),
        }
    }

    /// Get the length of the buffer in bytes
    pub fn len(&self) -> usize {
        self.length
    }

    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Get the buffer data as a slice
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.length) }
    }

    /// Consume the buffer and return the data as a vector
    ///
    /// NOTE: This method copies the data. For zero-copy access, use `as_slice()`.
    pub fn into_vec(self) -> Vec<u8> {
        self.as_slice().to_vec()
    }
}

impl std::ops::Deref for MmapBuffer {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

/// Information about memory-mapped buffer
pub(crate) struct MmapInfo {
    ptr: *mut u8, // Stored as *mut for munmap compatibility
    length: usize,
    #[allow(dead_code)]
    offset: u32,
}

impl Drop for MmapInfo {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                let result = libc::munmap(self.ptr as *mut libc::c_void, self.length);
                if result == -1 {
                    eprintln!(
                        "Error: munmap failed with errno {}",
                        *libc::__errno_location()
                    );
                }
            }
        }
    }
}

/// Zero-copy mmap stream that returns owned references
pub struct MmapStream {
    handle: Arc<Handle>,
    buffers: Vec<MmapBuffer>,
    buf_type: Type,
    buf_meta: Vec<Metadata>,
    active: bool,
    timeout: Option<i32>,
    current_index: usize,
}

impl MmapStream {
    /// Create a new zero-copy mmap stream with specified buffer count
    pub fn with_buffers(device: &Device, buf_type: Type, buf_count: u32) -> io::Result<Self> {
        let handle = device.handle();

        // Request buffers from the kernel
        let mut v4l2_reqbufs = v4l2_requestbuffers {
            count: buf_count,
            type_: buf_type as u32,
            memory: Memory::Mmap as u32,
            ..unsafe { mem::zeroed() }
        };

        unsafe {
            v4l2::ioctl(
                handle.fd(),
                v4l2::vidioc::VIDIOC_REQBUFS,
                &mut v4l2_reqbufs as *mut _ as *mut std::os::raw::c_void,
            )?;
        }

        let actual_count = v4l2_reqbufs.count as usize;
        let mut buffers = Vec::with_capacity(actual_count);
        let mut buf_meta = Vec::with_capacity(actual_count);

        // Memory map each buffer
        for i in 0..actual_count {
            let mut v4l2_buf = v4l2_buffer {
                index: i as u32,
                type_: buf_type as u32,
                memory: Memory::Mmap as u32,
                ..unsafe { mem::zeroed() }
            };

            unsafe {
                v4l2::ioctl(
                    handle.fd(),
                    v4l2::vidioc::VIDIOC_QUERYBUF,
                    &mut v4l2_buf as *mut _ as *mut std::os::raw::c_void,
                )?;
            }

            let length = v4l2_buf.length as usize;
            let offset = unsafe { v4l2_buf.m.offset };

            // Memory map the buffer
            let ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    length,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    handle.fd(),
                    offset as libc::off_t,
                )
            };

            if ptr == libc::MAP_FAILED {
                return Err(io::Error::last_os_error());
            }

            let mmap_info = Arc::new(MmapInfo {
                ptr: ptr as *mut u8,
                length,
                offset,
            });

            // Create a safe buffer wrapper for the mmap'd memory
            let buffer = unsafe { MmapBuffer::new(ptr as *const u8, length, mmap_info) };

            buffers.push(buffer);
            buf_meta.push(Metadata::default());
        }

        Ok(Self {
            handle,
            buffers,
            buf_type,
            buf_meta,
            active: false,
            timeout: None,
            current_index: 0,
        })
    }

    fn queue_buffer(&mut self, index: usize) -> io::Result<()> {
        let mut v4l2_buf = v4l2_buffer {
            index: index as u32,
            type_: self.buf_type as u32,
            memory: Memory::Mmap as u32,
            ..unsafe { mem::zeroed() }
        };

        unsafe {
            v4l2::ioctl(
                self.handle.fd(),
                v4l2::vidioc::VIDIOC_QBUF,
                &mut v4l2_buf as *mut _ as *mut std::os::raw::c_void,
            )?;
        }

        Ok(())
    }

    fn dequeue_buffer(&mut self) -> io::Result<usize> {
        let mut v4l2_buf = v4l2_buffer {
            type_: self.buf_type as u32,
            memory: Memory::Mmap as u32,
            ..unsafe { mem::zeroed() }
        };

        // Wait for buffer to be ready
        if self.handle.poll(libc::POLLIN, self.timeout.unwrap_or(-1))? == 0 {
            return Err(io::Error::new(io::ErrorKind::TimedOut, "Buffer timeout"));
        }

        unsafe {
            v4l2::ioctl(
                self.handle.fd(),
                v4l2::vidioc::VIDIOC_DQBUF,
                &mut v4l2_buf as *mut _ as *mut std::os::raw::c_void,
            )?;
        }

        let index = v4l2_buf.index as usize;

        // Update metadata
        self.buf_meta[index] = Metadata {
            bytesused: v4l2_buf.bytesused,
            flags: v4l2_buf.flags.into(),
            field: v4l2_buf.field,
            timestamp: v4l2_buf.timestamp.into(),
            sequence: v4l2_buf.sequence,
        };

        Ok(index)
    }

    /// Get the next frame and return the buffer as a zero-copy `MmapBuffer`.
    ///
    /// The buffer uses `Arc<MmapInfo>` to keep the mmap'd memory alive, enabling
    /// zero-copy access. Only the actual used bytes (from metadata.bytesused) are
    /// exposed if available, while the full mmap remains alive.
    pub fn next_frame(&mut self) -> io::Result<(MmapBuffer, Metadata)> {
        if !self.active {
            // Queue all buffers and start streaming
            for i in 0..self.buffers.len() {
                self.queue_buffer(i)?;
            }
            self.start()?;
        } else {
            // Re-queue the current buffer
            self.queue_buffer(self.current_index)?;
        }

        // Dequeue the next available buffer
        self.current_index = self.dequeue_buffer()?;

        let buffer = &self.buffers[self.current_index];
        let metadata = self.buf_meta[self.current_index];

        // Create a zero-copy buffer that only covers the used bytes
        // The Arc<MmapInfo> keeps the full mmap alive
        let used_bytes = if metadata.bytesused > 0 {
            metadata.bytesused as usize
        } else {
            buffer.len()
        };
        let frame_buffer = buffer.with_length(used_bytes);

        Ok((frame_buffer, metadata))
    }
}

impl StreamTrait for MmapStream {
    type Item = MmapBuffer;

    fn start(&mut self) -> io::Result<()> {
        if self.active {
            return Ok(());
        }

        let mut buf_type = self.buf_type as u32;
        unsafe {
            v4l2::ioctl(
                self.handle.fd(),
                v4l2::vidioc::VIDIOC_STREAMON,
                &mut buf_type as *mut _ as *mut std::os::raw::c_void,
            )?;
        }

        self.active = true;
        Ok(())
    }

    fn stop(&mut self) -> io::Result<()> {
        if !self.active {
            return Ok(());
        }

        let mut buf_type = self.buf_type as u32;
        unsafe {
            v4l2::ioctl(
                self.handle.fd(),
                v4l2::vidioc::VIDIOC_STREAMOFF,
                &mut buf_type as *mut _ as *mut std::os::raw::c_void,
            )?;
        }

        self.active = false;
        Ok(())
    }
}

impl Drop for MmapStream {
    fn drop(&mut self) {
        if let Err(e) = self.stop() {
            if let Some(code) = e.raw_os_error() {
                if code == 19 {
                    // Device disconnected, ignore
                    return;
                }
            }
            eprintln!("Error stopping stream: {e}");
        }

        // Release buffers
        let mut v4l2_reqbufs = v4l2_requestbuffers {
            count: 0,
            type_: self.buf_type as u32,
            memory: Memory::Mmap as u32,
            ..unsafe { mem::zeroed() }
        };

        unsafe {
            let _ = v4l2::ioctl(
                self.handle.fd(),
                v4l2::vidioc::VIDIOC_REQBUFS,
                &mut v4l2_reqbufs as *mut _ as *mut std::os::raw::c_void,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that MmapBuffer cloning preserves data access
    #[test]
    fn test_mmap_buffer_clone_preserves_data() {
        // Create a test buffer with some data
        let test_data = b"Hello, World!";
        let mmap_info = Arc::new(MmapInfo {
            ptr: std::ptr::null_mut(),
            length: test_data.len(),
            offset: 0,
        });

        // Note: This test uses unsafe to create a buffer for testing purposes
        // In real usage, buffers are created from actual mmap'd memory
        let buffer =
            unsafe { MmapBuffer::new(test_data.as_ptr(), test_data.len(), mmap_info.clone()) };

        // Clone the buffer
        let cloned = buffer.clone();

        // Both should have the same data
        assert_eq!(buffer.as_slice(), test_data);
        assert_eq!(cloned.as_slice(), test_data);
        assert_eq!(buffer.as_slice(), cloned.as_slice());

        // Both should have the same length
        assert_eq!(buffer.len(), test_data.len());
        assert_eq!(cloned.len(), test_data.len());
    }

    /// Test that with_length correctly restricts buffer size
    #[test]
    fn test_mmap_buffer_with_length() {
        let test_data = b"Hello, World!";
        let mmap_info = Arc::new(MmapInfo {
            ptr: std::ptr::null_mut(),
            length: test_data.len(),
            offset: 0,
        });

        let buffer =
            unsafe { MmapBuffer::new(test_data.as_ptr(), test_data.len(), mmap_info.clone()) };

        // Create a shorter buffer
        let shorter = buffer.with_length(5);
        assert_eq!(shorter.len(), 5);
        assert_eq!(shorter.as_slice(), b"Hello");

        // Create a buffer with length larger than original (should be clamped)
        let longer = buffer.with_length(100);
        assert_eq!(longer.len(), test_data.len());
        assert_eq!(longer.as_slice(), test_data);

        // Create a buffer with zero length
        let empty = buffer.with_length(0);
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
    }

    /// Test that as_slice returns correct data
    #[test]
    fn test_mmap_buffer_as_slice() {
        let test_data = b"Test Data";
        let mmap_info = Arc::new(MmapInfo {
            ptr: std::ptr::null_mut(),
            length: test_data.len(),
            offset: 0,
        });

        let buffer =
            unsafe { MmapBuffer::new(test_data.as_ptr(), test_data.len(), mmap_info.clone()) };

        let slice = buffer.as_slice();
        assert_eq!(slice, test_data);
        assert_eq!(slice.len(), test_data.len());
    }

    /// Test len() and is_empty() methods
    #[test]
    fn test_mmap_buffer_len_and_is_empty() {
        let test_data = b"Data";
        let mmap_info = Arc::new(MmapInfo {
            ptr: std::ptr::null_mut(),
            length: test_data.len(),
            offset: 0,
        });

        let buffer =
            unsafe { MmapBuffer::new(test_data.as_ptr(), test_data.len(), mmap_info.clone()) };

        assert_eq!(buffer.len(), 4);
        assert!(!buffer.is_empty());

        let empty = buffer.with_length(0);
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
    }

    /// Test that Deref works correctly
    #[test]
    fn test_mmap_buffer_deref() {
        let test_data = b"Hello";
        let mmap_info = Arc::new(MmapInfo {
            ptr: std::ptr::null_mut(),
            length: test_data.len(),
            offset: 0,
        });

        let buffer =
            unsafe { MmapBuffer::new(test_data.as_ptr(), test_data.len(), mmap_info.clone()) };

        // Test Deref to [u8]
        let slice: &[u8] = &*buffer;
        assert_eq!(slice, test_data);
    }

    /// Test into_vec copies data correctly
    #[test]
    fn test_mmap_buffer_into_vec() {
        let test_data = b"Copy Test";
        let mmap_info = Arc::new(MmapInfo {
            ptr: std::ptr::null_mut(),
            length: test_data.len(),
            offset: 0,
        });

        let buffer =
            unsafe { MmapBuffer::new(test_data.as_ptr(), test_data.len(), mmap_info.clone()) };

        let vec = buffer.into_vec();
        assert_eq!(vec.as_slice(), test_data);
        assert_eq!(vec.len(), test_data.len());
    }
}
