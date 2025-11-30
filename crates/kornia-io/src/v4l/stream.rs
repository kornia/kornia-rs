use std::{
    io, mem,
    sync::{
        atomic::{AtomicPtr, Ordering},
        Arc,
    },
};
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
/// This buffer wraps a raw pointer to mmap'd memory and ensures proper cleanup
/// via the `MmapInfo` struct. The buffer itself does not own the memory - it's
/// managed by the `MmapStream` and will be unmapped when the stream is dropped.
///
/// # Safety
///
/// This buffer is only valid while the `MmapStream` that created it is still alive.
/// The buffer should not be used after the stream is dropped.
pub struct MmapBuffer {
    /// Raw pointer to the mmap'd memory
    ptr: *const u8,
    /// Length of the buffer in bytes
    length: usize,
    /// Reference to the mmap info for proper cleanup tracking
    /// This ensures the memory is not unmapped while the buffer is still in use
    _mmap_info: Arc<MmapInfo>,
}

unsafe impl Send for MmapBuffer {}
unsafe impl Sync for MmapBuffer {}

impl MmapBuffer {
    /// Create a new MmapBuffer from a raw pointer and length.
    ///
    /// # Safety
    ///
    /// - `ptr` must be a valid pointer to `length` bytes of memory
    /// - The memory must remain valid for the lifetime of the buffer
    /// - `mmap_info` must be the same `MmapInfo` that was used to create the mmap
    unsafe fn new(ptr: *const u8, length: usize, mmap_info: Arc<MmapInfo>) -> Self {
        Self {
            ptr,
            length,
            _mmap_info: mmap_info,
        }
    }

    /// Get the buffer data as a slice
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.length) }
    }

    /// Convert the buffer to an owned `Box<[u8]>` by copying the data.
    ///
    /// This is necessary because the mmap'd memory is tied to the stream lifetime.
    /// Only copies the actual used bytes if `used_bytes` is provided.
    pub fn to_boxed_slice(&self, used_bytes: Option<usize>) -> Box<[u8]> {
        let len = used_bytes.unwrap_or(self.length).min(self.length);
        let slice = unsafe { std::slice::from_raw_parts(self.ptr, len) };
        slice.to_vec().into_boxed_slice()
    }
}

impl std::ops::Deref for MmapBuffer {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

/// Information about memory-mapped buffer
struct MmapInfo {
    ptr: AtomicPtr<u8>,
    length: usize,
    #[allow(dead_code)]
    offset: u32,
}

impl Drop for MmapInfo {
    fn drop(&mut self) {
        let ptr = self.ptr.load(Ordering::Relaxed);
        if !ptr.is_null() {
            unsafe {
                let result = libc::munmap(ptr as *mut libc::c_void, self.length);
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
    buffers: Vec<(MmapBuffer, Arc<MmapInfo>)>,
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
                ptr: AtomicPtr::new(ptr as *mut u8),
                length,
                offset,
            });

            // Create a safe buffer wrapper for the mmap'd memory
            let buffer = unsafe { MmapBuffer::new(ptr as *const u8, length, mmap_info.clone()) };

            buffers.push((buffer, mmap_info));
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

    /// Get the next frame and return the buffer data as an owned `Box<[u8]>`.
    ///
    /// The data is copied from the mmap'd buffer to ensure it can outlive the stream.
    /// Only the actual used bytes (from metadata.bytesused) are copied for efficiency.
    pub fn next_frame(&mut self) -> io::Result<(Box<[u8]>, Metadata)> {
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

        let (buffer, _) = &self.buffers[self.current_index];
        let metadata = self.buf_meta[self.current_index];

        // Copy the data from mmap'd memory to an owned buffer
        // Use bytesused if available, otherwise use full buffer length
        let used_bytes = if metadata.bytesused > 0 {
            Some(metadata.bytesused as usize)
        } else {
            None
        };
        let owned_buffer = buffer.to_boxed_slice(used_bytes);

        Ok((owned_buffer, metadata))
    }
}

impl StreamTrait for MmapStream {
    type Item = Box<[u8]>;

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
