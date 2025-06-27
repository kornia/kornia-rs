// Modified from https://github.com/raymanfx/libv4l-rs/blob/30f6dbaeed5cdb1b33760fc5b35c1b545cfe0e46/src/io/userptr/arena.rs
use std::{io, mem, sync::Arc};
use v4l::{buffer, device::Handle, memory::Memory, v4l2, v4l_sys::*};

#[cfg(feature = "arrow")]
use arrow::array::{Array, ArrayRef, UInt8Array};

/// Trait for converting to Arrow arrays
#[cfg(feature = "arrow")]
pub trait IntoArrow {
    type Output: Array;

    fn into_arrow(self) -> ArrayRef;
}

/// Abstracts a buffer from the v4l device.
#[derive(Clone)]
pub struct V4lBuffer(pub Arc<Vec<u8>>);

impl std::ops::Deref for V4lBuffer {
    type Target = Vec<u8>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(feature = "arrow")]
impl IntoArrow for V4lBuffer {
    type Output = UInt8Array;

    fn into_arrow(self) -> ArrayRef {
        // Try to unwrap the Arc to avoid copying if we're the only reference
        let data = match Arc::try_unwrap(self.0) {
            Ok(data) => data,
            Err(arc_data) => {
                // If there are other references, we need to copy the data
                arc_data.as_slice().to_vec()
            }
        };
        Arc::new(UInt8Array::from(data))
    }
}

#[cfg(feature = "arrow")]
impl V4lBuffer {
    /// Convert the V4lBuffer to an Arrow UInt8Array by reference (always copies)
    pub fn to_arrow(&self) -> ArrayRef {
        Arc::new(UInt8Array::from(self.0.as_slice().to_vec()))
    }
}

/// Manage user allocated buffers
///
/// All buffers are released in the Drop impl.
pub struct Arena {
    handle: Arc<Handle>,
    pub bufs: Vec<V4lBuffer>,
    pub buf_type: buffer::Type,
}

impl Arena {
    /// Returns a new buffer manager instance
    ///
    /// You usually do not need to use this directly.
    /// A UserBufferStream creates its own manager instance by default.
    ///
    /// # Arguments
    ///
    /// * `dev` - Device handle to get its file descriptor
    /// * `buf_type` - Type of the buffers
    pub fn new(handle: Arc<Handle>, buf_type: buffer::Type) -> Self {
        Arena {
            handle,
            bufs: Vec::new(),
            buf_type,
        }
    }

    fn requestbuffers_desc(&self) -> v4l2_requestbuffers {
        v4l2_requestbuffers {
            type_: self.buf_type as u32,
            memory: Memory::UserPtr as u32,
            ..unsafe { mem::zeroed() }
        }
    }

    pub fn allocate(&mut self, count: u32) -> io::Result<u32> {
        // we need to get the maximum buffer size from the format first
        let mut v4l2_fmt = v4l2_format {
            type_: self.buf_type as u32,
            ..unsafe { mem::zeroed() }
        };
        unsafe {
            v4l2::ioctl(
                self.handle.fd(),
                v4l2::vidioc::VIDIOC_G_FMT,
                &mut v4l2_fmt as *mut _ as *mut std::os::raw::c_void,
            )?;
        }

        let mut v4l2_reqbufs = v4l2_requestbuffers {
            count,
            ..self.requestbuffers_desc()
        };
        unsafe {
            v4l2::ioctl(
                self.handle.fd(),
                v4l2::vidioc::VIDIOC_REQBUFS,
                &mut v4l2_reqbufs as *mut _ as *mut std::os::raw::c_void,
            )?;
        }

        // allocate the new user buffers
        self.bufs.resize(
            v4l2_reqbufs.count as usize,
            V4lBuffer(Arc::new(vec![
                0;
                unsafe { v4l2_fmt.fmt.pix.sizeimage as usize }
            ])),
        );
        for i in 0..v4l2_reqbufs.count {
            let buf = &mut self.bufs[i as usize];
            *buf = V4lBuffer(Arc::new(vec![
                0;
                unsafe { v4l2_fmt.fmt.pix.sizeimage as usize }
            ]));
        }

        Ok(v4l2_reqbufs.count)
    }

    pub fn release(&mut self) -> io::Result<()> {
        // free all buffers by requesting 0
        let mut v4l2_reqbufs = v4l2_requestbuffers {
            count: 0,
            ..self.requestbuffers_desc()
        };
        unsafe {
            v4l2::ioctl(
                self.handle.fd(),
                v4l2::vidioc::VIDIOC_REQBUFS,
                &mut v4l2_reqbufs as *mut _ as *mut std::os::raw::c_void,
            )
        }
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        if self.bufs.is_empty() {
            // nothing to do
            return;
        }

        if let Err(e) = self.release() {
            if let Some(code) = e.raw_os_error() {
                // ENODEV means the file descriptor wrapped in the handle became invalid, most
                // likely because the device was unplugged or the connection (USB, PCI, ..)
                // broke down. Handle this case gracefully by ignoring it.
                if code == 19 {
                    /* ignore */
                    return;
                }
            }

            panic!("{e:?}")
        }
    }
}
