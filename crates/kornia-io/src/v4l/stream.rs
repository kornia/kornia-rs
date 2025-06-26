use super::arena::Arena;
use std::{io, mem, sync::Arc};
use v4l::{
    buffer::{Metadata, Type},
    device::{Device, Handle},
    io::traits::{CaptureStream, Stream as StreamTrait},
    memory::Memory,
    v4l2,
    v4l_sys::*,
};

/// Stream of user buffers
///
/// An arena instance is used internally for buffer handling.
pub struct Stream {
    handle: Arc<Handle>,
    arena: Arena,
    arena_index: usize,
    buf_type: Type,
    buf_meta: Vec<Metadata>,
    timeout: Option<i32>,

    active: bool,
}

impl Stream {
    /// Returns a stream for frame capturing
    ///
    /// # Arguments
    ///
    /// * `dev` - Device ref to get its file descriptor
    /// * `buf_type` - Type of the buffers
    ///
    /// # Example
    ///
    /// ```
    /// use v4l::buffer::Type;
    /// use v4l::device::Device;
    /// use v4l::io::userptr::Stream;
    ///
    /// let dev = Device::new(0);
    /// if let Ok(dev) = dev {
    ///     let stream = Stream::new(&dev, Type::VideoCapture);
    /// }
    /// ```
    #[allow(dead_code)]
    pub fn new(dev: &Device, buf_type: Type) -> io::Result<Self> {
        Stream::with_buffers(dev, buf_type, 4)
    }

    pub fn with_buffers(dev: &Device, buf_type: Type, buf_count: u32) -> io::Result<Self> {
        let mut arena = Arena::new(dev.handle(), buf_type);
        let count = arena.allocate(buf_count)?;
        let mut buf_meta = Vec::new();
        buf_meta.resize(count as usize, Metadata::default());

        Ok(Stream {
            handle: dev.handle(),
            arena,
            arena_index: 0,
            buf_type,
            buf_meta,
            active: false,
            timeout: None,
        })
    }

    fn buffer_desc(&self) -> v4l2_buffer {
        v4l2_buffer {
            type_: self.buf_type as u32,
            memory: Memory::UserPtr as u32,
            ..unsafe { mem::zeroed() }
        }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if let Err(e) = self.stop() {
            if let Some(code) = e.raw_os_error() {
                // ENODEV means the file descriptor wrapped in the handle became invalid, most
                // likely because the device was unplugged or the connection (USB, PCI, ..)
                // broke down. Handle this case gracefully by ignoring it.
                if code == 19 {
                    /* ignore */
                    return;
                }
            }

            panic!("{:?}", e)
        }
    }
}

impl StreamTrait for Stream {
    type Item = super::arena::V4lBuffer;

    fn start(&mut self) -> io::Result<()> {
        unsafe {
            let mut typ = self.buf_type as u32;
            v4l2::ioctl(
                self.handle.fd(),
                v4l2::vidioc::VIDIOC_STREAMON,
                &mut typ as *mut _ as *mut std::os::raw::c_void,
            )?;
        }

        self.active = true;
        Ok(())
    }

    fn stop(&mut self) -> io::Result<()> {
        unsafe {
            let mut typ = self.buf_type as u32;
            v4l2::ioctl(
                self.handle.fd(),
                v4l2::vidioc::VIDIOC_STREAMOFF,
                &mut typ as *mut _ as *mut std::os::raw::c_void,
            )?;
        }

        self.active = false;
        Ok(())
    }
}

impl CaptureStream<'_> for Stream {
    fn queue(&mut self, index: usize) -> io::Result<()> {
        let buffer_desc = self.buffer_desc();
        let buf = &mut self.arena.bufs[index];
        let buf = &buf;

        let mut v4l2_buf = v4l2_buffer {
            index: index as u32,
            m: v4l2_buffer__bindgen_ty_1 {
                userptr: buf.as_ptr() as std::os::raw::c_ulong,
            },
            length: buf.len() as u32,
            ..buffer_desc
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

    fn dequeue(&mut self) -> io::Result<usize> {
        let mut v4l2_buf = self.buffer_desc();

        if self.handle.poll(libc::POLLIN, self.timeout.unwrap_or(-1))? == 0 {
            // This condition can only happen if there was a timeout.
            // A timeout is only possible if the `timeout` value is non-zero, meaning we should
            // propagate it to the caller.
            return Err(io::Error::new(io::ErrorKind::TimedOut, "VIDIOC_DQBUF"));
        }

        unsafe {
            v4l2::ioctl(
                self.handle.fd(),
                v4l2::vidioc::VIDIOC_DQBUF,
                &mut v4l2_buf as *mut _ as *mut std::os::raw::c_void,
            )?;
        }
        self.arena_index = v4l2_buf.index as usize;

        self.buf_meta[self.arena_index] = Metadata {
            bytesused: v4l2_buf.bytesused,
            flags: v4l2_buf.flags.into(),
            field: v4l2_buf.field,
            timestamp: v4l2_buf.timestamp.into(),
            sequence: v4l2_buf.sequence,
        };

        Ok(self.arena_index)
    }

    fn next(&mut self) -> io::Result<(&Self::Item, &Metadata)> {
        if !self.active {
            // Enqueue all buffers once on stream start
            for index in 0..self.arena.bufs.len() {
                self.queue(index)?;
            }

            self.start()?;
        } else {
            self.queue(self.arena_index)?;
        }

        self.arena_index = self.dequeue()?;

        // The index used to access the buffer elements is given to us by v4l2, so we assume it
        // will always be valid.
        let bytes = &self.arena.bufs[self.arena_index];
        let meta = &self.buf_meta[self.arena_index];

        Ok((bytes, meta))
    }
}
