//! Ownership handles for a tensor's backing memory (host or device).
//!
//! This module provides the [`MemoryResource`] trait and two concrete implementations:
//! - [`HostResource`]: kornia-owned host memory (allocated and freed here).
//! - [`ForeignResource`]: memory owned externally (numpy, gstreamer, v4l, dlpack, cudarc-wrap).
//!
//! The [`MemoryDomain`] enum describes where a buffer can be legally dereferenced.

use std::{alloc::Layout, any::Any, ptr::NonNull, sync::Arc};

use crate::allocator::TensorAllocatorError;

/// Where a tensor's buffer can be legally dereferenced (the accessibility axis).
///
/// This three-state enum replaces the previous two-state `Host`/`Device` pair by
/// adding a [`Unified`](MemoryDomain::Unified) variant for pinned / managed / unified
/// memory that is simultaneously accessible from both the host and a device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryDomain {
    /// Host (CPU) memory; slice access is safe.
    Host,
    /// Device memory; host slice access is unsound.
    Device {
        /// CUDA (or other accelerator) device identifier.
        id: i32,
    },
    /// Host- AND device-accessible (pinned / managed / unified / NVMM-mapped).
    Unified {
        /// CUDA (or other accelerator) device identifier.
        id: i32,
    },
}

impl MemoryDomain {
    /// Returns `true` when the pointer may be dereferenced on the host (slice APIs).
    ///
    /// Both [`Host`](MemoryDomain::Host) and [`Unified`](MemoryDomain::Unified) memory
    /// satisfy this predicate.
    pub fn is_host_accessible(&self) -> bool {
        matches!(self, MemoryDomain::Host | MemoryDomain::Unified { .. })
    }

    /// Returns `true` when the pointer may be passed to a device kernel.
    ///
    /// Both [`Device`](MemoryDomain::Device) and [`Unified`](MemoryDomain::Unified) memory
    /// satisfy this predicate.
    pub fn is_device_accessible(&self) -> bool {
        matches!(self, MemoryDomain::Device { .. } | MemoryDomain::Unified { .. })
    }

    /// Returns the CUDA device id (0 for host; the embedded id for Device/Unified).
    pub fn device_id(&self) -> i32 {
        match self {
            MemoryDomain::Host => 0,
            MemoryDomain::Device { id } | MemoryDomain::Unified { id } => *id,
        }
    }
}

/// An owning handle to a tensor's backing memory.
///
/// Implementations are responsible for freeing the memory correctly on [`Drop`].
///
/// # Safety
///
/// Implementors must guarantee:
/// - `as_ptr()` returns a pointer that is valid for reads and writes of `len_bytes()` bytes
///   for the entire lifetime of the resource.
/// - `Drop` releases the memory exactly once (no double-free, no leak).
pub trait MemoryResource: Send + Sync {
    /// Base pointer (host- or device-addressable per [`domain`](Self::domain)).
    fn as_ptr(&self) -> *mut u8;

    /// Length of the backing buffer in bytes.
    fn len_bytes(&self) -> usize;

    /// Accessibility of the backing buffer.
    fn domain(&self) -> MemoryDomain;

    /// Downcast hook (e.g. recover a `&CudaSlice` from a `CudaResource`).
    fn as_any(&self) -> &dyn Any;
}

/// Host memory owned by kornia (allocated here, freed here on drop).
///
/// The backing buffer is allocated with [`std::alloc::alloc_zeroed`] and freed with
/// [`std::alloc::dealloc`].  A zero-sized layout produces a dangling (but valid) pointer
/// and no allocation / deallocation is performed.
pub struct HostResource {
    ptr: NonNull<u8>,
    layout: Layout,
}

impl HostResource {
    /// Allocate a zeroed host buffer of `layout`.
    ///
    /// Returns [`TensorAllocatorError::NullPointer`] if the global allocator returns null.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::alloc::Layout;
    /// use kornia_tensor::resource::HostResource;
    ///
    /// let layout = Layout::from_size_align(64, 8).unwrap();
    /// let r = HostResource::from_layout(layout).unwrap();
    /// assert_eq!(r.len_bytes(), 64);
    /// ```
    pub fn from_layout(layout: Layout) -> Result<Self, TensorAllocatorError> {
        if layout.size() == 0 {
            return Ok(Self {
                ptr: NonNull::dangling(),
                layout,
            });
        }
        // SAFETY: layout.size() > 0 checked above.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(ptr).ok_or(TensorAllocatorError::NullPointer)?;
        Ok(Self { ptr, layout })
    }

    /// Adopt a host pointer previously allocated with `layout`'s global allocator.
    ///
    /// # Safety
    ///
    /// `ptr` must have been returned by the global allocator using this exact `layout`.
    /// Ownership is transferred; this resource's [`Drop`] will call
    /// `std::alloc::dealloc(ptr, layout)`.
    pub unsafe fn from_raw(
        ptr: *mut u8,
        layout: Layout,
    ) -> Result<Self, TensorAllocatorError> {
        Ok(Self {
            ptr: NonNull::new(ptr).ok_or(TensorAllocatorError::NullPointer)?,
            layout,
        })
    }

    /// Returns the byte length of this resource's backing buffer.
    pub fn len_bytes(&self) -> usize {
        self.layout.size()
    }

    /// Returns `true` if the backing buffer has zero bytes.
    pub fn is_empty(&self) -> bool {
        self.layout.size() == 0
    }
}

impl MemoryResource for HostResource {
    fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    fn len_bytes(&self) -> usize {
        self.layout.size()
    }

    fn domain(&self) -> MemoryDomain {
        MemoryDomain::Host
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Drop for HostResource {
    fn drop(&mut self) {
        if self.layout.size() != 0 {
            // SAFETY: ptr was allocated with this exact layout.
            unsafe { std::alloc::dealloc(self.ptr.as_ptr(), self.layout) }
        }
    }
}

// SAFETY: the pointer is uniquely owned; no interior mutability shared across threads.
unsafe impl Send for HostResource {}
// SAFETY: shared references to HostResource never mutate the buffer.
unsafe impl Sync for HostResource {}

/// Foreign memory that kornia does NOT own: numpy/gstreamer/v4l/dlpack/cudarc-wrap.
///
/// On `Drop` the bytes themselves are NOT freed — only the optional `_keep` guard is dropped,
/// whose own `Drop` is responsible for releasing the source buffer (e.g. decrement a Python
/// refcount, unmap a GStreamer buffer, free a CUDA slice, …).
pub struct ForeignResource {
    ptr: NonNull<u8>,
    len_bytes: usize,
    domain: MemoryDomain,
    /// Keep-alive guard — held purely for its `Drop` side-effect.
    #[allow(dead_code)]
    _keep: Option<Arc<dyn Any + Send + Sync>>,
}

impl ForeignResource {
    /// Wrap a foreign pointer.
    ///
    /// # Arguments
    ///
    /// * `ptr`       — base pointer (host or device, per `domain`).
    /// * `len_bytes` — number of bytes in the buffer.
    /// * `domain`    — accessibility of the pointer.
    /// * `keep`      — optional keep-alive; its `Drop` releases the source allocation.
    ///
    /// # Safety
    ///
    /// - `ptr` must be valid (non-null) and readable/writable for `len_bytes` bytes.
    /// - The buffer must remain alive for the entire lifetime of this `ForeignResource`.
    ///   If `keep` is `Some`, it must own the underlying allocation and release it on drop.
    /// - `ptr` must NOT be freed by any code other than the `keep` drop path.
    pub unsafe fn new(
        ptr: *mut u8,
        len_bytes: usize,
        domain: MemoryDomain,
        keep: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Self, TensorAllocatorError> {
        Ok(Self {
            ptr: NonNull::new(ptr).ok_or(TensorAllocatorError::NullPointer)?,
            len_bytes,
            domain,
            _keep: keep,
        })
    }
}

impl MemoryResource for ForeignResource {
    fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    fn len_bytes(&self) -> usize {
        self.len_bytes
    }

    fn domain(&self) -> MemoryDomain {
        self.domain
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// SAFETY: the pointed-to memory is guaranteed valid by the caller of `new`; _keep is Arc.
unsafe impl Send for ForeignResource {}
// SAFETY: same reasoning; &ForeignResource never mutates the buffer.
unsafe impl Sync for ForeignResource {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn domain_accessibility() {
        assert!(MemoryDomain::Host.is_host_accessible());
        assert!(!MemoryDomain::Host.is_device_accessible());
        assert!(!MemoryDomain::Device { id: 1 }.is_host_accessible());
        assert!(MemoryDomain::Device { id: 1 }.is_device_accessible());
        assert!(MemoryDomain::Unified { id: 0 }.is_host_accessible());
        assert!(MemoryDomain::Unified { id: 0 }.is_device_accessible());
        assert_eq!(MemoryDomain::Device { id: 3 }.device_id(), 3);
        assert_eq!(MemoryDomain::Host.device_id(), 0);
    }

    #[test]
    fn host_resource_alloc_and_zeroed() {
        let layout = Layout::from_size_align(128, 8).unwrap();
        let r = HostResource::from_layout(layout).unwrap();
        assert_eq!(r.len_bytes(), 128);
        assert!(r.domain().is_host_accessible());
        assert!(!r.domain().is_device_accessible());
        // Must be zeroed.
        let slice = unsafe { std::slice::from_raw_parts(r.as_ptr(), 128) };
        assert!(slice.iter().all(|&b| b == 0));
    }

    #[test]
    fn host_resource_zero_size() {
        let layout = Layout::from_size_align(0, 1).unwrap();
        let r = HostResource::from_layout(layout).unwrap();
        assert_eq!(r.len_bytes(), 0);
        assert!(r.is_empty());
    }

    #[test]
    fn foreign_resource_keep_drops_once() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };

        struct Guard(Arc<AtomicUsize>);
        impl Drop for Guard {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let count = Arc::new(AtomicUsize::new(0));
        let buf = vec![1u8, 2, 3, 4];
        {
            let keep: Arc<dyn Any + Send + Sync> = Arc::new(Guard(count.clone()));
            let r = unsafe {
                ForeignResource::new(
                    buf.as_ptr() as *mut u8,
                    buf.len(),
                    MemoryDomain::Host,
                    Some(keep),
                )
            }
            .unwrap();
            assert_eq!(r.len_bytes(), 4);
            assert!(r.domain().is_host_accessible());
        }
        // Guard must have been dropped exactly once.
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }
}
