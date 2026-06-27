use std::{alloc::Layout, marker::PhantomData, ptr::NonNull};

use crate::allocator::{CpuAllocator, TensorAllocator};
use crate::resource::{ForeignResource, HostResource, MemoryResource};

// MemoryDomain is now defined in `resource` and re-exported from there.
// This re-export keeps all `storage::MemoryDomain` use-sites working unchanged.
pub use crate::resource::MemoryDomain;

/// Low-level memory buffer for tensor data.
///
/// `TensorStorage` manages a contiguous block of memory that holds the actual data for a tensor.
/// It uses a single `owner: Box<dyn MemoryResource>` that carries the correct deallocation
/// strategy for every provenance: kornia-owned host memory ([`HostResource`]), foreign/borrowed
/// memory ([`ForeignResource`]), or future device memory resources.
///
/// # Memory Management
///
/// The storage owns its `owner` handle and the backing buffer is freed when the handle's
/// `Drop` runs — exactly once, regardless of whether the storage was constructed from a
/// `Vec`, a raw pointer, or a borrowed external buffer.
///
/// # Thread Safety
///
/// `TensorStorage` is `Send` and `Sync` when the allocator is thread-safe and `T: Send + Sync`,
/// allowing tensors to be safely shared across threads.
///
/// # Examples
///
/// Creating storage from a vector:
///
/// ```rust
/// use kornia_tensor::{storage::TensorStorage, CpuAllocator};
///
/// let data = vec![1, 2, 3, 4, 5];
/// let storage = TensorStorage::from_vec(data, CpuAllocator);
///
/// assert_eq!(storage.as_slice(), &[1, 2, 3, 4, 5]);
/// assert!(!storage.is_empty());
/// ```
///
/// Converting back to a vector:
///
/// ```rust
/// use kornia_tensor::{storage::TensorStorage, CpuAllocator};
///
/// let data = vec![1.0, 2.0, 3.0];
/// let storage = TensorStorage::from_vec(data, CpuAllocator);
/// let recovered = storage.into_vec();
///
/// assert_eq!(recovered, vec![1.0, 2.0, 3.0]);
/// ```
pub struct TensorStorage<T, A: TensorAllocator = CpuAllocator> {
    /// Cached hot-path pointer to the first element of the backing buffer.
    pub(crate) ptr: NonNull<T>,
    /// Length of the backing buffer in bytes (NOT number of elements).
    pub(crate) len: usize,
    /// Owning handle that frees the backing buffer correctly on Drop.
    pub(crate) owner: Box<dyn MemoryResource>,
    /// The allocator associated with this storage (type tag / future allocation use).
    pub(crate) alloc: A,
    pub(crate) _marker: PhantomData<T>,
}

impl<T, A: TensorAllocator> TensorStorage<T, A> {
    /// Returns a raw const pointer to the storage's first element.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Returns a raw mutable pointer to the storage's first element.
    ///
    /// # Panics
    ///
    /// Panics if the storage was created with a read-only resource (e.g. via
    /// [`from_borrowed_readonly`](Self::from_borrowed_readonly)).
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        assert!(!self.owner.is_readonly(), "as_mut_ptr on read-only memory");
        self.ptr.as_ptr()
    }

    /// Returns the memory domain for this storage.
    #[inline]
    pub fn domain(&self) -> MemoryDomain {
        self.owner.domain()
    }

    /// Returns the device id for this storage (0 for host; CUDA device id for Device/Unified).
    #[inline]
    pub fn device_id(&self) -> i32 {
        self.owner.domain().device_id()
    }

    /// Returns the storage data as a slice.
    ///
    /// # Panics
    ///
    /// Panics if the storage is non-host-accessible (i.e. `MemoryDomain::Device`).
    /// Use explicit host-device transfer APIs to access device data.
    pub fn as_slice(&self) -> &[T] {
        let domain = self.owner.domain();
        assert!(
            domain.is_host_accessible(),
            "as_slice on non-host-accessible memory (domain={:?})",
            domain
        );
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.len / std::mem::size_of::<T>()) }
    }

    /// Returns the storage data as a mutable slice.
    ///
    /// # Panics
    ///
    /// Panics if the storage is non-host-accessible (i.e. `MemoryDomain::Device`).
    /// Also panics if the storage was created with a read-only resource (e.g. via
    /// [`from_borrowed_readonly`](Self::from_borrowed_readonly)).
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        let domain = self.owner.domain();
        assert!(
            domain.is_host_accessible(),
            "as_mut_slice on non-host-accessible memory (domain={:?})",
            domain
        );
        assert!(
            !self.owner.is_readonly(),
            "as_mut_slice on read-only memory"
        );
        unsafe {
            std::slice::from_raw_parts_mut(self.as_mut_ptr(), self.len / std::mem::size_of::<T>())
        }
    }

    /// Returns the number of bytes in this storage.
    ///
    /// Note: This returns the size in bytes, not the number of elements.
    /// To get the number of elements, divide by `std::mem::size_of::<T>()`.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the storage has a length of 0.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the memory layout of the storage (reconstructed from `len` and type alignment).
    #[inline]
    pub fn layout(&self) -> Layout {
        // Reconstruct from the owner's byte count and T's alignment requirement.
        Layout::from_size_align(self.owner.len_bytes(), std::mem::align_of::<T>()).unwrap_or_else(
            |_| unsafe {
                Layout::from_size_align_unchecked(self.owner.len_bytes(), std::mem::align_of::<T>())
            },
        )
    }

    /// Returns a reference to the allocator used by this storage.
    #[inline]
    pub fn alloc(&self) -> &A {
        &self.alloc
    }

    /// Creates a new tensor storage from a vector.
    ///
    /// Takes ownership of the vector's heap allocation (no copy), wraps it in a
    /// [`HostResource`], and transfers all ownership to this storage. When the storage
    /// is dropped, `HostResource::Drop` frees the allocation exactly once.
    ///
    /// # Arguments
    ///
    /// * `value` - The vector to convert into storage
    /// * `alloc` - The allocator to associate with this storage
    pub fn from_vec(value: Vec<T>, alloc: A) -> Self {
        // Extract the Vec's innards without copying.
        // SAFETY: Vec::as_ptr is non-null for non-zero-capacity vecs; for cap == 0
        // the pointer is dangling but len == 0 so we never dereference it.
        let ptr = unsafe { NonNull::new_unchecked(value.as_ptr() as *mut T) };
        let len = value.len() * std::mem::size_of::<T>();
        // Layout::array::<T>(cap) matches what Vec's RawVec uses.
        let capacity = value.capacity();
        let layout = unsafe { Layout::array::<T>(capacity).unwrap_unchecked() };
        // Prevent Vec from freeing the memory when it goes out of scope.
        std::mem::forget(value);

        // Build a HostResource that owns this allocation and will free it on drop.
        // SAFETY: ptr came from the global allocator via Vec with this exact layout.
        let owner: Box<dyn MemoryResource> = Box::new(unsafe {
            HostResource::from_raw(ptr.as_ptr() as *mut u8, layout)
                .expect("Vec ptr is always non-null")
        });

        Self {
            ptr,
            len,
            owner,
            alloc,
            _marker: PhantomData,
        }
    }

    /// Creates a new tensor storage from raw parts (borrowed, no deallocation).
    ///
    /// The resulting storage does NOT own the memory — it creates a [`ForeignResource`]
    /// with no keep-alive, meaning the caller is responsible for ensuring the memory
    /// outlives this storage.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - The pointer is non-null and properly aligned.
    /// - The memory region is valid for `len` bytes for the entire lifetime of this storage.
    /// - No other code will free this memory while this storage exists.
    pub unsafe fn from_raw_parts(data: *const T, len: usize, alloc: A) -> Self {
        let ptr = NonNull::new_unchecked(data as *mut T);
        let owner: Box<dyn MemoryResource> = Box::new(
            ForeignResource::new(data as *mut u8, len, MemoryDomain::Host, None)
                .expect("non-null pointer required"),
        );
        Self {
            ptr,
            len,
            owner,
            alloc,
            _marker: PhantomData,
        }
    }

    /// Creates a new tensor storage that owns a host allocation produced by the allocator.
    ///
    /// Unlike [`from_vec`](Self::from_vec), this allows callers to supply a custom layout
    /// (e.g. 64-byte alignment) and transfer ownership of an already-allocated pointer.
    ///
    /// # Safety
    ///
    /// - `data` must be a valid, non-null host pointer for at least `len_bytes` bytes,
    ///   valid for type `T`.
    /// - `layout` must exactly match the layout used when `data` was allocated via the
    ///   global allocator.
    /// - Ownership is transferred; `HostResource::Drop` will call
    ///   `std::alloc::dealloc(data, layout)` on drop.
    pub unsafe fn from_raw_host(data: *mut T, len_bytes: usize, layout: Layout, alloc: A) -> Self {
        // The viewed length must lie within the allocation `layout` describes; otherwise
        // `as_slice` (len = len_bytes / size_of::<T>()) reads out of bounds and `into_vec`
        // would build a Vec with len > capacity (instant UB). Enforce the documented
        // contract rather than trusting the caller.
        assert!(
            layout.size() >= len_bytes,
            "from_raw_host: layout.size() ({}) < len_bytes ({}) — buffer too small for the view",
            layout.size(),
            len_bytes,
        );
        // Defense-in-depth: a null pointer here is always a caller bug; make it a clear
        // panic rather than undefined behaviour via `new_unchecked`.
        let ptr = NonNull::new(data)
            .expect("from_raw_host: null pointer — data must be a valid, non-null host pointer");
        let owner: Box<dyn MemoryResource> = Box::new(
            HostResource::from_raw(data as *mut u8, layout).expect("non-null pointer required"),
        );
        Self {
            ptr,
            len: len_bytes,
            owner,
            alloc,
            _marker: PhantomData,
        }
    }

    /// Creates a borrowed storage view that keeps `keepalive` alive until this storage is dropped.
    ///
    /// The storage does NOT own the memory bytes — [`ForeignResource::Drop`] does NOT free them.
    /// Instead, `keepalive` is held in an `Arc` so the source object (e.g. a numpy array,
    /// GStreamer buffer) lives at least as long as this storage.
    ///
    /// # Arguments
    ///
    /// * `data`      - Pointer to the first element of the buffer.
    /// * `len_bytes` - Byte length of the buffer.
    /// * `alloc`     - Allocator type tag.
    /// * `domain`    - Accessibility of the buffer (`Host`, `Device{id}`, or `Unified{id}`).
    /// * `keepalive` - Owned arc whose `Drop` releases the underlying allocation.
    ///
    /// # Safety
    ///
    /// - `data` must point to a valid, non-null allocation of at least `len_bytes` bytes valid for `T`.
    /// - The memory must remain valid for the full lifetime of this storage (guaranteed by `keepalive`).
    /// - `domain` must correctly describe where `data` lives.
    /// - For `Device` domain: do NOT call `as_slice`/`as_mut_slice` (they will panic).
    pub unsafe fn from_borrowed(
        data: *const T,
        len_bytes: usize,
        alloc: A,
        domain: MemoryDomain,
        keepalive: std::sync::Arc<dyn core::any::Any + Send + Sync>,
    ) -> Self {
        let ptr = NonNull::new_unchecked(data as *mut T);
        let owner: Box<dyn MemoryResource> = Box::new(
            ForeignResource::new(data as *mut u8, len_bytes, domain, Some(keepalive))
                .expect("non-null pointer required"),
        );
        Self {
            ptr,
            len: len_bytes,
            owner,
            alloc,
            _marker: PhantomData,
        }
    }

    /// Creates a borrowed storage view that is read-only.
    ///
    /// Like [`from_borrowed`](Self::from_borrowed) but the resulting storage refuses
    /// mutable slice access: [`as_mut_slice`](Self::as_mut_slice) will panic with
    /// `"as_mut_slice on read-only memory"`.
    ///
    /// Use this when the underlying buffer is mapped read-only by the OS (e.g. a
    /// GStreamer / V4L2 `mmap` buffer) so that callers cannot accidentally write
    /// to kernel-owned memory.
    ///
    /// # Arguments
    ///
    /// * `data`      - Pointer to the first element of the buffer.
    /// * `len_bytes` - Byte length of the buffer.
    /// * `alloc`     - Allocator type tag.
    /// * `domain`    - Accessibility of the buffer.
    /// * `keepalive` - Owned arc whose `Drop` releases the underlying allocation.
    ///
    /// # Returns
    ///
    /// A new `TensorStorage` backed by the given pointer, refusing mutable access.
    ///
    /// # Safety
    ///
    /// Same contract as [`from_borrowed`](Self::from_borrowed).
    pub unsafe fn from_borrowed_readonly(
        data: *const T,
        len_bytes: usize,
        alloc: A,
        domain: MemoryDomain,
        keepalive: std::sync::Arc<dyn core::any::Any + Send + Sync>,
    ) -> Self {
        let ptr = NonNull::new_unchecked(data as *mut T);
        let owner: Box<dyn MemoryResource> = Box::new(
            ForeignResource::new_readonly(data as *mut u8, len_bytes, domain, Some(keepalive))
                .expect("non-null pointer required"),
        );
        Self {
            ptr,
            len: len_bytes,
            owner,
            alloc,
            _marker: PhantomData,
        }
    }

    /// Consumes the storage and returns the underlying data as a vector.
    ///
    /// This transfers ownership of the memory from the storage to a `Vec<T>` without copying.
    /// The storage is consumed in the process.
    ///
    /// # Panics
    ///
    /// Panics if the owner is not a [`HostResource`] (i.e., the memory is foreign- or device-backed).
    pub fn into_vec(self) -> Vec<T> {
        let host = self
            .owner
            .as_any()
            .downcast_ref::<HostResource>()
            .expect("cannot convert foreign-memory-backed storage into Vec");
        // A `Vec<T>` always frees with `Layout::array::<T>(cap)` (align = align_of::<T>()).
        // Reconstructing a Vec from an over-aligned allocation (e.g. foreign memory with
        // 64-byte alignment) would dealloc with a mismatched layout — undefined behaviour.
        // `from_vec` is always safe (it adopts the Vec's own layout); only the
        // allocate-path with an over-aligned layout could trip this.
        assert_eq!(
            host.layout().align(),
            std::mem::align_of::<T>(),
            "into_vec: backing allocation alignment ({}) != align_of::<T>() ({}); \
             over-aligned storage is not Vec-reconstructable without a copy",
            host.layout().align(),
            std::mem::align_of::<T>(),
        );

        let vec_len = self.len / std::mem::size_of::<T>();
        let ptr = self.ptr;

        // Deconstruct `self` without running Drop on any field.
        // We take the `owner` out manually so we can call `HostResource::into_raw_parts`
        // (which suppresses the HostResource destructor and transfers allocation ownership
        // to the Vec we are about to build).
        let this = std::mem::ManuallyDrop::new(self);

        // SAFETY: We verified above that `owner` is a `Box<HostResource>`.
        // `Box::into_raw` gives us the raw fat pointer; we then convert the data pointer
        // to `*mut HostResource` and reconstruct the `Box<HostResource>` for consumption.
        // This is sound because `Box<dyn MemoryResource>` and `Box<HostResource>` share the
        // same memory representation for the concrete type.
        let vec_capacity = unsafe {
            // Read the Box and alloc out of the ManuallyDrop (does NOT run their Drop).
            let owner_box: Box<dyn MemoryResource> = std::ptr::read(&this.owner);
            // Drop the allocator field (unit struct for CpuAllocator/ForeignAllocator, but be explicit).
            drop(std::ptr::read(&this.alloc));
            // Convert the Box<dyn MemoryResource> into a raw fat pointer.
            let raw: *mut dyn MemoryResource = Box::into_raw(owner_box);
            // Cast the data portion to *mut HostResource (safe: we asserted the type above).
            let host_ptr: *mut HostResource = raw as *mut HostResource;
            // Reconstruct as Box<HostResource> — this also frees the fat-pointer heap allocation
            // (the allocation holding the HostResource struct, distinct from the buffer it owns).
            let host_box = Box::from_raw(host_ptr);
            // Move HostResource out of the Box; Box's heap alloc is freed here.
            let host: HostResource = *host_box;
            // Consume the HostResource without running its Drop (which would dealloc the buffer).
            // The returned pointer is the same as `ptr` above; we just need the layout.
            let (_buf_ptr, layout) = HostResource::into_raw_parts(host);
            layout.size() / std::mem::size_of::<T>()
        };

        // SAFETY: ptr is valid for vec_capacity elements of T; vec_len <= vec_capacity;
        // the global allocator owns this allocation; the Vec becomes the new owner.
        unsafe { Vec::from_raw_parts(ptr.as_ptr(), vec_len, vec_capacity) }
    }
}

// SAFETY: TensorStorage is the sole owner of the pointed-to memory.
// Sending it across threads is safe iff T is Send (same rule as Vec<T>).
unsafe impl<T: Send, A: TensorAllocator> Send for TensorStorage<T, A> {}
// SAFETY: Shared references to TensorStorage only allow reading T.
// This is safe iff T is Sync (same rule as &[T]).
unsafe impl<T: Sync, A: TensorAllocator> Sync for TensorStorage<T, A> {}

/// Drop is empty: the `owner: Box<dyn MemoryResource>` field drops itself,
/// calling the correct deallocation path (HostResource::dealloc, or ForeignResource
/// dropping its keepalive Arc, etc.) exactly once.
impl<T, A: TensorAllocator> Drop for TensorStorage<T, A> {
    fn drop(&mut self) {
        // Nothing to do — `owner` drops automatically via Box<dyn MemoryResource>.
    }
}

/// Clones the storage by creating a new storage with copied data.
///
/// This performs a deep copy of the storage data using the cloned allocator.
///
/// # Panics
///
/// Panics if the storage is non-host-accessible. Device-to-device copy is not yet
/// implemented; use an explicit transfer API when it becomes available.
impl<T, A> Clone for TensorStorage<T, A>
where
    T: Clone,
    A: TensorAllocator,
{
    fn clone(&self) -> Self {
        let domain = self.owner.domain();
        assert!(
            domain.is_host_accessible(),
            "clone called on device storage — device-to-device copy is not yet implemented"
        );
        Self::from_vec(self.as_slice().to_vec(), self.alloc.clone())
    }
}

#[cfg(test)]
mod tests {

    use super::TensorStorage;
    use crate::allocator::{CpuAllocator, ForeignAllocator, TensorAllocatorError};
    use crate::resource::{ForeignResource, HostResource, MemoryDomain, MemoryResource};
    use crate::TensorAllocator;
    use std::alloc::Layout;
    use std::marker::PhantomData;
    use std::ptr::NonNull;
    use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering};
    use std::sync::Arc;

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Build a storage that claims `MemoryDomain::Device` without touching real GPU hardware.
    ///
    /// Uses a custom `FakeDeviceResource` that wraps a real `HostResource` allocation but
    /// reports `Device` domain, so the memory is properly freed on Drop (no leaks under miri).
    fn make_device_storage() -> TensorStorage<u8, CpuAllocator> {
        /// Wraps a properly-allocated HostResource but reports Device domain.
        /// This lets us test Device-domain guards without any real GPU or leaked memory.
        struct FakeDeviceResource {
            inner: HostResource,
        }
        impl MemoryResource for FakeDeviceResource {
            fn as_ptr(&self) -> *mut u8 {
                self.inner.as_ptr()
            }
            fn len_bytes(&self) -> usize {
                self.inner.len_bytes()
            }
            fn domain(&self) -> MemoryDomain {
                MemoryDomain::Device { id: 0 }
            }
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
            fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
                self
            }
        }
        unsafe impl Send for FakeDeviceResource {}
        unsafe impl Sync for FakeDeviceResource {}

        let layout = Layout::array::<u8>(1).unwrap();
        let inner = HostResource::from_layout(layout).unwrap();
        let owner_ptr = inner.as_ptr();
        let owner: Box<dyn MemoryResource> = Box::new(FakeDeviceResource { inner });
        let ptr = unsafe { NonNull::new_unchecked(owner_ptr) };
        TensorStorage {
            ptr,
            len: 1,
            owner,
            alloc: CpuAllocator,
            _marker: PhantomData,
        }
    }

    // ── Original tests (migrated to new struct fields) ────────────────────────

    #[test]
    fn test_tensor_buffer_create_raw() -> Result<(), TensorAllocatorError> {
        let size = 8;
        let layout = Layout::array::<u8>(size).map_err(TensorAllocatorError::LayoutError)?;
        let owner = Box::new(HostResource::from_layout(layout)?) as Box<dyn MemoryResource>;
        let ptr_raw = owner.as_ptr();
        let ptr = unsafe { NonNull::new_unchecked(ptr_raw) };

        let buffer = TensorStorage {
            alloc: CpuAllocator,
            len: size * std::mem::size_of::<u8>(),
            owner,
            ptr,
            _marker: PhantomData,
        };

        assert_eq!(buffer.ptr.as_ptr() as *const u8, ptr_raw);
        assert!(!ptr_raw.is_null());
        assert_eq!(buffer.len(), size);
        assert!(!buffer.is_empty());
        assert_eq!(buffer.len(), size * std::mem::size_of::<u8>());

        Ok(())
    }

    #[test]
    fn test_tensor_buffer_ptr() -> Result<(), TensorAllocatorError> {
        let size = 8;
        let layout = Layout::array::<u8>(size).map_err(TensorAllocatorError::LayoutError)?;
        let r = CpuAllocator.allocate(layout)?;

        // check alignment
        let ptr_raw = r.as_ptr() as usize;
        let alignment = std::mem::align_of::<u8>();
        assert_eq!(ptr_raw % alignment, 0);

        Ok(())
    }

    #[test]
    #[should_panic(expected = "not Vec-reconstructable")]
    fn into_vec_rejects_over_aligned_allocation() {
        // Manually allocate a 64-byte-aligned buffer; align_of::<f32>() is 4.
        // into_vec must refuse: handing a 64-byte-aligned allocation to Vec would cause
        // it to dealloc with align_of::<f32>()==4 — a mismatched layout (undefined behaviour).
        let layout = Layout::from_size_align(4 * std::mem::size_of::<f32>(), 64).unwrap();
        let raw_ptr = unsafe { std::alloc::alloc(layout) } as *mut f32;
        assert!(!raw_ptr.is_null(), "allocation failed");
        // SAFETY: raw_ptr is non-null, valid for len_bytes, allocated with `layout` above.
        // from_raw_host takes ownership and will store the 64-byte-aligned layout in
        // HostResource, which into_vec will detect and reject.
        let storage = unsafe {
            TensorStorage::<f32, CpuAllocator>::from_raw_host(
                raw_ptr,
                layout.size(),
                layout,
                CpuAllocator,
            )
        };
        let _ = storage.into_vec();
    }

    #[test]
    fn test_tensor_buffer_create_f32() -> Result<(), TensorAllocatorError> {
        let size = 8;
        let layout = Layout::array::<f32>(size).map_err(TensorAllocatorError::LayoutError)?;
        let owner = Box::new(HostResource::from_layout(layout)?) as Box<dyn MemoryResource>;
        let ptr = unsafe { NonNull::new_unchecked(owner.as_ptr() as *mut f32) };

        let buffer = TensorStorage {
            alloc: CpuAllocator,
            len: size,
            owner,
            ptr,
            _marker: PhantomData,
        };

        assert_eq!(buffer.len(), size);

        Ok(())
    }

    #[test]
    fn test_tensor_buffer_lifecycle() -> Result<(), TensorAllocatorError> {
        /// A simple allocator that counts the number of bytes allocated.
        /// Uses Arc<AtomicI32> so it is Send + Sync (required by TensorAllocator).
        #[derive(Clone)]
        struct TestAllocator {
            bytes_allocated: Arc<AtomicI32>,
        }

        impl TensorAllocator for TestAllocator {
            fn allocate(
                &self,
                layout: Layout,
            ) -> Result<Box<dyn MemoryResource>, TensorAllocatorError> {
                self.bytes_allocated
                    .fetch_add(layout.size() as i32, Ordering::SeqCst);
                let r = HostResource::from_layout(layout)?;
                // Wrap in a counting resource so we can observe dealloc too.
                Ok(Box::new(r))
            }
        }

        let allocator = TestAllocator {
            bytes_allocated: Arc::new(AtomicI32::new(0)),
        };
        assert_eq!(allocator.bytes_allocated.load(Ordering::SeqCst), 0);

        let size = 1024;

        // TensorStorage::from_vec() -> TensorStorage::into_vec()
        // from_vec does NOT call the custom allocator — it wraps the Vec's own allocation.
        {
            let vec = Vec::<u8>::with_capacity(size);
            let vec_ptr = vec.as_ptr();
            let vec_capacity = vec.capacity();

            let buffer = TensorStorage::from_vec(vec, allocator.clone());
            assert_eq!(allocator.bytes_allocated.load(Ordering::SeqCst), 0);

            let result_vec = buffer.into_vec();
            assert_eq!(allocator.bytes_allocated.load(Ordering::SeqCst), 0);

            assert_eq!(result_vec.capacity(), vec_capacity);
            assert!(std::ptr::eq(result_vec.as_ptr(), vec_ptr));
        }
        assert_eq!(allocator.bytes_allocated.load(Ordering::SeqCst), 0);

        Ok(())
    }

    #[test]
    fn test_tensor_buffer_from_vec() -> Result<(), TensorAllocatorError> {
        let vec: Vec<i32> = vec![1, 2, 3, 4, 5];
        let vec_ptr = vec.as_ptr();
        let vec_len = vec.len();

        let buffer = TensorStorage::<_, CpuAllocator>::from_vec(vec, CpuAllocator);

        // check NO copy
        let buffer_ptr = buffer.as_ptr();
        assert!(std::ptr::eq(buffer_ptr, vec_ptr));

        // check alignment
        let buffer_ptr_usize = buffer.as_ptr() as usize;
        let alignment = std::mem::align_of::<i32>();
        assert_eq!(buffer_ptr_usize % alignment, 0);

        // check accessors
        let data = buffer.as_slice();
        assert_eq!(data.len(), vec_len);
        assert_eq!(data[0], 1);
        assert_eq!(data[1], 2);
        assert_eq!(data[2], 3);
        assert_eq!(data[3], 4);
        assert_eq!(data[4], 5);

        assert_eq!(data.first(), Some(&1));
        assert_eq!(data.get(1), Some(&2));
        assert_eq!(data.get(2), Some(&3));
        assert_eq!(data.get(3), Some(&4));
        assert_eq!(data.get(4), Some(&5));
        assert_eq!(data.get(5), None);

        unsafe {
            assert_eq!(data.get_unchecked(0), &1);
            assert_eq!(data.get_unchecked(1), &2);
            assert_eq!(data.get_unchecked(2), &3);
            assert_eq!(data.get_unchecked(3), &4);
            assert_eq!(data.get_unchecked(4), &5);
        }

        Ok(())
    }

    #[test]
    fn test_tensor_buffer_from_empty_vec() -> Result<(), TensorAllocatorError> {
        // Empty vec: no heap allocation; Drop should not crash.
        let vec: Vec<i32> = vec![];
        let vec_ptr = vec.as_ptr();
        let vec_len = vec.len();

        let buffer = TensorStorage::<_, CpuAllocator>::from_vec(vec, CpuAllocator);

        // check NO copy
        let buffer_ptr = buffer.as_ptr();
        assert!(std::ptr::eq(buffer_ptr, vec_ptr));

        // check accessors
        let data = buffer.as_slice();
        assert_eq!(data.len(), vec_len);

        Ok(())
    }

    #[test]
    fn test_tensor_buffer_into_vec() -> Result<(), TensorAllocatorError> {
        let vec: Vec<i32> = vec![1, 2, 3, 4, 5];
        let vec_ptr = vec.as_ptr();
        let vec_cap = vec.capacity();

        let buffer = TensorStorage::<_, CpuAllocator>::from_vec(vec, CpuAllocator);

        // convert back to vec (no copy)
        let result_vec = buffer.into_vec();

        assert_eq!(result_vec.capacity(), vec_cap);
        assert!(std::ptr::eq(result_vec.as_ptr(), vec_ptr));

        Ok(())
    }

    #[test]
    fn test_tensor_mutability() -> Result<(), TensorAllocatorError> {
        let vec: Vec<i32> = vec![1, 2, 3, 4, 5];
        let mut buffer = TensorStorage::<_, CpuAllocator>::from_vec(vec, CpuAllocator);
        let ptr_mut = buffer.as_mut_ptr();
        unsafe {
            *ptr_mut.add(0) = 10;
        }
        assert_eq!(buffer.into_vec(), vec![10, 2, 3, 4, 5]);
        Ok(())
    }

    #[test]
    #[should_panic(expected = "non-host-accessible")]
    fn test_as_slice_panics_on_device() {
        let storage = make_device_storage();
        let _ = storage.as_slice();
    }

    #[test]
    #[should_panic(expected = "non-host-accessible")]
    fn test_as_mut_slice_panics_on_device() {
        let mut storage = make_device_storage();
        let _ = storage.as_mut_slice();
    }

    #[test]
    #[should_panic(expected = "clone called on device storage")]
    fn test_clone_panics_on_device() {
        let storage = make_device_storage();
        let _ = storage.clone();
    }

    // ── Task 3 safety tests ───────────────────────────────────────────────────

    /// from_vec roundtrip + verify Host domain.
    #[test]
    fn from_vec_roundtrip_and_host_domain() {
        let s = TensorStorage::from_vec(vec![1u8, 2, 3, 4], CpuAllocator);
        assert_eq!(s.as_slice(), &[1, 2, 3, 4]);
        assert!(matches!(s.domain(), MemoryDomain::Host));
        assert_eq!(s.into_vec(), vec![1u8, 2, 3, 4]);
    }

    /// device `as_slice` must panic with "non-host-accessible" in the message.
    #[test]
    #[should_panic(expected = "non-host-accessible")]
    fn device_storage_as_slice_panics() {
        // Uses FakeDeviceResource so no memory is leaked.
        let s = make_device_storage();
        let _ = s.as_slice(); // must panic: Device is not host-accessible
    }

    /// borrowed keepalive must drop exactly once.
    #[test]
    fn borrowed_keepalive_drops_once() {
        struct Guard(Arc<AtomicUsize>);
        impl Drop for Guard {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let n = Arc::new(AtomicUsize::new(0));
        let buf = [7u8; 8];
        {
            let keep: Arc<dyn core::any::Any + Send + Sync> = Arc::new(Guard(n.clone()));
            let s = unsafe {
                TensorStorage::<u8, ForeignAllocator>::from_borrowed(
                    buf.as_ptr(),
                    8,
                    ForeignAllocator,
                    MemoryDomain::Host,
                    keep,
                )
            };
            assert_eq!(s.as_slice(), &[7u8; 8]);
        }
        assert_eq!(n.load(Ordering::SeqCst), 1); // guard dropped exactly once
    }

    /// Unified domain IS host-accessible: as_slice() must not panic.
    #[test]
    fn unified_domain_is_host_accessible() {
        let buf = vec![42u8; 4];
        let keep: Arc<dyn core::any::Any + Send + Sync> = Arc::new(buf.clone());
        let s = unsafe {
            TensorStorage::<u8, ForeignAllocator>::from_borrowed(
                buf.as_ptr(),
                4,
                ForeignAllocator,
                MemoryDomain::Unified { id: 0 },
                keep,
            )
        };
        // Must NOT panic:
        let slice = s.as_slice();
        assert_eq!(slice, &[42u8; 4]);
    }

    /// Owned-host no-double-free / no-leak: alloc counter returns to zero after drop.
    ///
    /// We implement a custom `MemoryResource` with an Arc drop-counter and use it
    /// as the owner directly.
    #[test]
    fn owned_host_no_double_free_no_leak() {
        use std::alloc::Layout;

        /// Counting MemoryResource wrapper: increments counter on alloc, decrements on drop.
        struct CountingResource {
            inner: HostResource,
            counter: Arc<AtomicUsize>,
        }
        impl MemoryResource for CountingResource {
            fn as_ptr(&self) -> *mut u8 {
                self.inner.as_ptr()
            }
            fn len_bytes(&self) -> usize {
                self.inner.len_bytes()
            }
            fn domain(&self) -> MemoryDomain {
                MemoryDomain::Host
            }
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
            fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
                self
            }
        }
        unsafe impl Send for CountingResource {}
        unsafe impl Sync for CountingResource {}
        impl Drop for CountingResource {
            fn drop(&mut self) {
                // decrement on release
                self.counter.fetch_sub(1, Ordering::SeqCst);
            }
        }

        let counter = Arc::new(AtomicUsize::new(0));

        {
            let layout = Layout::array::<u8>(64).unwrap();
            let host = HostResource::from_layout(layout).unwrap();
            counter.fetch_add(1, Ordering::SeqCst); // simulated alloc registration
            let counting = CountingResource {
                inner: host,
                counter: counter.clone(),
            };
            let owner: Box<dyn MemoryResource> = Box::new(counting);
            let ptr = unsafe { NonNull::new_unchecked(owner.as_ptr()) };

            let _storage: TensorStorage<u8, CpuAllocator> = TensorStorage {
                ptr,
                len: 64,
                owner,
                alloc: CpuAllocator,
                _marker: PhantomData,
            };
            // counter == 1 while alive
            assert_eq!(counter.load(Ordering::SeqCst), 1);
        }
        // After drop: counter == 0, exactly once
        assert_eq!(counter.load(Ordering::SeqCst), 0);
    }

    /// from_vec -> into_vec frees nothing twice (no double-free).
    /// Verify the roundtrip preserves the exact pointer and the Vec is valid after recovery.
    #[test]
    fn from_vec_into_vec_no_double_free() {
        let original = vec![1u32, 2, 3, 4, 5, 6, 7, 8];
        let original_ptr = original.as_ptr();
        let original_cap = original.capacity();

        let storage = TensorStorage::from_vec(original, CpuAllocator);
        let recovered = storage.into_vec();

        // Same pointer, same capacity (no copy, no realloc)
        assert!(std::ptr::eq(recovered.as_ptr(), original_ptr));
        assert_eq!(recovered.capacity(), original_cap);
        assert_eq!(recovered, vec![1u32, 2, 3, 4, 5, 6, 7, 8]);
        // `recovered` drops here — Vec frees the memory exactly once.
    }

    /// Foreign resource does NOT free the buffer: after storage drops, the original
    /// buffer is still valid.
    #[test]
    fn foreign_does_not_free_bytes() {
        // Use a heap-owned Vec as the "foreign" source.
        let owned_buf: Vec<u8> = vec![11, 22, 33, 44];
        let ptr = owned_buf.as_ptr();
        let len = owned_buf.len();

        // Build a ForeignResource over the Vec — no keepalive; we'll verify the Vec
        // is still alive and readable after the storage drops.
        {
            let owner: Box<dyn MemoryResource> = Box::new(unsafe {
                ForeignResource::new(
                    ptr as *mut u8,
                    len,
                    MemoryDomain::Host,
                    None, // no keepalive: storage doesn't own the Vec
                )
                .unwrap()
            });
            let storage_ptr = unsafe { NonNull::new_unchecked(ptr as *mut u8) };
            let _storage: TensorStorage<u8, ForeignAllocator> = TensorStorage {
                ptr: storage_ptr,
                len,
                owner,
                alloc: ForeignAllocator,
                _marker: PhantomData,
            };
            // While alive: same pointer, same bytes.
            assert_eq!(_storage.as_slice(), &[11, 22, 33, 44]);
        }

        // After storage drops: original Vec is still valid (ForeignResource did not free it).
        assert_eq!(owned_buf.as_slice(), &[11, 22, 33, 44]);
    }

    // ── Read-only storage tests ───────────────────────────────────────────────

    #[test]
    fn readonly_foreign_as_slice_works() {
        let buf = [1u8, 2, 3, 4];
        let keep: Arc<dyn core::any::Any + Send + Sync> = Arc::new(()); // dummy keepalive
        let s = unsafe {
            TensorStorage::<u8, ForeignAllocator>::from_borrowed_readonly(
                buf.as_ptr(),
                buf.len(),
                ForeignAllocator,
                MemoryDomain::Host,
                keep,
            )
        };
        // buf is still alive here; s.as_slice() reads from buf
        assert_eq!(s.as_slice(), &[1u8, 2, 3, 4]);
        // s drops here, buf drops after (stack order)
    }

    #[test]
    #[should_panic(expected = "read-only")]
    fn readonly_foreign_as_mut_slice_panics() {
        let buf = [1u8, 2, 3, 4];
        let keep: Arc<dyn core::any::Any + Send + Sync> = Arc::new(()); // dummy keepalive
        let mut s = unsafe {
            TensorStorage::<u8, ForeignAllocator>::from_borrowed_readonly(
                buf.as_ptr(),
                buf.len(),
                ForeignAllocator,
                MemoryDomain::Host,
                keep,
            )
        };
        let _ = s.as_mut_slice(); // must panic: read-only memory
    }

    #[test]
    fn normal_foreign_as_mut_slice_works() {
        let mut buf = [10u8, 20, 30, 40];
        let keep: Arc<dyn core::any::Any + Send + Sync> = Arc::new(42u8); // dummy keepalive
        let mut s = unsafe {
            TensorStorage::<u8, ForeignAllocator>::from_borrowed(
                buf.as_mut_ptr(),
                buf.len(),
                ForeignAllocator,
                MemoryDomain::Host,
                keep,
            )
        };
        let sl = s.as_mut_slice();
        sl[0] = 99;
        assert_eq!(sl[0], 99);
    }

    #[test]
    fn tensor_storage_send_sync_static_check() {
        fn _assert_send<T: Send>() {}
        fn _assert_sync<T: Sync>() {}
        _assert_send::<TensorStorage<u8, CpuAllocator>>();
        _assert_sync::<TensorStorage<u8, CpuAllocator>>();
        _assert_send::<TensorStorage<f32, CpuAllocator>>();
        _assert_sync::<TensorStorage<f32, CpuAllocator>>();
    }
}
