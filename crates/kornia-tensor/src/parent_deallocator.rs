use std::panic::RefUnwindSafe;

/// A trait for deallocating memory of parent for tensor.
///
/// # Safety
///
/// The parent deallocator must be thread-safe.
pub trait ParentDeallocator: Send + Sync + RefUnwindSafe {
    /// Deallocates the parent.
    fn dealloc(&self);
}
