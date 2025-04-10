/// A trait for deallocating memory of parent for tensor.
///
/// # Safety
///
/// The parent deallocator must be thread-safe.
///
/// # NOTE
///
/// Whether the parent deallocator will be used and how, will depend on
/// the [`TensorAllocator`](super::TensorAllocator). For example the
/// [`CpuAllocator`](super::CpuAllocator) will not drop the data pointer
/// if parent is present and instead call [ParentDeallocator::dealloc].
///
/// **Use this if you know what you are doing.**
///
/// Refer: [`TensorStorage`](super::TensorStorage), [PR #338](https://github.com/kornia/kornia-rs/pull/338)
/// for more info
pub trait ParentDeallocator {
    /// Deallocates the parent.
    fn dealloc(&mut self);
}

impl<T: ?Sized + ParentDeallocator> ParentDeallocator for &mut T {
    fn dealloc(&mut self) {
        (**self).dealloc();
    }
}

impl<T: ?Sized + ParentDeallocator> ParentDeallocator for Box<T> {
    fn dealloc(&mut self) {
        (**self).dealloc();
    }
}
