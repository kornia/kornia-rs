use thiserror::Error;

use super::{
    allocator::{TensorAllocator, TensorAllocatorError},
    device_marker::{Cpu, DeviceMarker},
    storage::TensorStorage,
    view::TensorView,
};

/// An error type for tensor operations.
#[derive(Error, Debug, PartialEq)]
/// Error type for tensor operations.
///
/// This enum provides detailed error information for tensor creation,
/// manipulation, and computation operations.
pub enum TensorError {
    /// Type casting operation failed.
    ///
    /// This occurs when attempting to cast a tensor to an incompatible type,
    /// typically when the conversion would result in data loss or overflow.
    ///
    /// # Examples
    /// - Casting from f64 to u8 with values outside [0, 255]
    /// - Casting from signed to unsigned with negative values
    ///
    /// # Recommended Actions
    /// - Verify data ranges are compatible with target type
    /// - Use explicit bounds checking before casting
    /// - Consider using saturating casts if appropriate
    #[error("Type cast failed: source data cannot be safely converted to target type. Check value ranges.")]
    CastError,

    /// Tensor shape does not match the provided data.
    ///
    /// This error occurs when creating a tensor with data that doesn't match
    /// the specified shape. The shape must match the number of elements exactly.
    ///
    /// # Example
    /// ```ignore
    /// // Error: shape [2, 3] expects 6 elements, but got 5
    /// let data = vec![1, 2, 3, 4, 5];
    /// let tensor = Tensor2::from_shape_vec([2, 3], data)?;
    /// ```
    ///
    /// # Recommended Actions
    /// - Verify the product of shape dimensions equals data length
    /// - Check for off-by-one errors in shape specification
    /// - Ensure data generation matches intended dimensions
    #[error("Shape mismatch: expected {expected} elements for shape, but got {actual} elements in data")]
    InvalidShape {
        /// Expected number of elements based on shape
        expected: usize,
        /// Actual number of elements in the data
        actual: usize,
    },

    /// Index exceeds tensor bounds.
    ///
    /// This occurs when attempting to access elements outside the valid
    /// index range for the tensor's dimensions.
    ///
    /// # Common Causes
    /// - Off-by-one errors in indexing loops
    /// - Incorrect dimension calculations
    /// - Using indices from a different tensor
    ///
    /// # Recommended Actions
    /// - Verify index is less than the dimension size
    /// - Use `tensor.shape` to check valid bounds
    /// - Consider using checked indexing methods
    #[error("Index {index} out of bounds for dimension of size {size}")]
    IndexOutOfBounds {
        /// The invalid index that was attempted
        index: usize,
        /// The size of the dimension being indexed
        size: usize,
    },

    /// Underlying storage operation failed.
    ///
    /// This error wraps lower-level memory allocation, deallocation,
    /// or transfer errors. See [`TensorAllocatorError`] for details.
    ///
    /// # Common Causes
    /// - Out of memory (CPU or GPU)
    /// - Invalid device operations
    /// - Memory transfer failures
    ///
    /// # Recommended Actions
    /// - Check error details in the wrapped `TensorAllocatorError`
    /// - Free unused tensors to reclaim memory
    /// - Reduce batch sizes or tensor dimensions
    #[error("Storage error: {0}")]
    StorageError(#[from] TensorAllocatorError),

    /// Tensor dimensions incompatible for the requested operation.
    ///
    /// This occurs when attempting operations that require specific
    /// dimension relationships (e.g., matrix multiplication, broadcasting).
    ///
    /// # Examples
    /// - Matrix multiplication with incompatible inner dimensions
    /// - Element-wise operations on different-shaped tensors
    /// - Concatenation along incompatible dimensions
    ///
    /// # Recommended Actions
    /// - Verify tensor shapes with `tensor.shape`
    /// - Check operation requirements in documentation
    /// - Consider reshaping or transposing tensors
    /// - Use broadcasting-compatible operations when possible
    #[error("Dimension mismatch: {message}. Expected shape: {expected}, got: {actual}")]
    DimensionMismatch {
        /// Human-readable description of the mismatch
        message: String,
        /// Expected shape description
        expected: String,
        /// Actual shape description
        actual: String,
    },

    /// Operation not supported for this tensor configuration.
    ///
    /// This indicates that the requested operation is not available for
    /// the current data type, device, or tensor configuration.
    ///
    /// # Examples
    /// - Device-specific operations on wrong device type
    /// - Type-specific operations on incompatible types
    /// - Operations requiring specific tensor properties
    ///
    /// # Recommended Actions
    /// - Check operation compatibility with tensor type/device
    /// - Transfer tensor to compatible device if needed
    /// - Cast to compatible type if appropriate
    /// - Consult API documentation for operation requirements
    #[error("Unsupported operation: {operation} - {reason}")]
    UnsupportedOperation {
        /// Name of the operation that failed
        operation: String,
        /// Reason why the operation is not supported
        reason: String,
    },
}

impl TensorError {
    /// Creates an InvalidShape error with clear context.
    pub fn invalid_shape(expected: usize, actual: usize) -> Self {
        Self::InvalidShape { expected, actual }
    }

    /// Creates an IndexOutOfBounds error with clear context.
    pub fn index_out_of_bounds(index: usize, size: usize) -> Self {
        Self::IndexOutOfBounds { index, size }
    }

    /// Creates a DimensionMismatch error with formatted shapes.
    pub fn dimension_mismatch(message: impl Into<String>, expected: &[usize], actual: &[usize]) -> Self {
        Self::DimensionMismatch {
            message: message.into(),
            expected: format!("{:?}", expected),
            actual: format!("{:?}", actual),
        }
    }

    /// Creates an UnsupportedOperation error with context.
    pub fn unsupported_operation(operation: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::UnsupportedOperation {
            operation: operation.into(),
            reason: reason.into(),
        }
    }

    /// Returns true if this error is recoverable by freeing memory.
    pub fn is_out_of_memory(&self) -> bool {
        match self {
            Self::StorageError(e) => e.is_out_of_memory(),
            _ => false,
        }
    }

    /// Returns true if this error indicates a programming error.
    pub fn is_programming_error(&self) -> bool {
        match self {
            Self::CastError
            | Self::InvalidShape { .. }
            | Self::IndexOutOfBounds { .. }
            | Self::DimensionMismatch { .. }
            | Self::UnsupportedOperation { .. } => true,
            Self::StorageError(e) => e.is_programming_error(),
        }
    }

    /// Returns a user-friendly suggestion for resolving the error.
    pub fn suggestion(&self) -> &str {
        match self {
            Self::CastError => {
                "Check that source values are within the valid range for the target type"
            }
            Self::InvalidShape { .. } => {
                "Ensure the product of shape dimensions equals the number of data elements"
            }
            Self::IndexOutOfBounds { .. } => {
                "Verify indices are within bounds (0 <= index < dimension_size)"
            }
            Self::StorageError(e) => e.suggestion(),
            Self::DimensionMismatch { .. } => {
                "Check tensor shapes are compatible for the operation. Consider reshaping or broadcasting."
            }
            Self::UnsupportedOperation { .. } => {
                "Check API documentation for operation requirements and supported configurations"
            }
        }
    }
}

/// Computes the strides for a row-major (C-contiguous) tensor layout.
///
/// Strides define how many elements to skip in memory to move along each dimension.
/// For row-major layout, the rightmost dimension has stride 1, and each dimension's
/// stride is the product of all dimensions to its right.
///
/// # Arguments
///
/// * `shape` - The shape of the tensor
///
/// # Returns
///
/// An array of strides corresponding to each dimension.
///
/// # Examples
///
/// ```rust
/// use kornia_tensor::tensor::get_strides_from_shape;
///
/// // For a 2x3 matrix: [[a, b, c], [d, e, f]]
/// let strides = get_strides_from_shape([2, 3]);
/// assert_eq!(strides, [3, 1]); // Move 3 elements for rows, 1 for columns
///
/// // For a 2x3x4 tensor
/// let strides = get_strides_from_shape([2, 3, 4]);
/// assert_eq!(strides, [12, 4, 1]); // 12 = 3*4, 4 = 4*1, 1 = 1
/// ```
pub fn get_strides_from_shape<const N: usize>(shape: [usize; N]) -> [usize; N] {
    let mut strides: [usize; N] = [0; N];
    let mut stride = 1;
    for i in (0..shape.len()).rev() {
        strides[i] = stride;
        stride *= shape[i];
    }
    strides
}

/// A multi-dimensional array (tensor) with owned data.
///
/// `Tensor` is the core data structure for storing and manipulating multi-dimensional arrays.
/// It combines data storage, shape information, and memory layout (strides) into a single,
/// type-safe structure with compile-time dimensionality checking.
///
/// # Type Parameters
///
/// * `T` - The element type stored in the tensor
/// * `N` - The number of dimensions (const generic, checked at compile time)
/// * `D` - The device marker type (e.g., `Cpu`, `Cuda<0>`)
///
/// # Architecture
///
/// The tensor consists of three main components:
///
/// * **Storage** - Owns the actual data in a contiguous memory buffer managed by a device allocator
/// * **Shape** - Defines the size of each dimension
/// * **Strides** - Describes the memory layout for element access
///
/// # Memory Layout
///
/// By default, tensors use row-major (C-contiguous) layout where the rightmost dimension
/// varies fastest in memory. The strides array defines how many elements to skip when
/// moving along each dimension.
///
/// # Thread Safety
///
/// Tensors are `Send` and `Sync` when using thread-safe allocators, allowing safe sharing
/// across thread boundaries.
///
/// # Examples
///
/// Creating a tensor from data:
///
/// ```rust
/// use kornia_tensor::{Tensor2, Cpu};
///
/// let data: Vec<u8> = vec![1, 2, 3, 4];
/// let t = Tensor2::<u8, Cpu>::from_shape_vec([2, 2], data).unwrap();
/// assert_eq!(t.shape, [2, 2]);
/// ```
pub struct Tensor<T, const N: usize, D: DeviceMarker = Cpu> {
    /// The storage of the tensor.
    pub storage: TensorStorage<T, D>,
    /// The shape of the tensor.
    pub shape: [usize; N],
    /// The strides of the tensor data in memory.
    pub strides: [usize; N],
}

impl<T, const N: usize, D: DeviceMarker> Tensor<T, N, D> {
    /// Get the data of the tensor as a slice.
    ///
    /// # Returns
    ///
    /// A slice containing the data of the tensor.
    /// Returns the tensor data as a slice.
    ///
    /// This provides safe, immutable access to the tensor's underlying data.
    /// This method is only available for CPU devices, ensuring compile-time type safety.
    ///
    /// # Returns
    ///
    /// A slice containing all elements in the tensor.
    #[inline]
    pub fn as_slice(&self) -> &[T]
    where
        D: crate::device_marker::CpuDevice,
    {
        self.storage.as_slice()
    }

    /// Get the data of the tensor as a mutable slice.
    ///
    /// This provides safe, mutable access to the tensor's underlying data.
    /// This method is only available for CPU devices, ensuring compile-time type safety.
    ///
    /// # Returns
    ///
    /// A mutable slice containing the data of the tensor.
    #[inline]
    pub fn as_slice_mut(&mut self) -> &mut [T]
    where
        D: crate::device_marker::CpuDevice,
    {
        self.storage.as_mut_slice()
    }

    /// Get the data of the tensor as a pointer.
    ///
    /// # Returns
    ///
    /// A pointer to the data of the tensor.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.storage.as_ptr()
    }

    /// Get the data of the tensor as a mutable pointer.
    ///
    /// # Returns
    ///
    /// A mutable pointer to the data of the tensor.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.storage.as_mut_ptr()
    }

    /// Consumes the tensor and returns the underlying vector.
    ///
    /// This method destroys the tensor and returns ownership of the underlying data.
    /// The returned vector will have a length equal to the total number of elements in the tensor.
    ///
    /// # Panics
    ///
    /// Panics if the tensor is not on CPU.
    #[inline]
    pub fn into_vec(self) -> Vec<T> {
        self.storage.into_vec()
    }

    /// Returns the device where the tensor is allocated.
    #[inline]
    pub fn device(&self) -> crate::device::Device {
        self.storage.device()
    }

    /// Returns true if the tensor is on CPU.
    #[inline]
    pub fn is_cpu(&self) -> bool {
        self.storage.is_cpu()
    }

    /// Returns true if the tensor is on GPU.
    #[inline]
    pub fn is_gpu(&self) -> bool {
        self.storage.is_gpu()
    }

    /// Returns an iterator over the elements of the tensor.
    ///
    /// The iterator yields immutable references to elements in memory order
    /// (following the tensor's strides). For standard-layout tensors, this
    /// is row-major order.
    ///
    /// # Panics
    ///
    /// Panics if the tensor is not on CPU.
    ///
    /// # Examples
    ///
    /// ```
    /// use kornia_tensor::{Tensor1, Cpu};
    ///
    /// let data = vec![1, 2, 3, 4, 5];
    /// let tensor = Tensor1::<i32, Cpu>::from_shape_vec([5], data).unwrap();
    ///
    /// let sum: i32 = tensor.iter().sum();
    /// assert_eq!(sum, 15);
    ///
    /// let doubled: Vec<i32> = tensor.iter().map(|&x| x * 2).collect();
    /// assert_eq!(doubled, vec![2, 4, 6, 8, 10]);
    /// ```
    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, T>
    where
        D: crate::device_marker::CpuDevice,
    {
        self.as_slice().iter()
    }

    /// Returns a mutable iterator over the elements of the tensor.
    ///
    /// The iterator yields mutable references to elements in memory order
    /// (following the tensor's strides). For standard-layout tensors, this
    /// is row-major order.
    ///
    /// # Panics
    ///
    /// Panics if the tensor is not on CPU.
    ///
    /// # Examples
    ///
    /// ```
    /// use kornia_tensor::{Tensor1, Cpu};
    ///
    /// let data = vec![1, 2, 3, 4, 5];
    /// let mut tensor = Tensor1::<i32, Cpu>::from_shape_vec([5], data).unwrap();
    ///
    /// // Double each element in place
    /// tensor.iter_mut().for_each(|x| *x *= 2);
    ///
    /// assert_eq!(tensor.as_slice(), &[2, 4, 6, 8, 10]);
    /// ```
    #[inline]
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T>
    where
        D: crate::device_marker::CpuDevice,
    {
        self.as_slice_mut().iter_mut()
    }

    /// Transfer tensor to a different device.
    ///
    /// # Arguments
    ///
    /// * `target_alloc` - The allocator for the target device
    ///
    /// # Returns
    ///
    /// A new tensor on the target device
    ///
    /// # Example
    ///
    /// ```no_run
    /// use kornia_tensor::{Tensor1, Cpu};
    /// # #[cfg(feature = "cuda")]
    /// use kornia_tensor::Cuda;
    ///
    /// # #[cfg(feature = "cuda")]
    /// # {
    /// let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    /// let cpu_tensor = Tensor1::<f32, Cpu>::from_shape_vec([4], data).unwrap();
    ///
    /// // Transfer to CUDA
    /// let cuda_tensor: Tensor1<f32, Cuda<0>> = cpu_tensor.to_device().unwrap();
    /// # }
    /// ```
    /// Transfer tensor to a different device.
    ///
    /// This method is generic over the target device type and will be specialized
    /// at compile time for each device.
    ///
    /// # Type Parameters
    ///
    /// * `Target` - The target device marker type (e.g., `Cpu`, `Cuda<0>`)
    ///
    /// # Errors
    ///
    /// Returns an error if memory allocation or transfer fails.
    pub fn to_device<Target: DeviceMarker>(&self) -> Result<Tensor<T, N, Target>, TensorError>
    where
        T: Copy,
    {
        // Get target allocator
        let target_alloc = Target::allocator()?;
        let src_device = self.storage.device();
        let _target_device = target_alloc.device();

        // Allocate on target device and copy
        use crate::allocator::BufferOps;
        
        let layout = self.storage.layout();
        let mut dst_buffer = target_alloc.alloc(layout)?;
        
        // Create source buffer wrapper for copy_from
        let src_ptr = self.storage.as_ptr() as *const u8;
        
        // Copy memory from source to destination
        // SAFETY: Both buffers are valid, lengths match, and we own the destination
        unsafe {
            target_alloc.copy_from(
                &src_ptr as &dyn BufferOps,
                &mut dst_buffer as &mut dyn BufferOps,
                self.storage.len(),
                &src_device,
            )?;
        }

        // Convert buffer to storage using the allocator's conversion method
        let buffer = target_alloc.convert_to_storage_buffer(dst_buffer, self.storage.len(), layout)
            .map_err(crate::TensorError::StorageError)?;
        
        // Create storage from the buffer
        // For all devices, we create StorageImpl directly with the buffer
        // This ensures consistent memory management across all device types
        let storage = TensorStorage::from_buffer(buffer, self.storage.len());

        Ok(Tensor {
            storage,
            shape: self.shape,
            strides: self.strides,
        })
    }

    /// Transfer tensor to CPU.
    ///
    /// This is a convenience method equivalent to `to_device(CpuAllocator)`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # #[cfg(feature = "cuda")]
    /// use kornia_tensor::{Tensor1, Cpu, Cuda};
    ///
    /// # #[cfg(feature = "cuda")]
    /// # {
    /// let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    /// let cpu_tensor = Tensor1::<f32, Cpu>::from_shape_vec([4], data).unwrap();
    /// let cuda_tensor: Tensor1<f32, Cuda<0>> = cpu_tensor.to_device().unwrap();
    ///
    /// // Transfer back to CPU
    /// let cpu_tensor_2 = cuda_tensor.to_cpu().unwrap();
    /// # }
    /// ```
    pub fn to_cpu(&self) -> Result<Tensor<T, N, Cpu>, TensorError>
    where
        T: Copy,
    {
        self.to_device::<Cpu>()
    }

    /// Creates a new `Tensor` with the given shape and data.
    ///
    /// # Arguments
    ///
    /// * `shape` - An array containing the shape of the tensor.
    /// * `data` - A vector containing the data of the tensor.
    ///
    /// # Returns
    ///
    /// A new `Tensor` instance.
    ///
    /// # Errors
    ///
    /// If the number of elements in the data does not match the shape of the tensor, an error is returned.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_tensor::{Tensor2, Cpu};
    ///
    /// let data: Vec<u8> = vec![1, 2, 3, 4];
    /// let t = Tensor2::<u8, Cpu>::from_shape_vec([2, 2], data).unwrap();
    /// assert_eq!(t.shape, [2, 2]);
    /// ```
    pub fn from_shape_vec(shape: [usize; N], data: Vec<T>) -> Result<Self, TensorError> {
        let numel = shape.iter().product::<usize>();
        if numel != data.len() {
            return Err(TensorError::invalid_shape(numel, data.len()));
        }
        let storage = TensorStorage::from_vec(data)?;
        let strides = get_strides_from_shape(shape);
        Ok(Self {
            storage,
            shape,
            strides,
        })
    }

    /// Creates a new `Tensor` with the given shape and slice of data.
    ///
    /// # Arguments
    ///
    /// * `shape` - An array containing the shape of the tensor.
    /// * `data` - A slice containing the data of the tensor.
    ///
    /// # Returns
    ///
    /// A new `Tensor` instance.
    ///
    /// # Errors
    ///
    /// If the number of elements in the data does not match the shape of the tensor, an error is returned.
    pub fn from_shape_slice(shape: [usize; N], data: &[T]) -> Result<Self, TensorError>
    where
        T: Clone,
    {
        let numel = shape.iter().product::<usize>();
        if numel != data.len() {
            return Err(TensorError::invalid_shape(numel, data.len()));
        }
        let storage = TensorStorage::from_vec(data.to_vec())?;
        let strides = get_strides_from_shape(shape);
        Ok(Self {
            storage,
            shape,
            strides,
        })
    }

    /// Creates a new `Tensor` with the given shape and raw parts.
    ///
    /// # Arguments
    ///
    /// * `shape` - An array containing the shape of the tensor.
    /// * `data` - A pointer to the data of the tensor.
    /// * `len` - The length of the data.
    ///
    /// # Safety
    ///
    /// The pointer must be non-null and the length must be valid.
    pub unsafe fn from_raw_parts(
        shape: [usize; N],
        data: *const T,
        len: usize,
    ) -> Result<Self, TensorError>
    where
        T: Clone,
    {
        let storage = TensorStorage::from_raw_parts(data, len)?;
        let strides = get_strides_from_shape(shape);
        Ok(Self {
            storage,
            shape,
            strides,
        })
    }

    /// Creates a new `Tensor` with the given shape and a default value.
    /// Creates a new `Tensor` with the given shape and a default value.
    ///
    /// # Arguments
    ///
    /// * `shape` - An array containing the shape of the tensor.
    /// * `value` - The default value to fill the tensor with.
    ///
    /// # Returns
    ///
    /// A new `Tensor` instance.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_tensor::{Tensor1, Tensor2, Tensor3, Cpu};
    ///
    /// let t = Tensor1::<u8, Cpu>::from_shape_val([4], 0).unwrap();
    /// assert_eq!(t.as_slice(), vec![0, 0, 0, 0]);
    ///
    /// let t = Tensor2::<u8, Cpu>::from_shape_val([2, 2], 1).unwrap();
    /// assert_eq!(t.as_slice(), vec![1, 1, 1, 1]);
    ///
    /// let t = Tensor3::<u8, Cpu>::from_shape_val([2, 1, 3], 2).unwrap();
    /// assert_eq!(t.as_slice(), vec![2, 2, 2, 2, 2, 2]);
    /// ```
    pub fn from_shape_val(shape: [usize; N], value: T) -> Result<Self, TensorError>
    where
        T: Clone,
    {
        let numel = shape.iter().product::<usize>();
        let data = vec![value; numel];
        let storage = TensorStorage::from_vec(data)?;
        let strides = get_strides_from_shape(shape);
        Ok(Self {
            storage,
            shape,
            strides,
        })
    }

    /// Create a new `Tensor` with the given shape and a function to generate the data.
    ///
    /// The function `f` is called with the index of the element to generate.
    ///
    /// # Arguments
    ///
    /// * `shape` - An array containing the shape of the tensor.
    /// * `f` - The function to generate the data.
    ///
    /// # Returns
    ///
    /// A new `Tensor` instance.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_tensor::{Tensor1, Tensor2, Cpu};
    ///
    /// let t = Tensor1::<u8, Cpu>::from_shape_fn([4], |[i]| i as u8).unwrap();
    /// assert_eq!(t.as_slice(), vec![0, 1, 2, 3]);
    ///
    /// let t = Tensor2::<u8, Cpu>::from_shape_fn([2, 2], |[i, j]| (i * 2 + j) as u8).unwrap();
    /// assert_eq!(t.as_slice(), vec![0, 1, 2, 3]);
    /// ```
    pub fn from_shape_fn<F>(shape: [usize; N], f: F) -> Result<Self, TensorError>
    where
        F: Fn([usize; N]) -> T,
    {
        let numel = shape.iter().product::<usize>();
        let data: Vec<T> = (0..numel)
            .map(|i| {
                let mut index = [0; N];
                let mut j = i;
                for k in (0..N).rev() {
                    index[k] = j % shape[k];
                    j /= shape[k];
                }
                f(index)
            })
            .collect();
        let storage = TensorStorage::from_vec(data)?;
        let strides = get_strides_from_shape(shape);
        Ok(Self {
            storage,
            shape,
            strides,
        })
    }

    /// Returns the number of elements in the tensor.
    ///
    /// # Returns
    ///
    /// The number of elements in the tensor.
    #[inline]
    pub fn numel(&self) -> usize {
        self.storage.len() / std::mem::size_of::<T>()
    }

    /// Get the offset of the element at the given index.
    ///
    /// # Arguments
    ///
    /// * `index` - The list of indices to get the element from.
    ///
    /// # Returns
    ///
    /// The offset of the element at the given index.
    pub fn get_iter_offset(&self, index: [usize; N]) -> Option<usize> {
        let mut offset = 0;
        for ((&idx, dim_size), stride) in index.iter().zip(self.shape).zip(self.strides) {
            if idx >= dim_size {
                return None;
            }
            offset += idx * stride;
        }
        Some(offset)
    }

    /// Get the offset of the element at the given index without checking dim sizes.
    ///
    /// # Arguments
    ///
    /// * `index` - The list of indices to get the element from.
    ///
    /// # Returns
    ///
    /// The offset of the element at the given index.
    pub fn get_iter_offset_unchecked(&self, index: [usize; N]) -> usize {
        let mut offset = 0;
        for (&idx, stride) in index.iter().zip(self.strides) {
            offset += idx * stride;
        }
        offset
    }

    /// Get the index of the element at the given offset without checking dim sizes. The reverse of `Self::get_iter_offset_unchecked`.
    ///
    /// # Arguments
    ///
    /// * `offset` - The offset of the element at the given index.
    ///
    /// # Returns
    ///
    /// The array of indices to get the element from.
    pub fn get_index_unchecked(&self, offset: usize) -> [usize; N] {
        let mut idx = [0; N];
        let mut rem = offset;
        for (dim_i, s) in self.strides.iter().enumerate() {
            idx[dim_i] = rem / s;
            rem = offset % s;
        }

        idx
    }

    /// Get the index of the element at the given offset. The reverse of `Self::get_iter_offset`.
    ///
    /// # Arguments
    ///
    /// * `offset` - The offset of the element at the given index.
    ///
    /// # Returns
    ///
    /// The array of indices to get the element from.
    ///
    /// # Errors
    ///
    /// If the offset is out of bounds (>= numel), an error is returned.
    pub fn get_index(&self, offset: usize) -> Result<[usize; N], TensorError> {
        let numel = self.numel();
        if offset >= numel {
            return Err(TensorError::index_out_of_bounds(offset, numel));
        }
        let idx = self.get_index_unchecked(offset);

        Ok(idx)
    }

    /// Get the element at the given index without checking if the index is out of bounds.
    ///
    /// # Arguments
    ///
    /// * `index` - The list of indices to get the element from.
    ///
    /// # Returns
    ///
    /// A reference to the element at the given index.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_tensor::{Tensor1, Tensor2, Tensor3, Cpu};
    ///
    /// let data: Vec<u8> = vec![1, 2, 3, 4];
    ///
    /// let t = Tensor2::<u8, Cpu>::from_shape_vec([2, 2], data).unwrap();
    /// assert_eq!(*t.get_unchecked([0, 0]), 1);
    /// assert_eq!(*t.get_unchecked([0, 1]), 2);
    /// assert_eq!(*t.get_unchecked([1, 0]), 3);
    /// assert_eq!(*t.get_unchecked([1, 1]), 4);
    /// ```
    pub fn get_unchecked(&self, index: [usize; N]) -> &T
    where
        D: crate::device_marker::CpuDevice,
    {
        let offset = self.get_iter_offset_unchecked(index);
        unsafe { self.storage.as_slice().get_unchecked(offset) }
    }

    /// Get the element at the given index, checking if the index is out of bounds.
    ///
    /// # Arguments
    ///
    /// * `index` - The list of indices to get the element from.
    ///
    /// # Returns
    ///
    /// A reference to the element at the given index.
    ///
    /// # Errors
    ///
    /// If the index is out of bounds, an error is returned.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_tensor::{Tensor1, Tensor2, Tensor3, Cpu};
    ///
    /// let data: Vec<u8> = vec![1, 2, 3, 4];
    ///
    /// let t = Tensor2::<u8, Cpu>::from_shape_vec([2, 2], data).unwrap();
    ///
    /// assert_eq!(t.get([0, 0]), Some(&1));
    /// assert_eq!(t.get([0, 1]), Some(&2));
    /// assert_eq!(t.get([1, 0]), Some(&3));
    /// assert_eq!(t.get([1, 1]), Some(&4));
    ///
    /// assert!(t.get([2, 0]).is_none());
    /// ```
    pub fn get(&self, index: [usize; N]) -> Option<&T>
    where
        D: crate::device_marker::CpuDevice,
    {
        self.get_iter_offset(index)
            .and_then(|i| self.storage.as_slice().get(i))
    }

    /// Reshape the tensor to a new shape.
    ///
    /// # Arguments
    ///
    /// * `shape` - The new shape of the tensor.
    ///
    /// # Returns
    ///
    /// A new `TensorView` instance.
    ///
    /// # Errors
    ///
    /// If the number of elements in the new shape does not match the number of elements in the tensor, an error is returned.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_tensor::{Tensor1, Tensor2, Tensor3, Cpu};
    ///
    /// let data: Vec<u8> = vec![1, 2, 3, 4];
    ///
    /// let t = Tensor1::<u8, Cpu>::from_shape_vec([4], data).unwrap();
    /// let t2 = t.reshape([2, 2]).unwrap();
    /// assert_eq!(t2.shape, [2, 2]);
    /// assert_eq!(t2.as_slice(), vec![1, 2, 3, 4]);
    /// assert_eq!(t2.strides, [2, 1]);
    /// assert_eq!(t2.numel(), 4);
    /// ```
    pub fn reshape<const M: usize>(
        &self,
        shape: [usize; M],
    ) -> Result<TensorView<'_, T, M, D>, TensorError> {
        let numel = shape.iter().product::<usize>();
        if numel != self.numel() {
            return Err(TensorError::DimensionMismatch {
                message: "Reshape operation requires same number of elements".to_string(),
                expected: format!("{:?} ({} elements)", shape, numel),
                actual: format!("{:?} ({} elements)", self.shape, self.numel()),
            });
        }

        let strides = get_strides_from_shape(shape);

        Ok(TensorView {
            storage: &self.storage,
            shape,
            strides,
        })
    }

    /// Permutes (reorders) the dimensions of the tensor.
    ///
    /// This is a zero-copy operation that returns a view with reordered dimensions.
    /// The underlying data is not moved; only the shape and strides are adjusted.
    /// Common use cases include matrix transposition and changing data layout.
    ///
    /// # Arguments
    ///
    /// * `axes` - An array specifying the new dimension order. `axes[i]` indicates which
    ///   source dimension becomes the i-th dimension in the output.
    ///
    /// # Returns
    ///
    /// A view of the tensor with the dimensions permuted.
    pub fn permute_axes(&self, axes: [usize; N]) -> TensorView<'_, T, N, D> {
        let mut new_shape = [0; N];
        let mut new_strides = [0; N];
        for (i, &axis) in axes.iter().enumerate() {
            new_shape[i] = self.shape[axis];
            new_strides[i] = self.strides[axis];
        }

        TensorView {
            storage: &self.storage,
            shape: new_shape,
            strides: new_strides,
        }
    }

    /// Return a view of the tensor.
    ///
    /// The view is a reference to the tensor storage with a different shape and strides.
    ///
    /// # Returns
    ///
    /// A `TensorView` instance.
    pub fn view(&self) -> TensorView<'_, T, N, D> {
        TensorView {
            storage: &self.storage,
            shape: self.shape,
            strides: self.strides,
        }
    }

    /// Create a new tensor with all elements set to zero.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    ///
    /// # Returns
    ///
    /// A new tensor filled with zeros.
    ///
    /// # Errors
    ///
    /// Returns an error if memory allocation fails.
    pub fn zeros(shape: [usize; N]) -> Result<Tensor<T, N, D>, TensorError>
    where
        T: Clone + num_traits::Zero,
    {
        Self::from_shape_val(shape, T::zero())
    }

    /// Apply a function to each element of the tensor.
    ///
    /// # Arguments
    ///
    /// * `f` - The function to apply to each element.
    ///
    /// # Returns
    ///
    /// A new `Tensor` instance.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_tensor::{Tensor1, Cpu};
    ///
    /// let data: Vec<u8> = vec![1, 2, 3, 4];
    /// let t = Tensor1::<u8, Cpu>::from_shape_vec([4], data).unwrap();
    ///
    /// let t2 = t.map(|x| *x + 1).unwrap();
    /// assert_eq!(t2.as_slice(), vec![2, 3, 4, 5]);
    /// ```
    pub fn map<U, F>(&self, f: F) -> Result<Tensor<U, N, D>, TensorError>
    where
        D: crate::device_marker::CpuDevice,
        F: Fn(&T) -> U,
    {
        let data: Vec<U> = self.as_slice().iter().map(f).collect();
        let storage = TensorStorage::from_vec(data)?;

        Ok(Tensor {
            storage,
            shape: self.shape,
            strides: self.strides,
        })
    }

    /// Checks if the tensor has a standard contiguous (row-major) memory layout.
    ///
    /// A standard layout means the tensor's data is stored contiguously in memory
    /// following row-major order, where the rightmost dimension varies fastest.
    /// This is important for performance in many operations and for interoperability
    /// with other libraries.
    ///
    /// # Returns
    ///
    /// `true` if the tensor has contiguous row-major layout, `false` otherwise.
    ///
    /// # When is a layout non-standard?
    ///
    /// - After dimension permutation (e.g., transpose)
    /// - After certain slicing operations
    /// - When strides have been manually modified
    /// - When the tensor is a non-contiguous view
    ///
    /// # Examples
    ///
    /// ```
    /// use kornia_tensor::{Tensor1, Tensor2, Tensor3, Cpu};
    /// let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    /// let mut t = Tensor3::<u8, Cpu>::from_shape_vec([2, 2, 3], data).unwrap();
    /// // arbitrary incorrect stride
    /// t.strides = [10, 5, 1];
    /// assert!(!t.is_standard_layout());
    /// ```
    pub fn is_standard_layout(&self) -> bool {
        let mut expected_stride: usize = 1;
        for (&dim, &stride) in self.shape.iter().rev().zip(self.strides.iter().rev()) {
            if stride != expected_stride {
                return false;
            }
            expected_stride = expected_stride.saturating_mul(dim);
        }
        true
    }

    /// Converts the tensor to standard contiguous (row-major) memory layout.
    ///
    /// If the tensor already has standard layout, returns a clone. Otherwise, creates a new
    /// tensor with contiguous memory by copying elements in the correct order according to
    /// the current shape and strides.
    ///
    /// This is essential for:
    /// - Interfacing with libraries expecting contiguous memory
    /// - Optimizing performance of certain operations
    /// - Ensuring consistent memory layout after transformations
    ///
    /// # Arguments
    ///
    /// * `alloc` - The allocator to use for the new tensor if reallocation is needed
    ///
    /// # Returns
    ///
    /// A new tensor with standard layout on success.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::DimensionMismatch`] if the tensor's shape and data are inconsistent.
    ///
    /// # Performance
    ///
    /// This operation is O(1) if already standard layout (just clones), O(n) otherwise where
    /// n is the number of elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use kornia_tensor::{Tensor1, Tensor2, Tensor3, Cpu};
    /// let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    /// let mut t = Tensor3::<u8, Cpu>::from_shape_vec([2, 2, 3], data.clone()).unwrap();
    /// // altering strides
    /// t.strides = [1, 6, 2];
    /// assert!(!t.is_standard_layout());
    /// match t.to_standard_layout() {
    ///     Ok(t2) => {
    ///         assert!(t2.is_standard_layout());
    ///     }
    ///     Err(e) => {
    ///         eprintln!("to_standard_layout failed: {}", e);
    ///     }
    /// }
    /// ```
    pub fn to_standard_layout(&self) -> Result<Self, TensorError>
    where
        D: crate::device_marker::CpuDevice,
        T: Clone + std::fmt::Debug,
    {
        if self.is_standard_layout() {
            return Ok(self.clone());
        }

        let total_elems: usize = self.shape.iter().product();
        let mut flat = Vec::with_capacity(total_elems);
        let mut idx = [0; N];
        let slice = self.storage.as_slice();

        for _ in 0..total_elems {
            let offset = idx
                .iter()
                .zip(self.strides.iter())
                .map(|(&i, &s)| i * s)
                .sum::<usize>();

            flat.push(slice[offset].clone());

            // increment index
            for dim in (0..N).rev() {
                idx[dim] += 1;
                if idx[dim] < self.shape[dim] {
                    break;
                } else {
                    idx[dim] = 0;
                }
            }
        }

        Tensor::from_shape_vec(self.shape, flat).map_err(|e| {
            // If it's already a shape error, return it; otherwise wrap in dimension mismatch
            e
        })
    }

    /// Cast the tensor to a new type.
    ///
    /// # Returns
    ///
    /// A new `Tensor` instance.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_tensor::{Tensor1, Cpu};
    ///
    /// let data: Vec<u8> = vec![1, 2, 3, 4];
    /// let t = Tensor1::<u8, Cpu>::from_shape_vec([4], data).unwrap();
    ///
    /// let t2 = t.cast::<f32>().unwrap();
    /// assert_eq!(t2.as_slice(), vec![1.0, 2.0, 3.0, 4.0]);
    /// ```
    pub fn cast<U>(&self) -> Result<Tensor<U, N, Cpu>, TensorError>
    where
        D: crate::device_marker::CpuDevice,
        U: From<T>,
        T: Clone,
    {
        let mut data: Vec<U> = Vec::with_capacity(self.storage.len());
        self.as_slice().iter().for_each(|x| {
            data.push(U::from(x.clone()));
        });
        let storage = TensorStorage::from_vec(data)?;
        Ok(Tensor {
            storage,
            shape: self.shape,
            strides: self.strides,
        })
    }

    /// Perform an element-wise operation on two tensors.
    ///
    /// NOTE: This operation is currently CPU-only. If the tensors are on GPU,
    /// use `.to_cpu()` first to transfer to CPU.
    ///
    /// # Arguments
    ///
    /// * `other` - The other tensor to perform the operation with.
    /// * `op` - The operation to perform.
    ///
    /// # Returns
    ///
    /// A new `Tensor` instance.
    ///
    /// # Panics
    ///
    /// Panics if either tensor is not on CPU.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_tensor::{Tensor1, Cpu};
    ///
    /// let data1: Vec<u8> = vec![1, 2, 3, 4];
    /// let t1 = Tensor1::<u8, Cpu>::from_shape_vec([4], data1).unwrap();
    ///
    /// let data2: Vec<u8> = vec![1, 2, 3, 4];
    /// let t2 = Tensor1::<u8, Cpu>::from_shape_vec([4], data2).unwrap();
    ///
    /// let t3 = t1.element_wise_op(&t2, |a, b| *a + *b).unwrap();
    /// assert_eq!(t3.as_slice(), vec![2, 4, 6, 8]);
    ///
    /// let t4 = t1.element_wise_op(&t2, |a, b| *a - *b).unwrap();
    /// assert_eq!(t4.as_slice(), vec![0, 0, 0, 0]);
    ///
    /// let t5 = t1.element_wise_op(&t2, |a, b| *a * *b).unwrap();
    /// assert_eq!(t5.as_slice(), vec![1, 4, 9, 16]);
    ///
    /// let t6 = t1.element_wise_op(&t2, |a, b| *a / *b).unwrap();
    /// assert_eq!(t6.as_slice(), vec![1, 1, 1, 1]);
    /// ```
    pub fn element_wise_op<F>(
        &self,
        other: &Tensor<T, N, Cpu>,
        op: F,
    ) -> Result<Tensor<T, N, Cpu>, TensorError>
    where
        D: crate::device_marker::CpuDevice,
        F: Fn(&T, &T) -> T,
    {
        if self.shape != other.shape {
            return Err(TensorError::DimensionMismatch {
                message: "Element-wise operations require identical shapes".to_string(),
                expected: format!("{:?}", self.shape),
                actual: format!("{:?}", other.shape),
            });
        }

        let data = self
            .as_slice()
            .iter()
            .zip(other.as_slice().iter())
            .map(|(a, b)| op(a, b))
            .collect();

        let storage = TensorStorage::from_vec(data)?;

        Ok(Tensor {
            storage,
            shape: self.shape,
            strides: self.strides,
        })
    }
}

impl<T, const N: usize, D: DeviceMarker> Clone for Tensor<T, N, D>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(),
            shape: self.shape,
            strides: self.strides,
        }
    }
}

impl<T, const N: usize, D> std::fmt::Display for Tensor<T, N, D>
where
    D: DeviceMarker + crate::device_marker::CpuDevice,
    T: std::fmt::Display + std::fmt::LowerExp,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let width = self
            .storage
            .as_slice()
            .iter()
            .map(|v| format!("{v:.4}").len())
            .max()
            .unwrap();

        let scientific = width > 8;

        let should_mask: [bool; N] = self.shape.map(|s| s > 8);
        let mut skip_until = 0;

        for (i, v) in self.storage.as_slice().iter().enumerate() {
            if i < skip_until {
                continue;
            }
            let mut value = String::new();
            let mut prefix = String::new();
            let mut suffix = String::new();
            let mut separator = ",".to_string();
            let mut last_size = 1;
            for (dim, (&size, maskable)) in self.shape.iter().zip(should_mask).enumerate().rev() {
                let prod = size * last_size;
                if i % prod == (3 * last_size) && maskable {
                    let pad = if dim == (N - 1) { 0 } else { dim + 1 };
                    value = format!("{}...", " ".repeat(pad));
                    skip_until = i + (size - 4) * last_size;
                    prefix = "".to_string();
                    if dim != (N - 1) {
                        separator = "\n".repeat(N - 1 - dim);
                    }
                    break;
                } else if i % prod == 0 {
                    prefix.push('[');
                } else if (i + 1) % prod == 0 {
                    suffix.push(']');
                    separator.push('\n');
                    if dim == 0 {
                        separator = "".to_string();
                    }
                } else {
                    break;
                }
                last_size = prod;
            }
            if !prefix.is_empty() {
                prefix = format!("{prefix:>N$}");
            }

            if value.is_empty() {
                value = if scientific {
                    let num = format!("{v:.4e}");
                    let (before, after) = num.split_once('e').unwrap();
                    let after = if let Some(stripped) = after.strip_prefix('-') {
                        format!("-{:0>2}", &stripped)
                    } else {
                        format!("+{:0>2}", &after)
                    };
                    format!("{before}e{after}")
                } else {
                    let rounded = format!("{v:.4}");
                    format!("{rounded:>width$}")
                }
            };
            write!(f, "{prefix}{value}{suffix}{separator}",)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::device_marker::Cpu;
    use crate::tensor::TensorError;
    use crate::{Tensor1, Tensor2, Tensor3};

    #[test]
    fn constructor_1d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1];
        let t = Tensor1::<u8, Cpu>::from_shape_vec([1], data)?;
        assert_eq!(t.shape, [1]);
        assert_eq!(t.as_slice(), vec![1]);
        assert_eq!(t.strides, [1]);
        assert_eq!(t.numel(), 1);
        Ok(())
    }

    #[test]
    fn constructor_2d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2];
        let t = Tensor2::<u8, Cpu>::from_shape_vec([1, 2], data)?;
        assert_eq!(t.shape, [1, 2]);
        assert_eq!(t.as_slice(), vec![1, 2]);
        assert_eq!(t.strides, [2, 1]);
        assert_eq!(t.numel(), 2);
        Ok(())
    }

    #[test]
    fn get_1d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor1::<u8, Cpu>::from_shape_vec([4], data)?;
        assert_eq!(t.get([0]), Some(&1));
        assert_eq!(t.get([1]), Some(&2));
        assert_eq!(t.get([2]), Some(&3));
        assert_eq!(t.get([3]), Some(&4));
        assert!(t.get([4]).is_none());
        Ok(())
    }

    #[test]
    fn get_2d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor2::<u8, Cpu>::from_shape_vec([2, 2], data)?;
        assert_eq!(t.get([0, 0]), Some(&1));
        assert_eq!(t.get([0, 1]), Some(&2));
        assert_eq!(t.get([1, 0]), Some(&3));
        assert_eq!(t.get([1, 1]), Some(&4));
        assert!(t.get([2, 0]).is_none());
        assert!(t.get([0, 2]).is_none());
        Ok(())
    }

    #[test]
    fn get_3d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6];
        let t = Tensor3::<u8, Cpu>::from_shape_vec([2, 1, 3], data)?;
        assert_eq!(t.get([0, 0, 0]), Some(&1));
        assert_eq!(t.get([0, 0, 1]), Some(&2));
        assert_eq!(t.get([0, 0, 2]), Some(&3));
        assert_eq!(t.get([1, 0, 0]), Some(&4));
        assert_eq!(t.get([1, 0, 1]), Some(&5));
        assert_eq!(t.get([1, 0, 2]), Some(&6));
        assert!(t.get([2, 0, 0]).is_none());
        assert!(t.get([0, 1, 0]).is_none());
        assert!(t.get([0, 0, 3]).is_none());
        Ok(())
    }

    #[test]
    fn get_checked_1d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor1::<u8, Cpu>::from_shape_vec([4], data)?;
        assert_eq!(*t.get_unchecked([0]), 1);
        assert_eq!(*t.get_unchecked([1]), 2);
        assert_eq!(*t.get_unchecked([2]), 3);
        assert_eq!(*t.get_unchecked([3]), 4);
        Ok(())
    }

    #[test]
    fn get_checked_2d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor2::<u8, Cpu>::from_shape_vec([2, 2], data)?;
        assert_eq!(*t.get_unchecked([0, 0]), 1);
        assert_eq!(*t.get_unchecked([0, 1]), 2);
        assert_eq!(*t.get_unchecked([1, 0]), 3);
        assert_eq!(*t.get_unchecked([1, 1]), 4);
        Ok(())
    }
    #[test]
    fn reshape_1d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor1::<u8, Cpu>::from_shape_vec([4], data)?;

        let view = t.reshape([2, 2])?;

        assert_eq!(view.shape, [2, 2]);
        assert_eq!(view.as_slice(), vec![1, 2, 3, 4]);
        assert_eq!(view.strides, [2, 1]);
        assert_eq!(view.numel(), 4);
        assert_eq!(view.as_contiguous()?.as_slice(), vec![1, 2, 3, 4]);
        Ok(())
    }

    #[test]
    fn reshape_2d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor2::<u8, Cpu>::from_shape_vec([2, 2], data)?;
        let t2 = t.reshape([4])?;

        assert_eq!(t2.shape, [4]);
        assert_eq!(t2.as_slice(), vec![1, 2, 3, 4]);
        assert_eq!(t2.strides, [1]);
        assert_eq!(t2.numel(), 4);
        assert_eq!(t2.as_contiguous()?.as_slice(), vec![1, 2, 3, 4]);
        Ok(())
    }

    #[test]
    fn reshape_get_1d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor1::<u8, Cpu>::from_shape_vec([4], data)?;
        let view = t.reshape([2, 2])?;
        assert_eq!(*view.get_unchecked([0, 0]), 1);
        assert_eq!(*view.get_unchecked([0, 1]), 2);
        assert_eq!(*view.get_unchecked([1, 0]), 3);
        assert_eq!(*view.get_unchecked([1, 1]), 4);
        assert_eq!(view.numel(), 4);
        assert_eq!(view.as_contiguous()?.as_slice(), vec![1, 2, 3, 4]);
        Ok(())
    }

    #[test]
    fn permute_axes_1d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor1::<u8, Cpu>::from_shape_vec([4], data)?;
        let t2 = t.permute_axes([0]);
        assert_eq!(t2.shape, [4]);
        assert_eq!(t2.as_slice(), vec![1, 2, 3, 4]);
        assert_eq!(t2.strides, [1]);
        assert_eq!(t2.as_contiguous()?.as_slice(), vec![1, 2, 3, 4]);
        Ok(())
    }

    #[test]
    fn permute_axes_2d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor2::<u8, Cpu>::from_shape_vec([2, 2], data)?;
        let view = t.permute_axes([1, 0]);
        assert_eq!(view.shape, [2, 2]);
        assert_eq!(*view.get_unchecked([0, 0]), 1u8);
        assert_eq!(*view.get_unchecked([1, 0]), 2u8);
        assert_eq!(*view.get_unchecked([0, 1]), 3u8);
        assert_eq!(*view.get_unchecked([1, 1]), 4u8);
        assert_eq!(view.strides, [1, 2]);
        assert_eq!(view.as_contiguous()?.as_slice(), vec![1, 3, 2, 4]);
        Ok(())
    }

    #[test]
    fn contiguous_2d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6];
        let t = Tensor2::<u8, Cpu>::from_shape_vec([2, 3], data)?;

        let view = t.permute_axes([1, 0]);

        let contiguous = view.as_contiguous()?;

        assert_eq!(contiguous.shape, [3, 2]);
        assert_eq!(contiguous.strides, [2, 1]);
        assert_eq!(contiguous.as_slice(), vec![1, 4, 2, 5, 3, 6]);

        Ok(())
    }

    #[test]
    fn zeros_1d() -> Result<(), TensorError> {
        let t = Tensor1::<u8, Cpu>::zeros([4])?;
        assert_eq!(t.as_slice(), vec![0, 0, 0, 0]);
        Ok(())
    }

    #[test]
    fn zeros_2d() -> Result<(), TensorError> {
        let t = Tensor2::<u8, Cpu>::zeros([2, 2])?;
        assert_eq!(t.as_slice(), vec![0, 0, 0, 0]);
        Ok(())
    }

    #[test]
    fn map_1d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor1::<u8, Cpu>::from_shape_vec([4], data)?;
        let t2 = t.map(|x| *x + 1)?;
        assert_eq!(t2.as_slice(), vec![2, 3, 4, 5]);
        Ok(())
    }

    #[test]
    fn map_2d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor2::<u8, Cpu>::from_shape_vec([2, 2], data)?;
        let t2 = t.map(|x| *x + 1)?;
        assert_eq!(t2.as_slice(), vec![2, 3, 4, 5]);
        Ok(())
    }

    #[test]
    fn from_shape_val_1d() -> Result<(), TensorError> {
        let t = Tensor1::<u8, Cpu>::from_shape_val([4], 0)?;
        assert_eq!(t.as_slice(), vec![0, 0, 0, 0]);
        Ok(())
    }

    #[test]
    fn from_shape_val_2d() -> Result<(), TensorError> {
        let t = Tensor2::<u8, Cpu>::from_shape_val([2, 2], 1)?;
        assert_eq!(t.as_slice(), vec![1, 1, 1, 1]);
        Ok(())
    }

    #[test]
    fn from_shape_val_3d() -> Result<(), TensorError> {
        let t = Tensor3::<u8, Cpu>::from_shape_val([2, 1, 3], 2)?;
        assert_eq!(t.as_slice(), vec![2, 2, 2, 2, 2, 2]);
        Ok(())
    }

    #[test]
    fn cast_1d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor1::<u8, Cpu>::from_shape_vec([4], data)?;
        let t2 = t.cast::<u16>()?;
        assert_eq!(t2.as_slice(), vec![1, 2, 3, 4]);
        Ok(())
    }

    #[test]
    fn cast_2d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor2::<u8, Cpu>::from_shape_vec([2, 2], data)?;
        let t2 = t.cast::<u16>()?;
        assert_eq!(t2.as_slice(), vec![1, 2, 3, 4]);
        Ok(())
    }

    #[test]
    fn from_shape_fn_1d() -> Result<(), TensorError> {
        let t = Tensor2::<u8, Cpu>::from_shape_fn([3, 3], |[i, j]| ((1 + i) * (1 + j)) as u8)?;
        assert_eq!(t.as_slice(), vec![1, 2, 3, 2, 4, 6, 3, 6, 9]);
        Ok(())
    }

    #[test]
    fn from_shape_fn_2d() -> Result<(), TensorError> {
        let t = Tensor2::<f32, Cpu>::from_shape_fn([3, 3], |[i, j]| ((1 + i) * (1 + j)) as f32)?;
        assert_eq!(
            t.as_slice(),
            vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0]
        );
        Ok(())
    }

    #[test]
    fn from_shape_fn_3d() -> Result<(), TensorError> {
        let t = Tensor3::<i16, Cpu>::from_shape_fn([2, 3, 3], |[x, y, c]| {
            ((1 + x) * (1 + y) * (1 + c)) as i16
        })?;
        assert_eq!(
            t.as_slice(),
            vec![1, 2, 3, 2, 4, 6, 3, 6, 9, 2, 4, 6, 4, 8, 12, 6, 12, 18]
        );
        Ok(())
    }

    #[test]
    fn view_1d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor1::<u8, Cpu>::from_shape_vec([4], data)?;
        let view = t.view();

        // check that the view has the same data
        assert_eq!(view.as_slice(), t.as_slice());

        // check that the data pointer is the same
        assert!(std::ptr::eq(view.as_ptr(), t.as_ptr()));

        Ok(())
    }

    #[test]
    fn from_slice() -> Result<(), TensorError> {
        let data: [u8; 4] = [1, 2, 3, 4];
        let t = Tensor2::<u8, Cpu>::from_shape_slice([2, 2], &data)?;

        assert_eq!(t.shape, [2, 2]);
        assert_eq!(t.as_slice(), &[1, 2, 3, 4]);

        Ok(())
    }

    #[test]
    fn display_2d() -> Result<(), TensorError> {
        let data: [u8; 4] = [1, 2, 3, 4];
        let t = Tensor2::<u8, Cpu>::from_shape_slice([2, 2], &data)?;
        let disp = t.to_string();
        let lines = disp.lines().collect::<Vec<_>>();

        #[rustfmt::skip]
        assert_eq!(lines.as_slice(),
        ["[[1,2],",
         " [3,4]]"]);
        Ok(())
    }

    #[test]
    fn display_3d() -> Result<(), TensorError> {
        let data: [u8; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let t = Tensor3::<u8, Cpu>::from_shape_slice([2, 3, 2], &data)?;
        let disp = t.to_string();
        let lines = disp.lines().collect::<Vec<_>>();

        #[rustfmt::skip]
        assert_eq!(lines.as_slice(),
        ["[[[ 1, 2],",
         "  [ 3, 4],",
         "  [ 5, 6]],",
         "",
         " [[ 7, 8],",
         "  [ 9,10],",
         "  [11,12]]]"]);
        Ok(())
    }

    #[test]
    fn display_float() -> Result<(), TensorError> {
        let data: [f32; 4] = [1.00001, 1.00009, 0.99991, 0.99999];
        let t = Tensor2::<f32, Cpu>::from_shape_slice([2, 2], &data)?;
        let disp = t.to_string();
        let lines = disp.lines().collect::<Vec<_>>();

        #[rustfmt::skip]
        assert_eq!(lines.as_slice(),
        ["[[1.0000,1.0001],",
         " [0.9999,1.0000]]"]);
        Ok(())
    }

    #[test]
    fn display_big_float() -> Result<(), TensorError> {
        let data: [f32; 4] = [1000.00001, 1.00009, 0.99991, 0.99999];
        let t = Tensor2::<f32, Cpu>::from_shape_slice([2, 2], &data)?;
        let disp = t.to_string();
        let lines = disp.lines().collect::<Vec<_>>();

        #[rustfmt::skip]
        assert_eq!(lines.as_slice(),
        ["[[1.0000e+03,1.0001e+00],",
         " [9.9991e-01,9.9999e-01]]"]);
        Ok(())
    }

    #[test]
    fn display_big_tensor() -> Result<(), TensorError> {
        let data: [u8; 1000] = [0; 1000];
        let t = Tensor3::<u8, Cpu>::from_shape_slice([10, 10, 10], &data)?;
        let disp = t.to_string();
        let lines = disp.lines().collect::<Vec<_>>();

        #[rustfmt::skip]
        assert_eq!(lines.as_slice(),
        ["[[[0,0,0,...,0],",
         "  [0,0,0,...,0],",
         "  [0,0,0,...,0],",
         "  ...",
         "  [0,0,0,...,0]],",
         "",
         " [[0,0,0,...,0],",
         "  [0,0,0,...,0],",
         "  [0,0,0,...,0],",
         "  ...",
         "  [0,0,0,...,0]],",
         "",
         " [[0,0,0,...,0],",
         "  [0,0,0,...,0],",
         "  [0,0,0,...,0],",
         "  ...",
         "  [0,0,0,...,0]],",
         "",
         " ...",
         "",
         " [[0,0,0,...,0],",
         "  [0,0,0,...,0],",
         "  [0,0,0,...,0],",
         "  ...",
         "  [0,0,0,...,0]]]"]);
        Ok(())
    }

    #[test]
    fn get_index_unchecked_1d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor1::<u8, Cpu>::from_shape_vec([4], data)?;
        assert_eq!(t.get_index_unchecked(0), [0]);
        assert_eq!(t.get_index_unchecked(1), [1]);
        assert_eq!(t.get_index_unchecked(2), [2]);
        assert_eq!(t.get_index_unchecked(3), [3]);
        Ok(())
    }

    #[test]
    fn get_index_unchecked_2d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor2::<u8, Cpu>::from_shape_vec([2, 2], data)?;
        assert_eq!(t.get_index_unchecked(0), [0, 0]);
        assert_eq!(t.get_index_unchecked(1), [0, 1]);
        assert_eq!(t.get_index_unchecked(2), [1, 0]);
        assert_eq!(t.get_index_unchecked(3), [1, 1]);
        Ok(())
    }

    #[test]
    fn get_index_unchecked_3d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let t = Tensor3::<u8, Cpu>::from_shape_vec([2, 2, 3], data)?;
        assert_eq!(t.get_index_unchecked(0), [0, 0, 0]);
        assert_eq!(t.get_index_unchecked(1), [0, 0, 1]);
        assert_eq!(t.get_index_unchecked(2), [0, 0, 2]);
        assert_eq!(t.get_index_unchecked(3), [0, 1, 0]);
        assert_eq!(t.get_index_unchecked(4), [0, 1, 1]);
        assert_eq!(t.get_index_unchecked(5), [0, 1, 2]);
        assert_eq!(t.get_index_unchecked(6), [1, 0, 0]);
        assert_eq!(t.get_index_unchecked(7), [1, 0, 1]);
        assert_eq!(t.get_index_unchecked(8), [1, 0, 2]);
        assert_eq!(t.get_index_unchecked(9), [1, 1, 0]);
        assert_eq!(t.get_index_unchecked(10), [1, 1, 1]);
        assert_eq!(t.get_index_unchecked(11), [1, 1, 2]);
        Ok(())
    }

    #[test]
    fn get_index_to_offset_and_back() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let t = Tensor3::<u8, Cpu>::from_shape_vec([2, 2, 3], data)?;
        for offset in 0..12 {
            assert_eq!(
                t.get_iter_offset_unchecked(t.get_index_unchecked(offset)),
                offset
            );
        }
        Ok(())
    }

    #[test]
    fn get_offset_to_index_and_back() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let t = Tensor3::<u8, Cpu>::from_shape_vec([2, 2, 3], data)?;
        for ind in [
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 2],
            [0, 1, 0],
            [0, 1, 1],
            [0, 1, 2],
            [1, 0, 0],
            [1, 0, 1],
            [1, 0, 2],
            [1, 1, 0],
            [1, 1, 1],
            [1, 1, 2],
        ] {
            assert_eq!(t.get_index_unchecked(t.get_iter_offset_unchecked(ind)), ind);
        }
        Ok(())
    }

    #[test]
    fn get_index_1d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor1::<u8, Cpu>::from_shape_vec([4], data)?;
        assert_eq!(t.get_index(3), Ok([3]));
        assert!(t
            .get_index(4)
            .is_err_and(|x| matches!(x, TensorError::IndexOutOfBounds { index: 4, size: 4 })));
        Ok(())
    }

    #[test]
    fn get_index_2d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor2::<u8, Cpu>::from_shape_vec([2, 2], data)?;
        assert_eq!(t.get_index_unchecked(3), [1, 1]);
        assert!(t
            .get_index(4)
            .is_err_and(|x| matches!(x, TensorError::IndexOutOfBounds { index: 4, size: 4 })));
        Ok(())
    }

    #[test]
    fn get_index_3d() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let t = Tensor3::<u8, Cpu>::from_shape_vec([2, 2, 3], data)?;
        assert_eq!(t.get_index_unchecked(11), [1, 1, 2]);
        assert!(t
            .get_index(12)
            .is_err_and(|x| matches!(x, TensorError::IndexOutOfBounds { index: 12, size: 12 })));
        Ok(())
    }

    #[test]
    fn from_raw_parts() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = unsafe { Tensor2::<u8, Cpu>::from_raw_parts([2, 2], data.as_ptr(), data.len())? };
        std::mem::forget(data);
        assert_eq!(t.shape, [2, 2]);
        assert_eq!(t.as_slice(), &[1, 2, 3, 4]);
        Ok(())
    }

    #[test]
    fn contiguous_tensor_is_standard_layout_true() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let t = Tensor3::<u8, Cpu>::from_shape_vec([2, 2, 3], data)?;
        assert!(t.is_standard_layout());
        Ok(())
    }

    #[test]
    fn broken_stride_is_standard_layout_false() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let mut t = Tensor3::<u8, Cpu>::from_shape_vec([2, 2, 3], data)?;
        // arbitrary incorrect stride
        t.strides = [10, 5, 1];
        assert!(!t.is_standard_layout());
        Ok(())
    }

    #[test]
    fn contiguous_tensor_roundtrip() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let t =
            Tensor3::<u8, Cpu>::from_shape_vec([2, 2, 3], data.clone())?;
        assert!(t.is_standard_layout());
        let t2 = t.to_standard_layout()?;
        assert!(t2.is_standard_layout());
        assert_eq!(t2.storage.as_slice(), data.as_slice());
        Ok(())
    }

    #[test]
    fn non_contiguous_to_standard_layout() -> Result<(), TensorError> {
        let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let mut t =
            Tensor3::<u8, Cpu>::from_shape_vec([2, 2, 3], data.clone())?;
        // altering strides
        t.strides = [1, 6, 2];
        assert!(!t.is_standard_layout());
        let t2 = t.to_standard_layout()?;
        assert!(t2.is_standard_layout());
        Ok(())
    }

    #[test]
    fn test_iter() -> Result<(), TensorError> {
        let data = vec![1, 2, 3, 4, 5];
        let tensor = Tensor1::<i32, Cpu>::from_shape_vec([5], data)?;
        
        // Test sum via iterator
        let sum: i32 = tensor.iter().sum();
        assert_eq!(sum, 15);
        
        // Test map
        let doubled: Vec<i32> = tensor.iter().map(|&x| x * 2).collect();
        assert_eq!(doubled, vec![2, 4, 6, 8, 10]);
        
        // Test filter
        let evens: Vec<i32> = tensor.iter().filter(|&&x| x % 2 == 0).copied().collect();
        assert_eq!(evens, vec![2, 4]);
        
        Ok(())
    }

    #[test]
    fn test_iter_mut() -> Result<(), TensorError> {
        let data = vec![1, 2, 3, 4, 5];
        let mut tensor = Tensor1::<i32, Cpu>::from_shape_vec([5], data)?;
        
        // Double each element
        tensor.iter_mut().for_each(|x| *x *= 2);
        assert_eq!(tensor.as_slice(), &[2, 4, 6, 8, 10]);
        
        // Add 1 to each element
        tensor.iter_mut().for_each(|x| *x += 1);
        assert_eq!(tensor.as_slice(), &[3, 5, 7, 9, 11]);
        
        Ok(())
    }

    #[test]
    fn test_iter_2d() -> Result<(), TensorError> {
        let data = vec![1, 2, 3, 4, 5, 6];
        let tensor = Tensor2::<i32, Cpu>::from_shape_vec([2, 3], data)?;
        
        // Collect all elements
        let elements: Vec<i32> = tensor.iter().copied().collect();
        assert_eq!(elements, vec![1, 2, 3, 4, 5, 6]);
        
        // Count elements greater than 3
        let count = tensor.iter().filter(|&&x| x > 3).count();
        assert_eq!(count, 3);
        
        Ok(())
    }

    #[test]
    fn test_iter_chaining() -> Result<(), TensorError> {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let tensor = Tensor1::<i32, Cpu>::from_shape_vec([10], data)?;
        
        // Chain multiple operations
        let result: i32 = tensor
            .iter()
            .filter(|&&x| x % 2 == 0) // Get evens
            .map(|&x| x * x)           // Square them
            .sum();                    // Sum
        
        // 2^2 + 4^2 + 6^2 + 8^2 + 10^2 = 4 + 16 + 36 + 64 + 100 = 220
        assert_eq!(result, 220);
        
        Ok(())
    }

    // Arc Storage Tests

    #[test]
    fn test_arc_cheap_clone() -> Result<(), TensorError> {
        let data = vec![1.0f32; 1000000]; // 1M elements
        let t1 = Tensor1::<f32, Cpu>::from_shape_vec([1000000], data)?;
        
        // Clone should be cheap (O(1), just Arc increment)
        let t2 = t1.clone();
        
        // Both should have same data
        assert_eq!(t1.as_slice().len(), t2.as_slice().len());
        assert_eq!(t1.as_slice()[0], t2.as_slice()[0]);
        
        // Storage should be shared
        assert!(!t1.storage.is_unique());
        assert!(!t2.storage.is_unique());
        
        Ok(())
    }

    #[test]
    fn test_arc_storage_sharing() -> Result<(), TensorError> {
        let data = vec![1, 2, 3, 4, 5];
        let t1 = Tensor1::<i32, Cpu>::from_shape_vec([5], data)?;
        
        // Initially unique
        assert!(t1.storage.is_unique());
        
        // After clone, not unique
        let t2 = t1.clone();
        assert!(!t1.storage.is_unique());
        assert!(!t2.storage.is_unique());
        
        // After dropping t2, t1 becomes unique again
        drop(t2);
        assert!(t1.storage.is_unique());
        
        Ok(())
    }

    #[test]
    fn test_arc_unique_mutation() -> Result<(), TensorError> {
        let data = vec![1, 2, 3, 4, 5];
        let mut t1 = Tensor1::<i32, Cpu>::from_shape_vec([5], data)?;
        
        // Should work: storage is unique
        assert!(t1.storage.is_unique());
        t1.as_slice_mut()[0] = 10;
        assert_eq!(t1.as_slice()[0], 10);
        
        Ok(())
    }

    #[test]
    #[should_panic(expected = "Cannot get mutable slice when storage is shared")]
    fn test_arc_shared_mutation_panics() {
        let data = vec![1, 2, 3, 4, 5];
        let mut t1 = Tensor1::<i32, Cpu>::from_shape_vec([5], data).unwrap();
        let _t2 = t1.clone(); // Share storage
        
        // Should panic: storage is shared
        let _ = t1.as_slice_mut();
    }

    #[test]
    fn test_arc_drop_semantics() -> Result<(), TensorError> {
        let data = vec![1, 2, 3, 4, 5];
        let t1 = Tensor1::<i32, Cpu>::from_shape_vec([5], data)?;
        
        {
            let t2 = t1.clone();
            assert!(!t1.storage.is_unique());
            assert!(!t2.storage.is_unique());
            // t2 dropped here
        }
        
        // After t2 is dropped, t1 should be unique again
        assert!(t1.storage.is_unique());
        
        Ok(())
    }

}
