use kornia_tensor::{CpuAllocator, CpuTensor2, TensorError};
use num_traits::Float;
use rayon::iter::ParallelIterator;
use rayon::{iter::IndexedParallelIterator, slice::ParallelSliceMut};

/// Create a meshgrid of x and y coordinates using a custom function
///
/// # Arguments
///
/// * `cols` - The number of columns (width) of the grid
/// * `rows` - The number of rows (height) of the grid
/// * `f` - A function that takes column and row indices (u, v) and returns (x, y) coordinates
///
/// # Returns
///
/// A tuple of two 2D tensors of shape (rows, cols) containing the x and y coordinates
///
/// # Errors
///
/// Returns a `TensorError` if tensor allocation fails or if the provided function `f` returns an error
///
/// # Example
///
/// ```
/// use kornia_imgproc::interpolation::grid::meshgrid_from_fn;
///
/// let (map_x, map_y) = meshgrid_from_fn(3, 2, |u, v| {
///     Ok((u as f32 * 0.5, v as f32 * 2.0))
/// }).unwrap();
///
/// assert_eq!(map_x.shape, [2, 3]);
/// assert_eq!(map_y.shape, [2, 3]);
/// ```
pub fn meshgrid_from_fn<T>(
    cols: usize,
    rows: usize,
    f: impl Fn(usize, usize) -> Result<(T, T), Box<dyn std::error::Error + Send + Sync>> + Send + Sync,
) -> Result<(CpuTensor2<T>, CpuTensor2<T>), TensorError>
where
    T: Float + Send + Sync,
{
    // allocate the output tensors
    let mut map_x = CpuTensor2::<T>::zeros([rows, cols], CpuAllocator);
    let mut map_y = CpuTensor2::<T>::zeros([rows, cols], CpuAllocator);

    // fill the output tensors
    map_x
        .as_slice_mut()
        .par_chunks_exact_mut(cols)
        .zip_eq(map_y.as_slice_mut().par_chunks_exact_mut(cols))
        .enumerate()
        .try_for_each(|(v, (row_x, row_y))| {
            for (u, (x, y)) in row_x.iter_mut().zip(row_y.iter_mut()).enumerate() {
                // apply the function to the indices
                let (x_out, y_out) =
                    f(u, v).map_err(|e| TensorError::UnsupportedOperation(e.to_string()))?;

                // assign the output to the tensors
                *x = x_out;
                *y = y_out;
            }
            Ok::<(), TensorError>(())
        })?;

    Ok((map_x, map_y))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meshgrid_from_fn_identity() -> Result<(), TensorError> {
        let (map_x, map_y) = meshgrid_from_fn(3, 2, |u, v| Ok((u as f64, v as f64)))?;

        assert_eq!(map_x.shape, [2, 3]);
        assert_eq!(map_y.shape, [2, 3]);

        let expected_x = [0.0f64, 1.0, 2.0, 0.0, 1.0, 2.0];
        let expected_y = [0.0f64, 0.0, 0.0, 1.0, 1.0, 1.0];

        assert_eq!(map_x.as_slice(), expected_x.as_slice());
        assert_eq!(map_y.as_slice(), expected_y.as_slice());

        Ok(())
    }

    #[test]
    fn test_meshgrid_from_fn_scaled() -> Result<(), TensorError> {
        let (map_x, map_y) =
            meshgrid_from_fn(3, 2, |u, v| Ok((u as f32 * 0.5, v as f32 * 2.0))).unwrap();

        assert_eq!(map_x.shape, [2, 3]);
        assert_eq!(map_y.shape, [2, 3]);

        let expected_x = [0.0, 0.5, 1.0, 0.0, 0.5, 1.0];
        let expected_y = [0.0, 0.0, 0.0, 2.0, 2.0, 2.0];

        assert_eq!(map_x.as_slice(), expected_x.as_slice());
        assert_eq!(map_y.as_slice(), expected_y.as_slice());

        Ok(())
    }
}
