use kornia_core::{CpuAllocator, Tensor2, TensorError};

/// Create a meshgrid of x and y coordinates
///
/// # Arguments
///
/// * `rows` - The number of rows indicating the height of the grid
/// * `cols` - The number of columns indicating the width of the grid
///
/// # Returns
///
/// A tuple of 2D arrays of shape (rows, cols) containing the x and y coordinates
pub(crate) fn meshgrid(
    rows: usize,
    cols: usize,
) -> Result<(Tensor2<f32>, Tensor2<f32>), TensorError> {
    let mut map_x = vec![];
    for _ in 0..rows {
        for c in 0..cols {
            map_x.push(c as f32);
        }
    }

    let mut map_y = vec![];
    for r in 0..rows {
        for _ in 0..cols {
            map_y.push(r as f32);
        }
    }

    let map_x = Tensor2::from_shape_vec([rows, cols], map_x, CpuAllocator)?;
    let map_y = Tensor2::from_shape_vec([rows, cols], map_y, CpuAllocator)?;

    Ok((map_x, map_y))
}

/// Create a meshgrid of x and y coordinates
pub(crate) fn meshgrid_image(
    rows: usize,
    max_rows: usize,
    cols: usize,
    max_cols: usize,
) -> Result<(Tensor2<f32>, Tensor2<f32>), TensorError> {
    // TODO: review the implementation
    let mut map_x = vec![];
    let step_x = (max_cols - 1) as f32 / (cols - 1) as f32;
    for _ in 0..rows {
        for c in 0..cols {
            map_x.push((c as f32) * step_x);
        }
    }

    let mut map_y = vec![];
    let step_y = (max_rows - 1) as f32 / (rows - 1) as f32;
    for r in 0..rows {
        for _ in 0..cols {
            map_y.push((r as f32) * step_y);
        }
    }

    let map_x = Tensor2::from_shape_vec([rows, cols], map_x, CpuAllocator).unwrap();
    let map_y = Tensor2::from_shape_vec([rows, cols], map_y, CpuAllocator).unwrap();

    Ok((map_x, map_y))
}
