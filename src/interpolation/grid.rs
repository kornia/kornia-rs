use ndarray::Array2;

/// Create a meshgrid of x and y coordinates
///
/// # Arguments
///
/// * `x` - A 1D array of x coordinates
/// * `y` - A 1D array of y coordinates
///
/// # Returns
///
/// A tuple of 2D arrays of shape (height, width) containing the x and y coordinates
///
/// # Example
///
/// ```
/// let x = ndarray::Array::linspace(0., 4., 5).insert_axis(ndarray::Axis(0));
/// let y = ndarray::Array::linspace(0., 3., 4).insert_axis(ndarray::Axis(0));
/// let (xx, yy) = kornia_rs::interpolation::meshgrid(&x, &y);
/// assert_eq!(xx.shape(), &[4, 5]);
/// assert_eq!(yy.shape(), &[4, 5]);
/// assert_eq!(xx[[0, 0]], 0.);
/// assert_eq!(xx[[0, 4]], 4.);
/// ```
pub fn meshgrid(x: &Array2<f32>, y: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
    // create the meshgrid of x and y coordinates
    let nx = x.len_of(ndarray::Axis(1));
    let ny = y.len_of(ndarray::Axis(1));

    // broadcast the x and y coordinates to create a 2D grid, and then transpose the y coordinates
    // to create the meshgrid of x and y coordinates of shape (height, width)
    let xx = x.broadcast((ny, nx)).unwrap().to_owned();
    let yy = y.broadcast((nx, ny)).unwrap().t().to_owned();

    (xx, yy)
}
