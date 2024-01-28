/// Compute the strides from the shape of a tensor.
///
/// # Arguments
///
/// * `shape` - The shape of the tensor.
///
/// # Returns
///
/// * `strides` - The strides of the tensor.
fn get_strides_from_shape(shape: &[i64]) -> Vec<i64> {
    let mut strides = vec![0i64; shape.len()];

    let mut c = 1;
    strides[shape.len() - 1] = c;
    for i in (1..shape.len()).rev() {
        c *= shape[i];
        strides[i - 1] = c;
    }

    strides
}

/// A data structure to represent a tensor.
///
/// # Attributes
///
/// * `shape` - The shape of the tensor.
/// * `data` - The data of the tensor.
/// * `strides` - The strides of the tensor.
///
/// # Example
///
/// ```
/// use kornia_rs::tensor::Tensor;
///
/// let shape: Vec<i64> = vec![1, 1, 2, 2];
/// let data: Vec<u8> = vec![1, 2, 3, 4];
/// let t = Tensor::new(shape, data);
/// assert_eq!(t.shape, vec![1, 1, 2, 2]);
#[derive(Clone)]
pub struct Tensor {
    pub shape: Vec<i64>,
    pub data: Vec<u8>,
    pub strides: Vec<i64>,
}

/// Implementation of the Tensor struct.
impl Tensor {
    /// Creates a new `Tensor` with the given shape and data.
    ///
    /// # Arguments
    ///
    /// * `shape` - A vector representing the shape of the tensor.
    /// * `data` - A vector containing the data of the tensor.
    ///
    /// # Returns
    ///
    /// A new `Tensor` instance.
    pub fn new(shape: Vec<i64>, data: Vec<u8>) -> Self {
        let strides = get_strides_from_shape(&shape);
        Tensor {
            shape,
            data,
            strides,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::get_strides_from_shape;
    use crate::tensor::Tensor;

    #[test]
    fn constructor_default() {
        let shape: Vec<usize> = vec![1, 1, 2, 2];
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor {
            shape: shape.iter().map(|x| *x as i64).collect(),
            data,
            strides: vec![0, 0, 0, 0],
        };
        assert_eq!(t.shape, vec![1, 1, 2, 2]);
        assert_eq!(t.data, vec![1, 2, 3, 4]);
        assert_eq!(t.strides, vec![0, 0, 0, 0]);
    }

    #[test]
    fn constructor_new() {
        let shape: Vec<i64> = vec![1, 1, 2, 2];
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor::new(shape, data);
        assert_eq!(t.shape, vec![1, 1, 2, 2]);
        assert_eq!(t.data, vec![1, 2, 3, 4]);
        assert_eq!(t.strides, vec![4, 4, 2, 1]);
    }

    #[test]
    fn strides_from_shape() {
        let shape: Vec<i64> = vec![1, 1, 2, 2];
        let strides = get_strides_from_shape(&shape);
        assert_eq!(strides, vec![4, 4, 2, 1]);
    }
}
