use std::ops::Add;

/// Compute the strides from the shape of a tensor.
///
/// # Arguments
///
/// * `shape` - The shape of the tensor.
///
/// # Returns
///
/// * `strides` - The strides of the tensor.
fn get_strides_from_shape<const N: usize>(shape: [usize; N]) -> [usize; N] {
    let mut strides: [usize; N] = [0; N];
    let mut stride = 1;
    for i in (0..shape.len()).rev() {
        strides[i] = stride;
        stride *= shape[i];
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
pub struct Tensor<T, const N: usize> {
    pub data: Vec<T>,
    pub shape: [usize; N],
    pub strides: [usize; N],
}

/// Implementation of the Tensor struct.
impl<T, const N: usize> Tensor<T, N> {
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
    pub fn new(shape: [usize; N], data: Vec<T>) -> Self {
        let numel = shape.iter().product::<usize>();
        if numel != data.len() {
            panic!("The number of elements in the data does not match the shape of the tensor.");
        }
        let strides = get_strides_from_shape(shape);
        Tensor {
            shape,
            data,
            strides,
        }
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn get(&self, index: [usize; N]) -> &T {
        let mut offset = 0;
        for i in 0..N {
            offset += index[i] * self.strides[i];
        }
        &self.data[offset]
    }

    pub fn reshape<const M: usize>(self, shape: [usize; M]) -> Tensor<T, M> {
        let numel = shape.iter().product::<usize>();
        if numel != self.data.len() {
            panic!("The number of elements in the data does not match the shape of the tensor.");
        }
        let strides = get_strides_from_shape(shape);
        Tensor {
            shape,
            data: self.data,
            strides,
        }
    }

    pub fn add(&self, other: &Tensor<T, N>) -> Tensor<T, N>
    where
        T: Add<Output = T> + Copy,
    {
        let data: Vec<T> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| *a + *b)
            .collect();
        Tensor::new(self.shape, data)
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::Tensor;

    #[test]
    fn constructor_1d() {
        let data: Vec<u8> = vec![1];
        let t = Tensor::<u8, 1>::new([1], data);
        assert_eq!(t.shape, [1]);
        assert_eq!(t.data, vec![1]);
        assert_eq!(t.strides, [1]);
        assert_eq!(t.numel(), 1);
    }

    #[test]
    fn constructor_2d() {
        let data: Vec<u8> = vec![1, 2];
        let t = Tensor::<u8, 2>::new([1, 2], data);
        assert_eq!(t.shape, [1, 2]);
        assert_eq!(t.data, vec![1, 2]);
        assert_eq!(t.strides, [2, 1]);
        assert_eq!(t.numel(), 2);
    }

    #[test]
    fn get_1d() {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor::<u8, 1>::new([4], data);
        assert_eq!(*t.get([0]), 1);
        assert_eq!(*t.get([1]), 2);
        assert_eq!(*t.get([2]), 3);
        assert_eq!(*t.get([3]), 4);
    }

    #[test]
    fn get_2d() {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor::<u8, 2>::new([2, 2], data);
        assert_eq!(*t.get([0, 0]), 1);
        assert_eq!(*t.get([0, 1]), 2);
        assert_eq!(*t.get([1, 0]), 3);
        assert_eq!(*t.get([1, 1]), 4);
    }

    #[test]
    fn get_3d() {
        let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6];
        let t = Tensor::<u8, 3>::new([2, 1, 3], data);
        assert_eq!(*t.get([0, 0, 0]), 1);
        assert_eq!(*t.get([0, 0, 1]), 2);
        assert_eq!(*t.get([0, 0, 2]), 3);
        assert_eq!(*t.get([1, 0, 0]), 4);
        assert_eq!(*t.get([1, 0, 1]), 5);
        assert_eq!(*t.get([1, 0, 2]), 6);
    }

    #[test]
    fn add_1d() {
        let data1: Vec<u8> = vec![1, 2, 3, 4];
        let t1 = Tensor::<u8, 1>::new([4], data1);
        let data2: Vec<u8> = vec![1, 2, 3, 4];
        let t2 = Tensor::<u8, 1>::new([4], data2);
        let t3 = t1.add(&t2);
        assert_eq!(t3.data, vec![2, 4, 6, 8]);
    }

    #[test]
    fn add_2d() {
        let data1: Vec<u8> = vec![1, 2, 3, 4];
        let t1 = Tensor::<u8, 2>::new([2, 2], data1);
        let data2: Vec<u8> = vec![1, 2, 3, 4];
        let t2 = Tensor::<u8, 2>::new([2, 2], data2);
        let t3 = t1.add(&t2);
        assert_eq!(t3.data, vec![2, 4, 6, 8]);
    }

    #[test]
    fn add_3d() {
        let data1: Vec<u8> = vec![1, 2, 3, 4, 5, 6];
        let t1 = Tensor::<u8, 3>::new([2, 1, 3], data1);
        let data2: Vec<u8> = vec![1, 2, 3, 4, 5, 6];
        let t2 = Tensor::<u8, 3>::new([2, 1, 3], data2);
        let t3 = t1.add(&t2);
        assert_eq!(t3.data, vec![2, 4, 6, 8, 10, 12]);
    }

    #[test]
    fn reshape_1d() {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor::<u8, 1>::new([4], data);
        let t2 = t.reshape([2, 2]);
        assert_eq!(t2.shape, [2, 2]);
        assert_eq!(t2.data, vec![1, 2, 3, 4]);
        assert_eq!(t2.strides, [2, 1]);
        assert_eq!(t2.numel(), 4);
    }

    #[test]
    fn reshape_2d() {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor::<u8, 2>::new([2, 2], data);
        let t2 = t.reshape([4]);
        assert_eq!(t2.shape, [4]);
        assert_eq!(t2.data, vec![1, 2, 3, 4]);
        assert_eq!(t2.strides, [1]);
        assert_eq!(t2.numel(), 4);
    }

    #[test]
    fn reshape_get_1d() {
        let data: Vec<u8> = vec![1, 2, 3, 4];
        let t = Tensor::<u8, 1>::new([4], data);
        let t2 = t.reshape([2, 2]);
        assert_eq!(*t2.get([0, 0]), 1);
        assert_eq!(*t2.get([0, 1]), 2);
        assert_eq!(*t2.get([1, 0]), 3);
        assert_eq!(*t2.get([1, 1]), 4);
        assert_eq!(t2.numel(), 4);
    }
}
