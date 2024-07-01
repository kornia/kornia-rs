mod base;

pub use base::{Tensor, TensorError};

// aliases
pub type Tensor1<T> = Tensor<T, 1>;
pub type Tensor2<T> = Tensor<T, 2>;
pub type Tensor3<T> = Tensor<T, 3>;
pub type Tensor4<T> = Tensor<T, 4>;
