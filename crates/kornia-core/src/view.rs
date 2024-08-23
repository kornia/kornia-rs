use crate::TensorAllocator;

pub struct TensorView<T, const N: usize, A: TensorAllocator> {
    pub storage: A::Storage,
}
