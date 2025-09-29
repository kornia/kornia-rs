use kornia_image::{allocator::ImageAllocator, Image};

pub struct Video<T, const C: usize, A: ImageAllocator> {
    frames: Vec<Image<T, C, A>>,
}

impl<T, const C: usize, A: ImageAllocator> Video<T, C, A> {
    pub fn new(frames: Vec<Image<T, C, A>>) -> Self {
        Self { frames }
    }
}
