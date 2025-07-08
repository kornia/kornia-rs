use kornia_image::{allocator::CpuAllocator, Image};
use super::border::BorderType;

pub fn erode(
    input: <Image<u8, 3, CpuAllocator>,
    output: <Image<u8, 3, CpuAllocator>,
    kernel: &[bool],
    ksize: (usize, usize), 
    anchor: Option<(isize, isize)>,
    iterations: usize, 
    border_type: BorderType,
    border_value: u8, 
){

}