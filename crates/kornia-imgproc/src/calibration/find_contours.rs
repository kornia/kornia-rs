use kornia::prelude::*;
use kornia_image::{Image, ImageError};
use std::collections::VecDeque;

/// specification for border-type
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BorderType {
    Outer,
    Hole,
}

/// contour found in image
#[derive(Debug, Clone)]
pub struct Contour<T> {
    pub points: Vec<Point<T>>,
    pub border_type: BorderType,
    pub parent: Option<usize>,
}

/// constructor for contour
impl<T> Contour<T> {
    pub fn new(points: Vec<Point<T>>, border_type: BorderType, parent: Option<usize>) -> Self {
        Contour {
            points,
            border_type,
            parent,
        }
    }
}

pub fn find_contours(src: &Image<f32, 1>, threshold: u8) -> Vec<Contour<T>>
where
    T: num::Num + num::NumCast + Copy + PartialEq + Eq,
{
    let width = image.width() as usize;
    let height = image.height() as usize;
    let mut image_values = vec![0i32; height * width];
}
