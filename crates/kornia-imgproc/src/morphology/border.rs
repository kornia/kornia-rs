use kornia_image::{allocator::CpuAllocator, Image};

/// Border handling modes for morphological operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BorderType {
    /// Fill all out-of-bounds pixels with a constant value.
    /// Corresponds to OpenCV's `BORDER_CONSTANT`.
    Constant,

    /// Replicate the value of the nearest border pixel.
    /// Corresponds to OpenCV's `BORDER_REPLICATE`.
    Replicate,

    /// Reflect the image across the border, excluding the border pixel.
    /// Corresponds to OpenCV's `BORDER_REFLECT`.
    Reflect,

    /// Wrap the image around periodically.
    /// Corresponds to OpenCV's `BORDER_WRAP`.
    Wrap,

    /// Reflect the image across the border, including the border pixel.
    /// Corresponds to OpenCV's `BORDER_REFLECT_101`.
    Reflect101,

    /// Do not modify out-of-bounds pixels (transparent border).
    /// Corresponds to OpenCV's `BORDER_TRANSPARENT`.
    Transparent,
}

fn reflect_index(idx: isize, len: isize) -> isize {
    if idx < 0 {
        (-idx) - 1
    } else if idx >= len {
        2 * len - idx - 1
    } else {
        idx
    }
}

fn reflect101_index(idx: isize, len: isize) -> isize {
    if idx < 0 {
        -idx
    } else if idx >= len {
        2 * len - idx - 2
    } else {
        idx
    }
}

fn get_pixel_with_border<const C: usize>(
    img: &Image<u8, C, CpuAllocator>,
    x: isize,
    y: isize,
    c: usize,
    border: BorderType,
    border_value: u8,
) -> u8
where
    [(); C]:,
{
    let width = img.width() as isize;
    let height = img.height() as isize;

    let (nx, ny) = match border {
        BorderType::Constant => {
            if x < 0 || y < 0 || x >= width || y >= height {
                return border_value;
            }
            (x, y)
        }
        BorderType::Replicate => {
            let nx = x.clamp(0, width - 1);
            let ny = y.clamp(0, height - 1);
            (nx, ny)
        }
        BorderType::Reflect => {
            let nx = reflect_index(x, width);
            let ny = reflect_index(y, height);
            (nx, ny)
        }
        BorderType::Wrap => {
            let nx = ((x % width) + width) % width;
            let ny = ((y % height) + height) % height;
            (nx, ny)
        }
        BorderType::Reflect101 => {
            let nx = reflect101_index(x, width);
            let ny = reflect101_index(y, height);
            (nx, ny)
        }
        BorderType::Transparent => {
            if x < 0 || y < 0 || x >= width || y >= height {
                return border_value;
            }
            (x, y)
        }
    };

    img.get([ny as usize, nx as usize, c])
        .copied()
        .unwrap_or(border_value)
}
