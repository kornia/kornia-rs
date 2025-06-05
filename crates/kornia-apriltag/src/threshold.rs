use crate::{errors::AprilTagError, utils::PixelTrait};
use kornia_image::{allocator::ImageAllocator, Image, ImageError};
use std::ops::{Add, Div, Sub};

/// TODO
///
/// ## Recommended values for `min_white_black_diff`
///
/// | Image Type | `min_white_black_diff` |
/// |------------|------------------------|
/// | 8-bit      | 10-20                  |
/// | 16-bit     | 500-1000               |
/// | 32-bit     | 100,000-1,000,000      |
/// | float      | 0.05-0.1               |
// TODO: Add support for parallelism
pub fn adaptive_threshold<
    T: PixelTrait
        + PartialOrd
        + Copy
        + Add<T, Output = T>
        + Sub<T, Output = T>
        + Div<T, Output = T>
        + From<u8>,
    A1: ImageAllocator,
    A2: ImageAllocator,
>(
    src: &Image<T, 1, A1>,
    dst: &mut Image<T, 1, A2>,
    tile_size: usize,
    min_white_black_diff: T,
) -> Result<(), AprilTagError> {
    if src.size() != dst.size() {
        return Err(
            ImageError::InvalidImageSize(src.cols(), src.rows(), dst.cols(), dst.rows()).into(),
        );
    }

    let src_data = src.as_slice();
    let dst_data = dst.as_slice_mut();

    let tiles_x_len = (src.width() as f32 / tile_size as f32).ceil() as usize; // number of horizontal tiles
    let tiles_y_len = (src.height() as f32 / tile_size as f32).ceil() as usize; // number of vertical tiles

    // pixels available in the tiles at edge
    let last_tile_x_px = if src.width() % tile_size == 0 {
        tile_size
    } else {
        src.width() % tile_size
    };

    let last_tile_y_px = if src.height() % tile_size == 0 {
        tile_size
    } else {
        src.height() % tile_size
    };

    let mut tile_min: Vec<T> = Vec::with_capacity(tile_size * tile_size);
    let mut tile_max: Vec<T> = Vec::with_capacity(tile_size * tile_size);

    // Calculate extrema (i.e. min & max grayscale value) of each tile
    for y in 0..tiles_y_len {
        for x in 0..tiles_x_len {
            let mut local_min = T::WHITE;
            let mut local_max = T::BLACK;

            // Number of Horizontal Pixels in the current tile
            let tile_x_px = if x == tiles_x_len - 1 {
                last_tile_x_px
            } else {
                tile_size
            };

            // Number of vertical Pixels in the current tile
            let tile_y_px = if y == tiles_y_len - 1 {
                last_tile_y_px
            } else {
                tile_size
            };

            for y_px in 0..tile_y_px {
                let row = ((y * tile_size) + y_px) * src.width();
                let start_index = row + (x * tile_size);
                let end_index = start_index + tile_x_px;

                let row_pxs = &src_data[start_index..end_index];

                for px in row_pxs {
                    if px < &local_min {
                        local_min = *px;
                    }

                    if px > &local_max {
                        local_max = *px;
                    }
                }
            }

            tile_min.push(local_min);
            tile_max.push(local_max);
        }
    }

    // Binarize the image
    for y in 0..tiles_y_len {
        for x in 0..tiles_x_len {
            // Number of Horizontal Pixels in the current tile
            let tile_x_px = if x == tiles_x_len - 1 {
                last_tile_x_px
            } else {
                tile_size
            };

            // Number of vertical Pixels in the current tile
            let tile_y_px = if y == tiles_y_len - 1 {
                last_tile_y_px
            } else {
                tile_size
            };

            let tile_index = (y * tiles_x_len) + x;

            let mut neighbor_min = tile_min[tile_index];
            let mut neighbor_max = tile_max[tile_index];

            // Low constrast tile, Skip processing
            if neighbor_max - neighbor_min < min_white_black_diff {
                for y_px in 0..tile_y_px {
                    let row = ((y * tile_size) + y_px) * src.width();
                    let start_index = row + (x * tile_size);
                    let end_index = start_index + tile_x_px;

                    for px in dst_data.iter_mut().take(end_index).skip(start_index) {
                        *px = T::SKIP_PROCESSING;
                    }
                }

                continue;
            }

            if x + 1 != tiles_x_len {
                // Rightmost neigbor exists

                if tile_min[tile_index + 1] < neighbor_min {
                    neighbor_min = tile_min[tile_index + 1];
                }

                if tile_max[tile_index + 1] > neighbor_max {
                    neighbor_max = tile_max[tile_index + 1];
                }
            }

            if x != 0 {
                // Leftmost neighbor exists

                if tile_min[tile_index - 1] < neighbor_min {
                    neighbor_min = tile_min[tile_index - 1];
                }

                if tile_max[tile_index - 1] > neighbor_max {
                    neighbor_max = tile_max[tile_index - 1];
                }
            }

            if y + 1 != tiles_y_len {
                // Bottom-most neighbor exists
                let bottom_tile_index = tile_index + tiles_x_len;

                if tile_min[bottom_tile_index] < neighbor_min {
                    neighbor_min = tile_min[bottom_tile_index];
                }

                if tile_max[bottom_tile_index] > neighbor_max {
                    neighbor_max = tile_max[bottom_tile_index];
                }
            }

            if y != 0 {
                // Uppermost neighbor exists
                let upper_tile_index = tile_index - tiles_x_len;

                if tile_min[upper_tile_index] < neighbor_min {
                    neighbor_min = tile_min[upper_tile_index];
                }

                if tile_max[upper_tile_index] > neighbor_max {
                    neighbor_max = tile_max[upper_tile_index];
                }
            }

            let thresh = neighbor_min + (neighbor_max - neighbor_min) / T::from(2u8);

            for y_px in 0..tile_y_px {
                let row = ((y * tile_size) + y_px) * src.width();
                let start_index = row + (x * tile_size);
                let end_index = start_index + tile_x_px;

                for px_index in start_index..end_index {
                    let px = &src_data[px_index];

                    dst_data[px_index] = if px > &thresh { T::WHITE } else { T::BLACK };
                }
            }
        }
    }

    Ok(())
}
