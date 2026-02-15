// NOTE: Consider whether to place these utilities under a more general-purpose crate,
// such as kornia-linalg, or another location we decide on later.

use kornia_algebra::Mat3F32;
use kornia_image::{allocator::ImageAllocator, Image, ImageSize};

/// Calculates the total number of tiles needed to cover an image of the given size,
/// including partial tiles at the edges.
///
/// # Arguments
///
/// * `size` - The size of the image.
/// * `tile_size` - The size of each tile.
///
/// # Returns
///
/// A `Point2d` representing the number of tiles along the x and y axes.
pub(crate) fn find_total_tiles(size: ImageSize, tile_size: usize) -> Point2d {
    Point2d {
        x: size.width.div_ceil(tile_size),
        y: size.height.div_ceil(tile_size),
    }
}

/// Calculates the number of full tiles that fit within an image of the given size.
///
/// # Arguments
///
/// * `size` - The size of the image.
/// * `tile_size` - The size of each tile.
///
/// # Returns
///
/// A `Point2d` representing the number of full tiles along the x and y axes.
pub(crate) fn find_full_tiles(size: ImageSize, tile_size: usize) -> Point2d {
    Point2d {
        x: size.width / tile_size,
        y: size.height / tile_size,
    }
}

/// A 'Point2d' struct that can be used to represent a point in 2D space.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct Point2d<T = usize> {
    /// The x-coordinate of the point.
    pub x: T,
    /// The y-coordinate of the point.
    pub y: T,
}

impl From<Point2d<f32>> for kornia_algebra::Vec2F32 {
    fn from(p: Point2d<f32>) -> Self {
        Self::new(p.x, p.y)
    }
}

impl From<kornia_algebra::Vec2F32> for Point2d<f32> {
    fn from(v: kornia_algebra::Vec2F32) -> Self {
        Self { x: v.x, y: v.y }
    }
}

#[derive(Default, Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
/// Represents a pixel that can be white, black, or skipped.
pub enum Pixel {
    /// A white pixel.
    White = 255,
    /// A black pixel.
    Black = 0,
    /// A pixel to be skipped.
    #[default]
    Skip = 127,
}

impl PartialEq<u8> for Pixel {
    fn eq(&self, other: &u8) -> bool {
        *self as u8 == *other
    }
}

/// Computes the homography matrix from four point correspondences.
///
/// # Arguments
///
/// * `c` - A 4x4 array where each row contains the coordinates of a correspondence:
///   [x, y, x', y'] where (x, y) maps to (x', y').
///
/// # Returns
///
/// An `Option<Mat3F32>` containing the 3x3 homography matrix if successful, or `None` if the matrix is singular.
pub(crate) fn homography_compute(c: [[f32; 4]; 4]) -> Option<Mat3F32> {
    #[rustfmt::skip]
    let mut a = [
        c[0][0], c[0][1], 1.0, 0.0,     0.0,     0.0, -c[0][0]*c[0][2], -c[0][1]*c[0][2], c[0][2],
        0.0,     0.0,     0.0, c[0][0], c[0][1], 1.0, -c[0][0]*c[0][3], -c[0][1]*c[0][3], c[0][3],
        c[1][0], c[1][1], 1.0, 0.0,     0.0,     0.0, -c[1][0]*c[1][2], -c[1][1]*c[1][2], c[1][2],
        0.0,     0.0,     0.0, c[1][0], c[1][1], 1.0, -c[1][0]*c[1][3], -c[1][1]*c[1][3], c[1][3],
        c[2][0], c[2][1], 1.0, 0.0,     0.0,     0.0, -c[2][0]*c[2][2], -c[2][1]*c[2][2], c[2][2],
        0.0,     0.0,     0.0, c[2][0], c[2][1], 1.0, -c[2][0]*c[2][3], -c[2][1]*c[2][3], c[2][3],
        c[3][0], c[3][1], 1.0, 0.0,     0.0,     0.0, -c[3][0]*c[3][2], -c[3][1]*c[3][2], c[3][2],
        0.0,     0.0,     0.0, c[3][0], c[3][1], 1.0, -c[3][0]*c[3][3], -c[3][1]*c[3][3], c[3][3],
    ];

    const EPSILON: f32 = 1e-10;

    // Eliminate
    for col in 0..8 {
        // Find best row to swap with
        let mut max_val = 0.0;
        let mut max_val_idx = -1;

        for row in col..8 {
            let val = a[row * 9 + col].abs();
            if val > max_val {
                max_val = val;
                max_val_idx = row as isize;
            }
        }

        if max_val_idx < 0 {
            return None;
        }

        let max_val_idx = max_val_idx as usize;

        if max_val < EPSILON {
            // Matrix is singular
            return None;
        }

        // Swap to get best row
        if max_val_idx != col {
            for i in col..9 {
                a.swap(col * 9 + i, max_val_idx * 9 + i);
            }
        }

        // Do eliminate
        for i in (col + 1)..8 {
            let f = a[i * 9 + col] / a[col * 9 + col];
            a[i * 9 + col] = 0.0;
            for j in (col + 1)..9 {
                a[i * 9 + j] -= f * a[col * 9 + j];
            }
        }
    }

    // Back solve
    for col in (0..8).rev() {
        let mut sum = 0.0;
        for i in (col + 1)..8 {
            sum += a[col * 9 + i] * a[i * 9 + 8];
        }
        a[col * 9 + 8] = (a[col * 9 + 8] - sum) / a[col * 9 + col];
    }

    // Variables solve as: h11, h12, h13, h21, h22, h23, h31, h32. h33 is 1.0.
    // glam::Mat3 is column-major: [h11, h21, h31, h12, h22, h32, h13, h23, h33]
    Some(Mat3F32::from_cols_array(&[
        a[8], a[35], a[62], // col 0
        a[17], a[44], a[71], // col 1
        a[26], a[53], 1.0, // col 2
    ]))
}

/// Returns the interpolated value for a given floating-point pixel coordinate in a grayscale image.
///
/// # Arguments
///
/// * `src` - The source grayscale image.
/// * `p` - The floating-point pixel coordinate.
///
/// # Returns
///
/// An `Option<f32>` containing the interpolated pixel value, or `None` if the coordinate is out of bounds.
// TODO: Make interpolate function in kornia-imgproc generic and use that instead
pub(crate) fn value_for_pixel<A: ImageAllocator>(
    src: &Image<u8, 1, A>,
    p: kornia_algebra::Vec2F32,
) -> Option<f32> {
    let src_slice = src.as_slice();

    let x1 = (p.x - 0.5).floor() as isize;
    let x2 = (p.x - 0.5).ceil() as isize;
    let x = p.x - 0.5 - x1 as f32;

    let y1 = (p.y - 0.5).floor() as isize;
    let y2 = (p.y - 0.5).ceil() as isize;
    let y = p.y - 0.5 - y1 as f32;

    if x1 < 0 || x2 >= src.width() as isize || y1 < 0 || y2 >= src.height() as isize {
        return None;
    }

    let x1 = x1 as usize;
    let x2 = x2 as usize;

    let y1 = y1 as usize;
    let y2 = y2 as usize;

    Some(
        src_slice[y1 * src.width() + x1] as f32 * (1.0 - x) * (1.0 - y)
            + src_slice[y1 * src.width() + x2] as f32 * x * (1.0 - y)
            + src_slice[y2 * src.width() + x1] as f32 * (1.0 - x) * y
            + src_slice[y2 * src.width() + x2] as f32 * x * y,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_io::png::read_image_png_mono8;

    #[test]
    fn test_pixel_value() {
        let white = Pixel::White;
        let black = Pixel::Black;
        let skip = Pixel::Skip;

        assert_eq!(white, 255);
        assert_eq!(black, 0);
        assert_eq!(skip, 127);
    }

    #[test]
    fn test_find_total_tiles() {
        let size = ImageSize {
            width: 101,
            height: 257,
        };

        let tiles = find_total_tiles(size, 4);
        let expected = Point2d { x: 26, y: 65 };

        assert_eq!(tiles, expected)
    }

    #[test]
    fn test_find_full_tiles() {
        let size = ImageSize {
            width: 101,
            height: 257,
        };

        let tiles = find_full_tiles(size, 4);
        let expected = Point2d { x: 25, y: 64 };

        assert_eq!(tiles, expected)
    }

    #[test]
    fn test_matrix_mul() {
        #[rustfmt::skip]
        let a = Mat3F32::from_cols_array(&[
            12.0, 3.0, 9.0, // col 0
            8.0, 17.0, 8.0, // col 1
            4.0, 14.0, 10.0, // col 2
        ]);

        #[rustfmt::skip]
        let b = Mat3F32::from_cols_array(&[
            5.0, 6.0, 7.0, // col 0
            19.0, 15.0, 8.0, // col 1
            3.0, 9.0, 16.0, // col 2
        ]);

        #[rustfmt::skip]
        let expected = Mat3F32::from_cols_array(&[
            136.0, 215.0, 163.0, // col 0
            380.0, 424.0, 371.0, // col 1
            172.0, 386.0, 259.0, // col 2
        ]);

        let mul = a * b;
        assert_eq!(mul, expected);
    }

    #[test]
    fn test_homography_compute() {
        #[rustfmt::skip]
        let corr_arr = [
            [-1.0, -1.0, 27.0,  3.0],
            [ 1.0, -1.0, 27.0, 27.0],
            [ 1.0,  1.0,  3.0, 27.0],
            [-1.0,  1.0,  3.0,  3.0],
        ];

        let h = homography_compute(corr_arr).unwrap();
        // glam::Mat3 is column-major: [h11, h21, h31, h12, h22, h32, h13, h23, h33]
        // expected results from original row-major: [-0.0, -12.0, 15.0, 12.0, -0.0, 15.0, -0.0, 0.0, 1.0]
        let expected = Mat3F32::from_cols_array(&[
            -0.0, 12.0, -0.0, // col 0: h11, h21, h31
            -12.0, -0.0, 0.0, // col 1: h12, h22, h32
            15.0, 15.0, 1.0, // col 2: h13, h23, h33
        ]);

        assert_eq!(h, expected);
    }

    #[test]
    fn test_value_for_pixel() -> Result<(), Box<dyn std::error::Error>> {
        let src = read_image_png_mono8("../../tests/data/apriltag.png")?;

        assert_eq!(
            value_for_pixel(&src, kornia_algebra::Vec2F32::new(15.0, 15.0)),
            Some(191.25)
        );

        assert_eq!(
            value_for_pixel(&src, kornia_algebra::Vec2F32::new(3.0, 15.0)),
            Some(127.5)
        );

        assert_eq!(
            value_for_pixel(&src, kornia_algebra::Vec2F32::new(13.0, 8.0)),
            Some(0.0)
        );

        assert_eq!(
            value_for_pixel(&src, kornia_algebra::Vec2F32::new(26.0, 1.0)),
            Some(255.0)
        );

        Ok(())
    }
}
