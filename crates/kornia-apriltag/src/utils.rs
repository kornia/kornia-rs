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

/// Represents a point in 2D space.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct Point2d<T = usize> {
    /// The x-coordinate of the point.
    pub x: T,
    /// The y-coordinate of the point.
    pub y: T,
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
/// An `Option<[f32; 9]>` containing the flattened 3x3 homography matrix if successful, or `None` if the matrix is singular.
pub(crate) fn homography_compute(c: [[f32; 4]; 4]) -> Option<[f32; 9]> {
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

    Some([a[8], a[17], a[26], a[35], a[44], a[53], a[62], a[71], 1.0])
}

pub(crate) fn matrix_3x3_cholesky(a: &[[f32; 3]; 3], r: &mut [f32; 9]) {
    r[0] = a[0][0].sqrt();
    r[3] = a[0][1] / r[0];
    r[6] = a[0][2] / r[0];

    r[4] = (a[1][1] - r[3] * r[3]).sqrt();
    r[7] = (a[1][2] - r[3] * r[6]) / r[4];

    r[8] = (a[2][2] - r[6] * r[6] - r[7] * r[7]).sqrt();

    r[1] = 0.0;
    r[2] = 0.0;
    r[5] = 0.0;
}

pub(crate) fn matrix_3x3_lower_triange_inverse(a: &[f32; 9], r: &mut [f32; 9]) {
    r[0] = 1.0 / a[0];
    r[3] = -a[3] * r[0] / a[4];
    r[4] = 1.0 / a[4];
    r[6] = (-a[6] * r[0] - a[7] * r[3]) / a[8];
    r[7] = -a[7] * r[4] / a[8];
    r[8] = 1.0 / a[8];
}

#[rustfmt::skip]
pub(crate) const fn matrix_3x3_mul(a: &[f32; 9], b: &[f32; 9]) -> [f32; 9] {
    [
        a[0]*b[0] + a[1]*b[3] + a[2]*b[6],      a[0]*b[1] + a[1]*b[4] + a[2]*b[7],      a[0]*b[2] + a[1]*b[5] + a[2]*b[8],
        a[3]*b[0] + a[4]*b[3] + a[5]*b[6],      a[3]*b[1] + a[4]*b[4] + a[5]*b[7],      a[3]*b[2] + a[4]*b[5] + a[5]*b[8],
        a[6]*b[0] + a[7]*b[3] + a[8]*b[6],      a[6]*b[1] + a[7]*b[4] + a[8]*b[7],      a[6]*b[2] + a[7]*b[5] + a[8]*b[8],
    ]
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
pub(crate) fn value_for_pixel<A: ImageAllocator>(
    src: &Image<u8, 1, A>,
    p: Point2d<f32>,
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
        let a = [
            12.0,  8.0,  4.0,
             3.0, 17.0, 14.0,
             9.0,  8.0, 10.0,
        ];

        #[rustfmt::skip]
        let b = [
            5.0, 19.0,  3.0,
            6.0, 15.0,  9.0,
            7.0,  8.0, 16.0,
        ];

        #[rustfmt::skip]
        let expected = [
            136.0, 380.0,  172.0,
            215.0, 424.0,  386.0,
            163.0, 371.0,  259.0,
        ];

        let mul = matrix_3x3_mul(&a, &b);
        assert_eq!(mul, expected)
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

        let h = homography_compute(corr_arr);
        let expected = Some([-0.0, -12.0, 15.0, 12.0, -0.0, 15.0, -0.0, 0.0, 1.0]);

        assert_eq!(h, expected)
    }

    #[test]
    fn test_value_for_pixel() -> Result<(), Box<dyn std::error::Error>> {
        let src = read_image_png_mono8("../../tests/data/apriltag.png")?;

        assert_eq!(
            value_for_pixel(&src, Point2d { x: 15.0, y: 15.0 }),
            Some(191.25)
        );

        assert_eq!(
            value_for_pixel(&src, Point2d { x: 3.0, y: 15.0 }),
            Some(127.5)
        );

        assert_eq!(
            value_for_pixel(&src, Point2d { x: 13.0, y: 8.0 }),
            Some(0.0)
        );

        assert_eq!(
            value_for_pixel(&src, Point2d { x: 26.0, y: 1.0 }),
            Some(255.0)
        );

        Ok(())
    }

    #[test]
    fn test_matrix_3x3_cholesky() {
        let a = [[6.0, 15.0, 55.0], [15.0, 55.0, 225.0], [55.0, 225.0, 979.0]];
        let mut r = [0.0; 9];

        matrix_3x3_cholesky(&a, &mut r);
        let expected_r = [
            2.4495, 0.0, 0.0, 6.1237, 4.1833, 0.0, 22.4537, 20.9165, 6.1101,
        ];

        for i in 0..9 {
            assert!((r[i] - expected_r[i]).abs() < 0.0001);
        }
    }

    #[test]
    fn test_matrix_3x3_lower_triange_inverse() {
        let a = [5.0, 0.0, 0.0, 3.0, 4.0, 0.0, 1.0, 2.0, 6.0];
        let mut r = [0.0; 9];

        matrix_3x3_lower_triange_inverse(&a, &mut r);
        let expected_r = [0.2, 0.0, 0.0, -0.15, 0.25, 0.0, 0.01666, -0.08333, 0.16666];

        for i in 0..9 {
            assert!((r[i] - expected_r[i]).abs() < 0.0001)
        }
    }
}
