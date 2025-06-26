use std::{collections::HashMap, ops::Mul};

use crate::{
    errors::AprilTagError,
    union_find::UnionFind,
    utils::{Pixel, Point2d},
};
use kornia_image::{allocator::ImageAllocator, Image};

/// Finds connected components in a binary image using union-find.
///
/// # Arguments
///
/// * `src` - Reference to the source image containing `Pixel` values.
/// * `uf` - Mutable reference to a [`UnionFind`] structure for tracking connected components.
///   Make sure to call [`UnionFind::reset`] if you are using this function multiple times with the same `uf`.
///
/// # Returns
///
/// * `Result<(), AprilTagError>` - Returns `Ok(())` if successful, or an error if the union-find size is invalid.
pub fn find_connected_components<A: ImageAllocator>(
    src: &Image<Pixel, 1, A>,
    uf: &mut UnionFind,
) -> Result<(), AprilTagError> {
    let src_size = src.size();
    let src_data = src.as_slice();
    let src_len = src_data.len();

    if src_len != uf.len() {
        return Err(AprilTagError::InvalidUnionFindSize(src_len, uf.len()));
    }

    src_data.iter().enumerate().for_each(|(i, pixel)| {
        if *pixel == Pixel::Skip {
            return;
        }

        let row_y = i / src_size.width;
        let row_x = i % src_size.width;

        if row_x == 0 || row_x == src_size.width - 1 {
            return; // Skip boundary pixels
        }

        // Check left neighbor
        let left_i = i - 1;
        if *pixel == src_data[left_i] {
            uf.connect(i, left_i);
        }

        // Check top neighbor
        if row_y > 0 {
            let top_i = i - src_size.width;
            if *pixel == src_data[top_i] {
                uf.connect(i, top_i);
            }

            if *pixel == Pixel::White {
                // Check top-left neighbor
                let top_left_i = top_i - 1;
                if (row_x == 1
                    || !(src_data[top_left_i] == src_data[left_i]
                        || src_data[top_left_i] == src_data[top_i]))
                    && *pixel == src_data[top_left_i]
                {
                    uf.connect(i, top_left_i);
                }

                // Check top-right neighbor
                let top_right_i = top_i + 1;
                if src_data[top_i] != src_data[top_right_i] && *pixel == src_data[top_right_i] {
                    uf.connect(i, top_right_i);
                }
            }
        }
    });

    Ok(())
}

/// Information about the gradient at a specific pixel location.
#[derive(Debug, Clone, Copy)]
pub struct GradientInfo {
    /// The coordinates of the pixel, represented as the mid-point assuming twice the size of the image.
    pub pos: Point2d,
    /// The gradient direction in the x-axis.
    pub gx: GradientDirection,
    /// The gradient direction in the y-axis.
    pub gy: GradientDirection,
}

/// Represents the direction of a gradient between two pixels.
///
/// Used to indicate whether the gradient is towards a white pixel, towards a black pixel, or if there is no gradient.
#[derive(Debug, Clone, Copy)]
pub enum GradientDirection {
    /// Gradient is towards a white pixel (value 255).
    TowardsWhite,
    /// Gradient is towards a black pixel (value -255).
    TowardsBlack,
    /// No gradient (value 0).
    None,
}

impl Mul<isize> for GradientDirection {
    type Output = GradientDirection;

    fn mul(self, rhs: isize) -> Self::Output {
        match rhs.cmp(&0) {
            std::cmp::Ordering::Equal => GradientDirection::None,
            std::cmp::Ordering::Greater => self,
            std::cmp::Ordering::Less => match self {
                GradientDirection::TowardsWhite => GradientDirection::TowardsBlack,
                GradientDirection::TowardsBlack => GradientDirection::TowardsWhite,
                _ => GradientDirection::None,
            },
        }
    }
}

impl Pixel {
    /// Computes the gradient direction between two pixels.
    ///
    /// # Arguments
    /// * `other` - The pixel to compare against.
    ///
    /// # Returns
    /// A `GradientDirection` indicating the direction of the gradient.
    pub fn gradient_to(&self, other: Pixel) -> GradientDirection {
        match (self, other) {
            (Pixel::Black, Pixel::White) => GradientDirection::TowardsBlack,
            (Pixel::White, Pixel::Black) => GradientDirection::TowardsWhite,
            _ => GradientDirection::None,
        }
    }
}

/// Finds and groups gradient transitions between connected components in a binary image.
///
/// For each pixel, this function checks its neighbors and, if the neighbor belongs to a different
/// connected component (with sufficient size), records the gradient information between the two components.
/// The results are stored in the `clusters` map, keyed by the pair of component representatives.
///
/// # Arguments
///
/// * `src` - Reference to the source image containing `Pixel` values.
/// * `uf` - Mutable reference to a [`UnionFind`] structure for tracking connected components.
/// * `clusters` - Mutable reference to a map where the gradient information between component pairs will be stored.
///   Make sure to call [`HashMap::clear`] if you are using this function multiple times with the same `clusters`
pub fn find_gradient_clusters<A: ImageAllocator>(
    src: &Image<Pixel, 1, A>,
    uf: &mut UnionFind,
    clusters: &mut HashMap<(usize, usize), Vec<GradientInfo>>,
) {
    let src_slice = src.as_slice();

    (1..src.height() - 1).for_each(|y| {
        let mut connected_last = false;

        (1..src.width() - 1).for_each(|x| {
            let i = y * src.width() + x;
            let current_pixel = src_slice[i];

            if current_pixel == Pixel::Skip {
                connected_last = false;
                return;
            }

            let current_pixel_representative = uf.get_representative(i);

            // Ignore components smaller than 25 pixels to filter out noise and very small regions.
            if uf.get_set_size(current_pixel_representative) < 25 {
                connected_last = false;
                return;
            }

            let mut any_connected = false;
            let mut do_conn =
                |dx: isize, dy: isize, neighbor_i: usize, any_connected: &mut bool| {
                    let neighbor_pixel = src_slice[neighbor_i];
                    if neighbor_pixel == Pixel::Skip {
                        return;
                    }

                    if current_pixel != neighbor_pixel {
                        let neighbor_pixel_representative = uf.get_representative(neighbor_i);

                        // Ignore components smaller than 25 pixels to filter out noise and very small regions.
                        if uf.get_set_size(neighbor_pixel_representative) < 25 {
                            return;
                        }

                        let key = if current_pixel_representative < neighbor_pixel_representative {
                            (current_pixel_representative, neighbor_pixel_representative)
                        } else {
                            (neighbor_pixel_representative, current_pixel_representative)
                        };

                        let entry = clusters.entry(key).or_default();

                        let delta = neighbor_pixel.gradient_to(current_pixel);
                        let gradient_info = GradientInfo {
                            pos: Point2d {
                                x: (2 * x as isize + dx) as usize,
                                y: (2 * y as isize + dy) as usize,
                            },
                            gx: delta * dx,
                            gy: delta * dy,
                        };

                        entry.push(gradient_info);
                        *any_connected = true;
                    }
                };

            do_conn(1, 0, i + 1, &mut any_connected);
            do_conn(0, 1, i + src.width(), &mut any_connected);

            if !connected_last {
                do_conn(-1, 1, i + src.width() - 1, &mut any_connected)
            }

            any_connected = false;

            do_conn(1, 1, i + src.width() + 1, &mut any_connected);

            connected_last = any_connected;
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::threshold::{adaptive_threshold, TileMinMax};
    use kornia_image::{allocator::CpuAllocator, ImageSize};
    use kornia_io::png::read_image_png_mono8;

    #[test]
    fn test_basic_segmentation() -> Result<(), Box<dyn std::error::Error>> {
        use Pixel::*;
        #[rustfmt::skip]
        let bin_data = vec![
            Black, Black, White, White, White,
            Black, Black, White, White, White,
            White, White, Black, Black, Black,
            White, White, Black, Black, Black
        ];

        let bin = Image::new(
            ImageSize {
                width: 5,
                height: 4,
            },
            bin_data,
            CpuAllocator,
        )?;

        let mut uf = UnionFind::new(20);
        find_connected_components(&bin, &mut uf)?;

        assert_eq!(uf.get_representative(0), 0);
        assert_eq!(uf.get_representative(1), 0);
        assert_eq!(uf.get_representative(2), 2);
        assert_eq!(uf.get_representative(3), 2);
        assert_eq!(uf.get_representative(4), 4);
        assert_eq!(uf.get_representative(5), 0);
        assert_eq!(uf.get_representative(6), 0);
        assert_eq!(uf.get_representative(7), 2);
        assert_eq!(uf.get_representative(8), 2);
        assert_eq!(uf.get_representative(9), 9);
        assert_eq!(uf.get_representative(10), 2);
        assert_eq!(uf.get_representative(11), 2);
        assert_eq!(uf.get_representative(12), 12);
        assert_eq!(uf.get_representative(13), 12);
        assert_eq!(uf.get_representative(14), 14);
        assert_eq!(uf.get_representative(15), 2);
        assert_eq!(uf.get_representative(16), 2);
        assert_eq!(uf.get_representative(17), 12);
        assert_eq!(uf.get_representative(18), 12);
        assert_eq!(uf.get_representative(19), 19);

        Ok(())
    }

    #[test]
    fn test_segmentation() -> Result<(), Box<dyn std::error::Error>> {
        let src = read_image_png_mono8("../../tests/data/apriltag.png")?;
        let mut bin = Image::from_size_val(src.size(), Pixel::Skip, CpuAllocator)?;

        let mut tile_min_max = TileMinMax::new(src.size(), 4);
        adaptive_threshold(&src, &mut bin, &mut tile_min_max, 20)?;

        let mut uf = UnionFind::new(bin.as_slice().len());
        find_connected_components(&bin, &mut uf)?;

        let mut union_representatives = String::new();
        let expected =
            std::fs::read_to_string("../../tests/data/apriltag_pixel_representatives.txt")?;

        for i in 0..bin.as_slice().len() {
            let representative = uf.get_representative(i).to_string();

            union_representatives.push_str(&representative);
            union_representatives.push(' ');
        }

        assert_eq!(union_representatives, expected);

        Ok(())
    }

    #[test]
    fn test_gradient_clusters() -> Result<(), Box<dyn std::error::Error>> {
        let src = read_image_png_mono8("../../tests/data/apriltag.png")?;
        let mut bin = Image::from_size_val(src.size(), Pixel::Skip, CpuAllocator)?;

        let mut tile_min_max = TileMinMax::new(src.size(), 4);
        adaptive_threshold(&src, &mut bin, &mut tile_min_max, 20)?;

        let mut uf = UnionFind::new(bin.as_slice().len());
        find_connected_components(&bin, &mut uf)?;

        let mut gradient_clusters = HashMap::new();
        find_gradient_clusters(&bin, &mut uf, &mut gradient_clusters);

        // Since the order of HashMap iteration is random, we cannot rely on the order of clusters.
        // However, we know from the expected data file that there are exactly 3 unique clusters,
        // each with a distinct length: 48, 188, and 192. We match clusters by their length and
        // compare their string representations to the expected output for each size.
        let expected = std::fs::read_to_string("../../tests/data/apriltag_gradient_clusters.txt")?;
        let mut expected_len_48 = String::new();
        let mut expected_len_188 = String::new();
        let mut expected_len_192 = String::new();

        for line in expected.lines() {
            if line.starts_with("size 48:") {
                expected_len_48 = line.to_string();
            } else if line.starts_with("size 188:") {
                expected_len_188 = line.to_string();
            } else if line.starts_with("size 192:") {
                expected_len_192 = line.to_string();
            }
        }

        for (_, infos) in gradient_clusters.iter() {
            let mut clusters = format!("size {}:\t", infos.len());

            for info in infos {
                let g_str = |g: GradientDirection| match g {
                    GradientDirection::None => 0,
                    GradientDirection::TowardsBlack => -255,
                    GradientDirection::TowardsWhite => 255,
                };
                clusters.push_str(
                    format!(
                        " (x={} y={} gx={} gy={})",
                        info.pos.x,
                        info.pos.y,
                        g_str(info.gx),
                        g_str(info.gy)
                    )
                    .as_str(),
                );
            }

            match infos.len() {
                48 => assert_eq!(expected_len_48, clusters),
                188 => assert_eq!(expected_len_188, clusters),
                192 => assert_eq!(expected_len_192, clusters),
                _ => panic!(
                    "Unexpected length of clusters, expected either 48, 188, or 192 but found {}",
                    infos.len()
                ),
            }
        }

        Ok(())
    }
}
