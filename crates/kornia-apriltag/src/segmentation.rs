use crate::{errors::AprilTagError, union_find::UnionFind, utils::Pixel};
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
            uf.union(i, left_i);
        }

        // Check top neighbor
        if row_y > 0 {
            let top_i = i - src_size.width;
            if *pixel == src_data[top_i] {
                uf.union(i, top_i);
            }

            if *pixel == Pixel::White {
                // Check top-left neighbor
                let top_left_i = top_i - 1;
                if (row_x == 1
                    || !(src_data[top_left_i] == src_data[left_i]
                        || src_data[top_left_i] == src_data[top_i]))
                    && *pixel == src_data[top_left_i]
                {
                    uf.union(i, top_left_i);
                }

                // Check top-right neighbor
                let top_right_i = top_i + 1;
                if src_data[top_i] != src_data[top_right_i] && *pixel == src_data[top_right_i] {
                    uf.union(i, top_right_i);
                }
            }
        }
    });

    Ok(())
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
            Black, Black, White, White,
            Black, Black, White, White,
            White, White, Black, Black,
            White, White, Black, Black
        ];

        let bin = Image::new(
            ImageSize {
                width: 4,
                height: 4,
            },
            bin_data,
            CpuAllocator,
        )?;

        let mut uf = UnionFind::new(16);
        find_connected_components(&bin, &mut uf)?;

        assert_eq!(uf.get_representative(0), 0);
        assert_eq!(uf.get_representative(1), 0);
        assert_eq!(uf.get_representative(2), 2);
        assert_eq!(uf.get_representative(3), 3);
        assert_eq!(uf.get_representative(4), 0);
        assert_eq!(uf.get_representative(5), 0);
        assert_eq!(uf.get_representative(6), 2);
        assert_eq!(uf.get_representative(7), 7);
        assert_eq!(uf.get_representative(8), 2);
        assert_eq!(uf.get_representative(9), 2);
        assert_eq!(uf.get_representative(10), 10);
        assert_eq!(uf.get_representative(11), 11);
        assert_eq!(uf.get_representative(12), 2);
        assert_eq!(uf.get_representative(13), 2);
        assert_eq!(uf.get_representative(14), 10);
        assert_eq!(uf.get_representative(15), 15);

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
}
