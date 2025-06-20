use crate::union_find::UnionFind;
use crate::utils::Pixel;
use kornia_image::{allocator::ImageAllocator, Image};

/// Finds connected components in a binary image using union-find.
///
/// # Arguments
///
/// * `src` - A reference to a binary image where each pixel is of type `Pixel`.
///
/// # Returns
///
/// A `QuickFindUf<UnionBySize>` structure representing the connected components.
pub fn find_connected_components<A: ImageAllocator>(src: &Image<Pixel, 1, A>) -> UnionFind {
    let src_size = src.size();
    let src_data = src.as_slice();
    let src_len = src_data.len();

    let mut uf = UnionFind::new(src_len);

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
        if row_x > 0 {
            let left_i = i - 1;
            if *pixel == src_data[left_i] {
                uf.union(i, left_i);
            }
        }

        // Check top neighbor
        if row_y > 0 {
            let top_i = i - src_size.width;
            if *pixel == src_data[top_i] {
                uf.union(i, top_i);
            }

            if *pixel == Pixel::White {
                // Check top-left neighbor
                if row_x > 0 && row_y > 0 {
                    let top_left_i = top_i - 1;
                    if (row_x == 1
                        || !(src_data[top_left_i] == src_data[i - 1]
                            || src_data[top_left_i] == src_data[top_i]))
                        && *pixel == src_data[top_left_i]
                    {
                        uf.union(i, top_left_i);
                    }
                }

                // Check top-right neighbor
                if row_x < src_size.width - 1 && row_y > 0 {
                    let top_right_i = top_i + 1;
                    if !(src_data[top_i] == src_data[top_right_i])
                        && *pixel == src_data[top_right_i]
                    {
                        uf.union(i, top_right_i);
                    }
                }
            }
        }
    });

    uf
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::threshold::{adaptive_threshold, TileMinMax};
    use kornia_image::allocator::CpuAllocator;
    use kornia_io::png::read_image_png_mono8;

    #[test]
    fn test_segmentation() -> Result<(), Box<dyn std::error::Error>> {
        let src = read_image_png_mono8("../../tests/data/apriltag.png")?;
        let mut bin = Image::from_size_val(src.size(), Pixel::Skip, CpuAllocator)?;

        let mut tile_min_max = TileMinMax::new(src.size(), 4);
        adaptive_threshold(&src, &mut bin, &mut tile_min_max, 20)?;

        let mut uf = find_connected_components(&bin);

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
