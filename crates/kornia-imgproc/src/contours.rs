use kornia_image::Image;
use std::collections::{HashMap, VecDeque};

/// Find contours in a binary image.
pub fn find_contours(src: &Image<u8, 1>) -> Vec<Vec<(f64, f64)>> {
    // Convert u8 image to f64 and normalize to 0-1
    let src_f64 = src
        .as_slice()
        .iter()
        .map(|&x| x as f64 / 255.0)
        .collect::<Vec<_>>();

    let src_f64 = Image::new(src.size().clone(), src_f64).unwrap();

    // Get contour segments using marching squares
    let segments = get_contour_segments(&src_f64, 0.5, false);
    println!("{:?}", segments);

    // Convert segments to contours
    // This part needs to be implemented to connect segments into continuous contours
    // and convert the f64 coordinates to i32
    //let contours = assemble_contours(segments);
    let mut contours = Vec::new();

    contours
}

fn assemble_contours(segments: Vec<(Point, Point)>) -> Vec<Vec<(i32, i32)>> {
    let mut current_index = 0;
    let mut contours: HashMap<usize, VecDeque<Point>> = HashMap::new();
    let mut starts: HashMap<Point, (VecDeque<Point>, usize)> = HashMap::new();
    let mut ends: HashMap<Point, (VecDeque<Point>, usize)> = HashMap::new();

    for (from_point, to_point) in segments {
        // Ignore degenerate segments
        if (from_point.0 - to_point.0).abs() < 0 && (from_point.1 - to_point.1).abs() < 0 {
            continue;
        }

        let tail = starts.remove(&to_point);
        let head = ends.remove(&from_point);

        match (tail, head) {
            (Some((mut tail_deque, tail_num)), Some((mut head_deque, head_num))) => {
                // Need to connect two contours
                if std::ptr::eq(&tail_deque, &head_deque) {
                    // Close the contour
                    head_deque.push_back(to_point);
                    starts.insert(*head_deque.front().unwrap(), (head_deque.clone(), head_num));
                    ends.insert(*head_deque.back().unwrap(), (head_deque, head_num));
                } else {
                    // Join two distinct contours
                    if tail_num > head_num {
                        // Append tail to head
                        head_deque.extend(tail_deque.iter());
                        contours.remove(&tail_num);
                        starts.insert(*head_deque.front().unwrap(), (head_deque.clone(), head_num));
                        ends.insert(*head_deque.back().unwrap(), (head_deque, head_num));
                    } else {
                        // Prepend head to tail
                        let mut new_deque = VecDeque::new();
                        new_deque.extend(head_deque.iter().rev());
                        new_deque.extend(tail_deque.iter());
                        contours.remove(&head_num);
                        contours.insert(tail_num, new_deque.clone());
                        starts.insert(*new_deque.front().unwrap(), (new_deque.clone(), tail_num));
                        ends.insert(*new_deque.back().unwrap(), (new_deque, tail_num));
                    }
                }
            }
            (None, None) => {
                // Add new contour
                let mut new_contour = VecDeque::new();
                new_contour.push_back(from_point);
                new_contour.push_back(to_point);
                contours.insert(current_index, new_contour.clone());
                starts.insert(from_point, (new_contour.clone(), current_index));
                ends.insert(to_point, (new_contour, current_index));
                current_index += 1;
            }
            (Some((mut tail_deque, tail_num)), None) => {
                // Prepend to existing tail
                tail_deque.push_front(from_point);
                starts.insert(from_point, (tail_deque.clone(), tail_num));
                ends.insert(*tail_deque.back().unwrap(), (tail_deque, tail_num));
            }
            (None, Some((mut head_deque, head_num))) => {
                // Append to existing head
                head_deque.push_back(to_point);
                starts.insert(*head_deque.front().unwrap(), (head_deque.clone(), head_num));
                ends.insert(to_point, (head_deque, head_num));
            }
        }
    }

    // Convert to final format and sort by index
    let mut result: Vec<Vec<(i32, i32)>> = contours
        .into_iter()
        .map(|(_, contour)| {
            contour
                .into_iter()
                .map(|p| (p.0 as i32, p.1 as i32))
                .collect()
        })
        .collect();
    result.sort_by_key(|_| current_index);

    result
}

/// Compress a contour using the chain approxmation algorithm.
fn chain_approx_simple(contour: &Vec<(i32, i32)>) -> Vec<(i32, i32)> {
    let mut simplified = Vec::new();

    // add the first point
    if let Some((px, py)) = contour.first() {
        simplified.push((*px, *py));
    }

    // iterate over the contour and skip points that are close to the previous point
    for i in 1..contour.len() - 1 {
        let (x1, y1) = contour[i - 1];
        let (x2, y2) = contour[i];
        let (x3, y3) = contour[i + 1];

        println!("{} {} {} {} {} {}", x1, y1, x2, y2, x3, y3);

        // check if the point is close is in a straight line
        if (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2) != 0 {
            simplified.push((x2, y2));
        }
    }

    // add the last point
    if let Some((px, py)) = contour.last() {
        simplified.push((*px, *py));
    }

    simplified
}
/// A point in 2D space represented by integer coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Point(i64, i64);

/// Calculates the fractional position between two values where a level threshold is crossed.
///
/// # Arguments
/// * `from_value` - The starting value
/// * `to_value` - The ending value
/// * `level` - The threshold level to find the crossing point for
///
/// # Returns
/// A fraction between 0 and 1 indicating where the level threshold is crossed
fn get_fraction(from_value: f64, to_value: f64, level: f64) -> f64 {
    if to_value == from_value {
        return 0.0;
    }
    (level - from_value) / (to_value - from_value)
}

/// Returns line segments that make up contours at a given threshold level in an image.
///
/// # Arguments
/// * `src` - Input single-channel image
/// * `level` - Threshold level to find contours at
/// * `vertex_connect_high` - Whether to connect vertices for values above the threshold (true) or below (false)
///
/// # Returns
/// Vector of line segments making up the contours, represented as pairs of Points
pub fn get_contour_segments(
    src: &Image<f64, 1>,
    level: f64,
    vertex_connect_high: bool,
) -> Vec<((f64, f64), (f64, f64))> {
    let mut segments = Vec::new();

    for r0 in 0..src.rows() - 1 {
        for c0 in 0..src.cols() - 1 {
            let r1 = r0 + 1;
            let c1 = c0 + 1;

            let ul = src.as_slice()[r0 * src.cols() + c0];
            let ur = src.as_slice()[r0 * src.cols() + c1];
            let ll = src.as_slice()[r1 * src.cols() + c0];
            let lr = src.as_slice()[r1 * src.cols() + c1];

            // Skip if any values are NaN
            if ul.is_nan() || ur.is_nan() || ll.is_nan() || lr.is_nan() {
                continue;
            }

            let mut square_case = 0;
            if ul > level {
                square_case += 1;
            }
            if ur > level {
                square_case += 2;
            }
            if ll > level {
                square_case += 4;
            }
            if lr > level {
                square_case += 8;
            }

            // Skip cases with no lines
            if square_case == 0 || square_case == 15 {
                continue;
            }

            // Calculate intersection points
            let top = (r0 as f64, c0 as f64 + get_fraction(ul, ur, level));
            let bottom = (r1 as f64, c0 as f64 + get_fraction(ll, lr, level));
            let left = (r0 as f64 + get_fraction(ul, ll, level), c0 as f64);
            let right = (r0 as f64 + get_fraction(ur, lr, level), c1 as f64);

            match square_case {
                1 => segments.push((top, left)),
                2 => segments.push((right, top)),
                3 => segments.push((right, left)),
                4 => segments.push((left, bottom)),
                5 => segments.push((top, bottom)),
                6 => {
                    if vertex_connect_high {
                        segments.push((left, top));
                        segments.push((right, bottom));
                    } else {
                        segments.push((right, top));
                        segments.push((left, bottom));
                    }
                }
                7 => segments.push((right, bottom)),
                8 => segments.push((bottom, right)),
                9 => {
                    if vertex_connect_high {
                        segments.push((top, right));
                        segments.push((bottom, left));
                    } else {
                        segments.push((top, left));
                        segments.push((bottom, right));
                    }
                }
                10 => segments.push((bottom, top)),
                11 => segments.push((bottom, left)),
                12 => segments.push((left, right)),
                13 => segments.push((top, right)),
                14 => segments.push((left, top)),
                _ => {}
            }
        }
    }

    segments
}

#[cfg(test)]
mod tests {
    use kornia_image::{Image, ImageError, ImageSize};

    use super::*;

    #[test]
    fn test_find_contours() -> Result<(), ImageError> {
        #[rustfmt::skip]
        let src = Image::<u8, 1>::new(
            ImageSize {
                width: 5,
                height: 5,
            },
            vec![
                0, 0, 0, 0, 0, //
                0, 255, 255, 255, 0, //
                0, 255, 255, 255, 0, //
                0, 255, 255, 255, 0, //
                0, 0, 0, 0, 0, //
            ],
        )?;
        let contours = find_contours(&src);

        println!("{:?}", contours);
        assert_eq!(contours.len(), 1);

        Ok(())
    }
}
