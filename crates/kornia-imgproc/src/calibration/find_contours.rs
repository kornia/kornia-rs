use kornia_image::Image;
use num::Num;
use std::ops::{Add, Sub};

/// Specification for Border Type.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BorderType {
    /// Perimeter of foreground regions
    Outer,
    /// Perimeter of background regions enclosed by foreground.
    Hole,
}

/// A 2-dimensional point.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Point<T> {
    /// x-coordinate.
    pub x: T,
    /// y-coordinate.
    pub y: T,
}

impl<T> Point<T> {
    /// Construct a point at (x, y).
    pub fn new(x: T, y: T) -> Point<T> {
        Point::<T> { x, y }
    }
}

impl<T: Num> Add for Point<T> {
    type Output = Self;

    fn add(self, other: Point<T>) -> Point<T> {
        Point::new(self.x + other.x, self.y + other.y)
    }
}

impl<T: Num> Sub for Point<T> {
    type Output = Self;

    fn sub(self, other: Point<T>) -> Point<T> {
        Point::new(self.x - other.x, self.y - other.y)
    }
}

/// The border for any region.
#[derive(Debug, Clone)]
pub struct Contour<T> {
    /// The points on the border.
    pub points: Vec<Point<T>>,
    /// The type of the border. Outer or Hole.
    pub border_type: BorderType,
    /// The parent of the border.
    pub parent: Option<usize>,
}

impl<T> Contour<T> {
    /// Constructor for contour.
    pub fn new(points: Vec<Point<T>>, border_type: BorderType, parent: Option<usize>) -> Self {
        Contour {
            points,
            border_type,
            parent,
        }
    }
}

/// Finds the borders of the foreground regions of the image. All pixels with
/// intensity greater than `threshold` are treated as belonging to the foreground.
///
/// Code written based on algorithm proposed by Suzuki and Abe for border following.
///
/// # Arguments
///
/// * `src` - A reference to the source grayscale image (with `f32` pixel values).
/// * `threshold` - The threshold for treating a pixel as a foreground pixel.
///
/// # Returns
///
/// A vector array containing all the `Contour`s found in the input image.
///
pub fn find_contours<T>(src: &Image<u8, 1>, threshold: f32) -> Vec<Contour<T>>
where
    T: num::Num + num::NumCast + Copy + PartialEq + Eq,
{
    let width = src.width() as usize;
    let height = src.height() as usize;
    let mut image_values = vec![0i32; height * width];
    let mut contours: Vec<Contour<T>> = Vec::new();
    let mut nbd = 1i32; // new-border label

    let at = |x: usize, y: usize| x + y * width;

    // Convert image to binary based on threshold.
    for y in 0..height {
        for x in 0..width {
            if let Ok(pixel_value) = src.get_pixel(x, y, 0) {
                if *pixel_value as f32 > threshold {
                    image_values[at(x, y)] = 1;
                }
            }
        }
    }

    // Direction offsets: E, SE, S, SW, W, NW, N, NE.
    let mut directions = std::collections::VecDeque::from(vec![
        Point::new(1, 0),   // East
        Point::new(1, 1),   // South-east
        Point::new(0, 1),   // South
        Point::new(-1, 1),  // South-west
        Point::new(-1, 0),  // West
        Point::new(-1, -1), // North-west
        Point::new(0, -1),  // North
        Point::new(1, -1),  // North-east
    ]);

    let get_position_if_non_zero_pixel = |image: &[i32], curr: Point<i32>| {
        let (x, y) = (curr.x, curr.y);
        let in_bounds = x > -1 && x < width as i32 && y > -1 && y < height as i32;

        if in_bounds && image[at(x as usize, y as usize)] != 0 {
            Some(Point::new(x as usize, y as usize))
        } else {
            None
        }
    };

    fn rotate_to_value<U: PartialEq + Copy>(values: &mut std::collections::VecDeque<U>, value: U) {
        if let Some(rotate_pos) = values.iter().position(|x| *x == value) {
            values.rotate_left(rotate_pos);
        }
    }

    for y in 0..height {
        let mut lnbd = 0i32;

        for x in 0..width {
            // Only consider pixels that are foreground.
            if image_values[at(x, y)] <= 0 {
                continue;
            }

            // Determine border type and parent.
            let maybe_border =
                if image_values[at(x, y)] == 1 && x > 0 && image_values[at(x - 1, y)] == 0 {
                    Some((Point::new(x - 1, y), BorderType::Outer))
                } else if image_values[at(x, y)] == 1
                    && x + 1 < width
                    && image_values[at(x + 1, y)] == 0
                {
                    lnbd = nbd;
                    Some((Point::new(x + 1, y), BorderType::Hole))
                } else {
                    None
                };

            if let Some((adj, border_type)) = maybe_border {
                nbd += 1;

                let parent = if lnbd.abs() >= 2 {
                    let parent_index = (lnbd.abs() - 2) as usize;
                    let parent_contour = &contours[parent_index];
                    if (border_type == BorderType::Outer)
                        ^ (parent_contour.border_type == BorderType::Outer)
                    {
                        Some(parent_index)
                    } else {
                        parent_contour.parent
                    }
                } else {
                    None
                };

                let mut contour_points = Vec::new();
                let curr = Point::new(x, y);
                rotate_to_value(
                    &mut directions,
                    Point::new(adj.x as i32 - curr.x as i32, adj.y as i32 - curr.y as i32),
                );

                if let Some(pos1) = directions.iter().find_map(|diff| {
                    get_position_if_non_zero_pixel(
                        &image_values,
                        Point::new(curr.x as i32 + diff.x, curr.y as i32 + diff.y),
                    )
                }) {
                    let mut pos2 = pos1;
                    let mut pos3 = curr;

                    loop {
                        contour_points.push(Point::new(
                            num::cast(pos3.x).unwrap(),
                            num::cast(pos3.y).unwrap(),
                        ));

                        rotate_to_value(
                            &mut directions,
                            Point::new(
                                pos2.x as i32 - pos3.x as i32,
                                pos2.y as i32 - pos3.y as i32,
                            ),
                        );

                        let pos4 = directions
                            .iter()
                            .rev()
                            .find_map(|diff| {
                                get_position_if_non_zero_pixel(
                                    &image_values,
                                    Point::new(pos3.x as i32 + diff.x, pos3.y as i32 + diff.y),
                                )
                            })
                            .unwrap();

                        let mut is_right_edge = false;
                        for diff in directions.iter().rev() {
                            if *diff
                                == Point::new(
                                    pos4.x as i32 - pos3.x as i32,
                                    pos4.y as i32 - pos3.y as i32,
                                )
                            {
                                break;
                            }
                            if *diff == Point::new(1, 0) {
                                is_right_edge = true;
                                break;
                            }
                        }

                        if pos3.x + 1 == width || is_right_edge {
                            image_values[at(pos3.x, pos3.y)] = -nbd;
                        } else if image_values[at(pos3.x, pos3.y)] == 1 {
                            image_values[at(pos3.x, pos3.y)] = nbd;
                        }

                        if pos4 == curr && pos3 == pos1 {
                            break;
                        }
                        pos2 = pos3;
                        pos3 = pos4;
                    }
                } else {
                    // Single pixel contour.
                    contour_points.push(Point::new(num::cast(x).unwrap(), num::cast(y).unwrap()));
                    image_values[at(x, y)] = -nbd;
                }

                // if hole border, reverse points to make counter clockwise.
                if border_type == BorderType::Hole {
                    contour_points.reverse();
                }

                contours.push(Contour::new(contour_points, border_type, parent));
            }

            if image_values[at(x, y)] != 1 {
                lnbd = nbd.abs();
            }
        }
    }
    contours
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::{Image, ImageError, ImageSize};

    fn create_test_image_basic() -> Result<Image<u8, 1>, ImageError> {
        let mut img = Image::new(
            ImageSize {
                width: 10,
                height: 10,
            },
            vec![0; 10 * 10],
        )?;

        for y in 2..=7 {
            for x in 2..=7 {
                img.set_pixel(x, y, 0, 255).unwrap();
            }
        }

        for y in 4..=5 {
            for x in 4..=5 {
                img.set_pixel(x, y, 0, 0).unwrap();
            }
        }

        Ok(img)
    }

    fn create_test_image_nested() -> Result<Image<u8, 1>, ImageError> {
        let mut img = Image::new(
            ImageSize {
                width: 20,
                height: 20,
            },
            vec![0; 20 * 20],
        )?;

        for y in 2..=17 {
            for x in 2..=17 {
                img.set_pixel(x, y, 0, 255).unwrap();
            }
        }

        for y in 5..=14 {
            for x in 5..=14 {
                img.set_pixel(x, y, 0, 0).unwrap();
            }
        }

        for y in 8..=11 {
            for x in 8..=11 {
                img.set_pixel(x, y, 0, 255).unwrap();
            }
        }

        Ok(img)
    }

    #[test]
    fn test_basic_contours() {
        let img = create_test_image_basic().unwrap();
        let contours = find_contours::<i32>(&img, 0.5);

        assert_eq!(contours.len(), 2);

        // Outer contour
        let outer = &contours[0];
        assert_eq!(outer.border_type, BorderType::Outer);
        assert!(outer.points.contains(&Point::new(2, 2)));
        assert!(outer.points.contains(&Point::new(7, 2)));
        assert!(outer.points.contains(&Point::new(7, 7)));
        assert!(outer.points.contains(&Point::new(2, 7)));

        // Hole contour
        let hole = &contours[1];
        assert_eq!(hole.border_type, BorderType::Hole);
        assert_eq!(hole.parent, Some(0));
        assert!(hole.points.contains(&Point::new(3, 4)));
        assert!(hole.points.contains(&Point::new(6, 4)));
        assert!(hole.points.contains(&Point::new(5, 6)));
        assert!(hole.points.contains(&Point::new(5, 3)));
    }

    #[test]
    fn test_single_pixel() -> Result<(), ImageError> {
        let mut img = Image::new(
            ImageSize {
                width: 5,
                height: 5,
            },
            vec![0; 5 * 5],
        )?;
        img.set_pixel(2, 2, 0, 255).unwrap();

        let contours = find_contours::<i32>(&img, 0.5);
        assert_eq!(contours.len(), 1);
        assert_eq!(contours[0].points.len(), 1);
        assert_eq!(contours[0].points[0], Point::new(2, 2));
        assert_eq!(contours[0].border_type, BorderType::Outer);

        Ok(())
    }

    #[test]
    fn test_nested_contours() -> Result<(), ImageError> {
        let img = create_test_image_nested().unwrap();

        let contours = find_contours::<i32>(&img, 0.5);
        assert_eq!(contours.len(), 3);

        // Checking if hierarchy holds.
        // Outer contour
        assert_eq!(contours[0].border_type, BorderType::Outer);
        // Middle contour (hole)
        assert_eq!(contours[1].border_type, BorderType::Hole);
        assert_eq!(contours[1].parent, Some(0));
        // Inner contour
        assert_eq!(contours[2].border_type, BorderType::Outer);
        assert_eq!(contours[2].parent, Some(1));
        Ok(())
    }
}
