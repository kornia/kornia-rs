//! Fast corner detection implementations.
//!
//! This module provides an implementation of the FAST (Features from Accelerated Segment Test)
//! corner detection algorithm. FAST is an efficient corner detection algorithm that examines
//! pixels in a circle around a candidate point to determine if it's a corner.
//!
//! # Algorithm
//!
//! The FAST algorithm works by:
//! 1. Considering a circle of 16 pixels around the candidate point
//! 2. A pixel is classified as a corner if there exists a set of contiguous pixels in the circle
//!    which are all brighter than the candidate pixel by some threshold, or all darker by some threshold.
//!
//! # Variants
//!
//! This module implements two variants of FAST:
//! - **FAST-9**: A pixel is a corner if at least 9 contiguous pixels in the circle are all brighter
//!   or all darker than the candidate pixel by some threshold.
//! - **FAST-12**: A pixel is a corner if at least 12 contiguous pixels in the circle are all brighter
//!   or all darker than the candidate pixel by some threshold.
//!
//! # Corner Scoring
//!
//! The score of a corner is the maximum threshold for which the pixel is still considered a corner.
//! This provides a measure of the corner's "strength" or distinctiveness.
//!
//! # Performance Notes
//!
//! The current implementation prioritizes correctness over performance. Future optimizations
//! may include:
//! - Using precomputed pixel offsets
//! - Reducing unnecessary checks
//! - Optimizing the span search logic
//!
//! # References
//!
//! - Rosten, E., & Drummond, T. (2006). Machine learning for high-speed corner detection.
//!   In European Conference on Computer Vision (pp. 430-443).
use kornia_image::image::Image;
type GrayImage = Image<f32, 1>;

/// Represents a detected corner point in an image with its position and score.
/// This structure stores the (x, y) coordinates of a corner point along with a score
/// that typically indicates the corner's strength or reliability.
#[derive(Copy, Clone, Debug)]
pub struct Corner {
    /// The x-coordinate of the corner in the image.
    pub x: u32,
    /// The y-coordinate of the corner in the image.
    pub y: u32,
    /// A measure of the corner's strength or distinctiveness.
    pub score: f32,
}

impl Corner {
    /// Creates a new corner with the specified coordinates and score.
    pub fn new(x: u32, y: u32, score: f32) -> Self {
        Corner { x, y, score }
    }
}
/// Specifies which variant of the FAST corner detector to use.
/// 
/// The FAST algorithm classifies a pixel as a corner if there exists a contiguous arc
/// of pixels in a circle around the candidate point that are all brighter or all darker
/// than the center pixel by some threshold. The variants differ in the required length
/// of this arc.
///
/// # Variants
/// 
/// * `Nine` - FAST-9 algorithm that requires at least 9 contiguous pixels in the circle
///   to be all brighter or all darker than the center pixel by the threshold value.
///
/// * `Twelve` - FAST-12 algorithm that requires at least 12 contiguous pixels in the circle
///   to be all brighter or all darker than the center pixel by the threshold value.
///
/// This enum is used to select the appropriate algorithm variant when computing corner scores
/// or detecting corners.
pub enum Fast {
    /// Corners require a section of length as least nine.
    Nine,
    /// Corners require a section of length as least twelve.
    Twelve,
}

/// Finds corners using FAST-12 features. See comment on `Fast`.
pub fn corners_fast12(image: &GrayImage, threshold: u8) -> Vec<Corner> {
    let (width, height) = (image.width() as u32, image.height() as u32);
    let mut corners = vec![];

    for y in 0..height {
        for x in 0..width {
            if is_corner_fast12(image, threshold, x, y) {
                let score = fast_corner_score(image, threshold, x, y, Fast::Twelve);
                corners.push(Corner::new(x, y, score as f32));
            }
        }
    }

    corners
}

/// Finds corners using FAST-9 features. See comment on Fast enum.
pub fn corners_fast9(image: &GrayImage, threshold: u8) -> Vec<Corner> {
    let (width, height) = (image.width() as u32, image.height() as u32);
    let mut corners = vec![];

    for y in 0..height {
        for x in 0..width {
            if is_corner_fast9(image, threshold, x, y) {
                let score = fast_corner_score(image, threshold, x, y, Fast::Nine);
                corners.push(Corner::new(x, y, score as f32));
            }
        }
    }

    corners
}


/// The score of a corner detected using the FAST
/// detector is the largest threshold for which this
/// pixel is still a corner. We input the threshold at which
/// the corner was detected as a lower bound on the search.
/// Note that the corner check uses a strict inequality, so if
/// the smallest intensity difference between the center pixel
/// and a corner pixel is n then the corner will have a score of n - 1.
pub fn fast_corner_score(image: &GrayImage, threshold: u8, x: u32, y: u32, variant: Fast) -> u8 {
    let mut max = 255u8;
    let mut min = threshold;

    loop {
        if max == min {
            return max;
        }

        let mean = ((max as u16 + min as u16) / 2u16) as u8;
        let probe = if max == min + 1 { max } else { mean };

        let is_corner = match variant {
            Fast::Nine => is_corner_fast9(image, probe, x, y),
            Fast::Twelve => is_corner_fast12(image, probe, x, y),
        };

        if is_corner {
            min = probe;
        } else {
            max = probe - 1;
        }
    }
}

// Note [FAST circle labels]
//
//          15 00 01
//       14          02
//     13              03
//     12       p      04
//     11              05
//       10          06
//          09 08 07

/// Checks if the given pixel is a corner according to the FAST9 detector.
/// The current implementation is extremely inefficient.
// TODO: Make this much faster!
fn is_corner_fast9(image: &GrayImage, threshold: u8, x: u32, y: u32) -> bool {
    // UNSAFETY JUSTIFICATION
    //  Benefit
    //      Removing all unsafe pixel accesses in this file makes
    //      bench_is_corner_fast9_9_contiguous_lighter_pixels 60% slower, and
    //      bench_is_corner_fast12_12_noncontiguous 40% slower
    //  Correctness
    //      All pixel accesses in this function, and in the called get_circle,
    //      access pixels with x-coordinate in the range [x - 3, x + 3] and
    //      y-coordinate in the range [y - 3, y + 3]. The precondition below
    //      guarantees that these are within image bounds.
    let (width, height) = (image.width() as u32, image.height() as u32);
    if x >= u32::MAX - 3 || y >= u32::MAX - 3 || x < 3 || y < 3 || width <= x + 3 || height <= y + 3
    {
        return false;
    }

    // JUSTIFICATION - see comment at the start of this function
    let c = image.get_pixel(x as usize, y as usize, 1usize).unwrap().to_owned();
    let low_thresh: i16 = c as i16 - threshold as i16;
    let high_thresh: i16 = c as i16 + threshold as i16;

    // See Note [FAST circle labels]
    // JUSTIFICATION - see comment at the start of this function
    let (p0, p4, p8, p12) = (
        image.get_pixel(x as usize, (y - 3) as usize, 1usize).unwrap().to_owned() as i16,
        image.get_pixel(x as usize, (y + 3) as usize, 1usize).unwrap().to_owned() as i16,
        image.get_pixel((x + 3) as usize, y as usize, 1usize).unwrap().to_owned() as i16,
        image.get_pixel((x - 3) as usize, y as usize, 1usize).unwrap().to_owned() as i16,
    );

    let above = (p0 > high_thresh && p4 > high_thresh)
        || (p4 > high_thresh && p8 > high_thresh)
        || (p8 > high_thresh && p12 > high_thresh)
        || (p12 > high_thresh && p0 > high_thresh);

    let below = (p0 < low_thresh && p4 < low_thresh)
        || (p4 < low_thresh && p8 < low_thresh)
        || (p8 < low_thresh && p12 < low_thresh)
        || (p12 < low_thresh && p0 < low_thresh);

    if !above && !below {
        return false;
    }

    // JUSTIFICATION - see comment at the start of this function
    let pixels = get_circle(image, x, y, p0, p4, p8, p12);

    // above and below could both be true
    (above && has_bright_span(&pixels, 9, high_thresh))
        || (below && has_dark_span(&pixels, 9, low_thresh))
}

/// Checks if the given pixel is a corner according to the FAST12 detector.
fn is_corner_fast12(image: &GrayImage, threshold: u8, x: u32, y: u32) -> bool {
    // UNSAFETY JUSTIFICATION
    //  Benefit
    //      Removing all unsafe pixel accesses in this file makes
    //      bench_is_corner_fast9_9_contiguous_lighter_pixels 60% slower, and
    //      bench_is_corner_fast12_12_noncontiguous 40% slower
    //  Correctness
    //      All pixel accesses in this function, and in the called get_circle,
    //      access pixels with x-coordinate in the range [x - 3, x + 3] and
    //      y-coordinate in the range [y - 3, y + 3]. The precondition below
    //      guarantees that these are within image bounds.
    let (width, height) = (image.width() as u32, image.height() as u32);
    if x >= u32::MAX - 3 || y >= u32::MAX - 3 || x < 3 || y < 3 || width <= x + 3 || height <= y + 3
    {
        return false;
    }

    // JUSTIFICATION - see comment at the start of this function
    let c = image.get_pixel(x as usize, y as usize, 1usize).unwrap().to_owned();
    let low_thresh: i16 = c as i16 - threshold as i16;
    let high_thresh: i16 = c as i16 + threshold as i16;

    // See Note [FAST circle labels]
    // JUSTIFICATION - see comment at the start of this function
    let (p0, p8) = (
        image.get_pixel(x as usize, (y - 3) as usize, 1usize).unwrap().to_owned() as i16,
        image.get_pixel(x as usize, (y + 3) as usize, 1usize).unwrap().to_owned() as i16,
    );

    let mut above = p0 > high_thresh && p8 > high_thresh;
    let mut below = p0 < low_thresh && p8 < low_thresh;

    if !above && !below {
        return false;
    }

    // JUSTIFICATION - see comment at the start of this function
    let (p4, p12) = (
        image.get_pixel((x + 3) as usize, y as usize, 1usize).unwrap().to_owned() as i16,
        image.get_pixel((x - 3) as usize, y as usize, 1usize).unwrap().to_owned() as i16,
    );

    above = above && ((p4 > high_thresh) || (p12 > high_thresh));
    below = below && ((p4 < low_thresh) || (p12 < low_thresh));

    if !above && !below {
        return false;
    }

    // TODO: Generate a list of pixel offsets once per image,
    // TODO: and use those offsets directly when reading pixels.
    // TODO: This is a little tricky as we can't always do it - we'd
    // TODO: need to distinguish between GenericImages and ImageBuffers.
    // TODO: We can also reduce the number of checks we do below.

    // JUSTIFICATION - see comment at the start of this function
    let pixels = get_circle(image, x, y, p0, p4, p8, p12);

    // Exactly one of above or below is true
    if above {
        has_bright_span(&pixels, 12, high_thresh)
    } else {
        has_dark_span(&pixels, 12, low_thresh)
    }
}

#[inline]
fn get_circle(
    image: &GrayImage,
    x: u32,
    y: u32,
    p0: i16,
    p4: i16,
    p8: i16,
    p12: i16,
) -> [i16; 16] {
    [
        p0,
        image.get_pixel((x + 1) as usize, (y - 3) as usize, 1usize).unwrap().to_owned() as i16,
        image.get_pixel((x + 2) as usize, (y - 2) as usize, 1usize).unwrap().to_owned() as i16,
        image.get_pixel((x + 3) as usize, (y - 1) as usize, 1usize).unwrap().to_owned() as i16,
        p4,
        image.get_pixel((x + 3) as usize, (y + 1) as usize, 1usize).unwrap().to_owned() as i16,
        image.get_pixel((x + 2) as usize, (y + 2) as usize, 1usize).unwrap().to_owned() as i16,
        image.get_pixel((x + 1) as usize, (y + 3) as usize, 1usize).unwrap().to_owned() as i16,
        p8,
        image.get_pixel((x - 1) as usize, (y + 3) as usize, 1usize).unwrap().to_owned() as i16,
        image.get_pixel((x - 2) as usize, (y + 2) as usize, 1usize).unwrap().to_owned() as i16,
        image.get_pixel((x - 3) as usize, (y + 1) as usize, 1usize).unwrap().to_owned() as i16,
        p12,
        image.get_pixel((x - 3) as usize, (y - 1) as usize, 1usize).unwrap().to_owned() as i16,
        image.get_pixel((x - 2) as usize, (y - 2) as usize, 1usize).unwrap().to_owned() as i16,
        image.get_pixel((x - 1) as usize, (y - 3) as usize, 1usize).unwrap().to_owned() as i16,
    ]
}

/// True if the circle has a contiguous section of at least the given length, all
/// of whose pixels have intensities strictly greater than the threshold.
fn has_bright_span(circle: &[i16; 16], length: u8, threshold: i16) -> bool {
    search_span(circle, length, |c| *c > threshold)
}

/// True if the circle has a contiguous section of at least the given length, all
/// of whose pixels have intensities strictly less than the threshold.
fn has_dark_span(circle: &[i16; 16], length: u8, threshold: i16) -> bool {
    search_span(circle, length, |c| *c < threshold)
}

/// True if the circle has a contiguous section of at least the given length, all
/// of whose pixels match f condition.
fn search_span<F>(circle: &[i16; 16], length: u8, f: F) -> bool
where
    F: Fn(&i16) -> bool,
{
    if length > 16 {
        return false;
    }

    let mut nb_ok = 0u8;
    let mut nb_ok_start = None;

    for c in circle.iter() {
        if f(c) {
            nb_ok += 1;
            if nb_ok == length {
                return true;
            }
        } else {
            if nb_ok_start.is_none() {
                nb_ok_start = Some(nb_ok);
            }
            nb_ok = 0;
        }
    }

    nb_ok + nb_ok_start.unwrap() >= length
}