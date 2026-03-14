//! 1. Basic usage - find_contours with RetrievalMode::External and
//!    ContourApproximationMode::Simple on a synthetic binary image
//!
//! 2. Buffer reuse across frames - FindContoursExecutor processes successive
//!    frames without re-allocating scratch buffers after the first warm-up call
//!
//! 3. CComp hierarchy traversal - reading [next, prev, first_child, parent]
//!    entries from RetrievalMode::CComp on a hollow square image
//!
//! cargo run --example find_contours -p kornia-imgproc

use kornia_image::{allocator::CpuAllocator, Image, ImageSize};
use kornia_imgproc::contours::{
    find_contours, ContourApproximationMode, FindContoursExecutor, RetrievalMode,
};

fn make_image(
    width: usize,
    height: usize,
    data: Vec<u8>,
) -> Result<Image<u8, 1, CpuAllocator>, Box<dyn std::error::Error>> {
    Ok(Image::new(ImageSize { width, height }, data, CpuAllocator)?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    basic_usage()?;
    buffer_reuse_across_frames()?;
    ccomp_hierarchy_traversal()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// 1. Basic usage
// ---------------------------------------------------------------------------

/// Find the outermost contours of a binary image
///
/// Image layout (1 = foreground, 0 = background)
///
///  0 0 0 0 0 0 0
///  0 1 1 1 1 1 0
///  0 1 1 1 1 1 0
///  0 1 1 1 1 1 0
///  0 1 1 1 1 1 0
///  0 1 1 1 1 1 0
///  0 0 0 0 0 0 0
///
/// ContourApproximationMode::Simple compresses collinear runs to their
/// endpoints, so the 5×5 filled block is represented by its 4 corners
fn basic_usage() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== 1. Basic usage ===");

    #[rustfmt::skip]
    let image = make_image(7, 7, vec![
        0, 0, 0, 0, 0, 0, 0,
        0, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 0,
        0, 0, 0, 0, 0, 0, 0,
    ])?;

    let result = find_contours(
        &image,
        RetrievalMode::External,
        ContourApproximationMode::Simple,
    )?;

    println!("  Contours found : {}", result.contours.len());
    for (i, contour) in result.contours.iter().enumerate() {
        println!(
            "  Contour #{i:>2} : {} point(s)  {:?}",
            contour.len(),
            contour
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 2. Buffer reuse across frames
// ---------------------------------------------------------------------------

/// Process two frames with a single FindContoursExecutor
///
/// Internal scratch buffers (img, arena, ranges, hierarchy, border_types) are
/// allocated on the first call and reused on every subsequent call, retaining
/// their capacity so the OS allocator is not touched again. The returned
/// ContoursResult is newly allocated on every call because ownership of the
/// contour and hierarchy vecs transfers to the caller
/// This is the recommended pattern for video pipelines or batch processing
fn buffer_reuse_across_frames() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== 2. Buffer reuse across frames ===");

    #[rustfmt::skip]
    let frame_a = make_image(3, 3, vec![
        0, 0, 0,
        0, 1, 0,
        0, 0, 0,
    ])?;

    #[rustfmt::skip]
    let frame_b = make_image(5, 5, vec![
        0, 0, 0, 0, 0,
        0, 1, 1, 1, 0,
        0, 1, 1, 1, 0,
        0, 1, 1, 1, 0,
        0, 0, 0, 0, 0,
    ])?;

    let mut executor = FindContoursExecutor::new();

    let result_a = executor.find_contours(
        &frame_a,
        RetrievalMode::External,
        ContourApproximationMode::None,
    )?;
    println!("  Frame A - contours : {}", result_a.contours.len());
    println!("           points   : {:?}", result_a.contours[0]);

    let result_b = executor.find_contours(
        &frame_b,
        RetrievalMode::External,
        ContourApproximationMode::None,
    )?;
    println!("  Frame B - contours : {}", result_b.contours.len());
    println!(
        "           points   : {} perimeter pixels",
        result_b.contours[0].len()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// 3. CComp hierarchy traversal
// ---------------------------------------------------------------------------

/// Read hierarchy entries from RetrievalMode::CComp on a hollow square
///
/// Image layout
///
///  0 0 0 0 0 0
///  0 1 1 1 1 0
///  0 1 0 0 1 0
///  0 1 0 0 1 0
///  0 1 1 1 1 0
///  0 0 0 0 0 0
///
/// Produces two contours:
///  Contour 0 - outer ring  (no parent)
///  Contour 1 - inner hole  (parent = 0)
///
/// Each HierarchyEntry is [next, prev, first_child, parent] where every
/// field is a 0-based contour index, or -1 for "no link"
fn ccomp_hierarchy_traversal() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== 3. CComp hierarchy traversal ===");

    #[rustfmt::skip]
    let image = make_image(6, 6, vec![
        0, 0, 0, 0, 0, 0,
        0, 1, 1, 1, 1, 0,
        0, 1, 0, 0, 1, 0,
        0, 1, 0, 0, 1, 0,
        0, 1, 1, 1, 1, 0,
        0, 0, 0, 0, 0, 0,
    ])?;

    let result = find_contours(
        &image,
        RetrievalMode::CComp,
        ContourApproximationMode::Simple,
    )?;

    println!("  Contours found : {}", result.contours.len());
    println!();

    for (i, (contour, hier)) in result
        .contours
        .iter()
        .zip(result.hierarchy.iter())
        .enumerate()
    {
        let [next, prev, first_child, parent] = *hier;
        println!("  Contour #{i}");
        println!("    points      : {}", contour.len());
        println!("    next        : {}", fmt_link(next));
        println!("    prev        : {}", fmt_link(prev));
        println!("    first_child : {}", fmt_link(first_child));
        println!("    parent      : {}", fmt_link(parent));
        println!();
    }

    Ok(())
}

fn fmt_link(v: i32) -> String {
    if v < 0 {
        "none".to_string()
    } else {
        format!("#{v}")
    }
}
