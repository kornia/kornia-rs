//! Dump contours produced by kornia find_contours as JSON.
//! Usage: dump_contours <png_path> <external|list> <simple|none>
//! Used by check_correctness.py to compare bit-exact against cv2.findContours.

use kornia_image::{allocator::CpuAllocator, Image, ImageSize};
use kornia_imgproc::contours::{
    find_contours, ContourApproximationMode, RetrievalMode,
};
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: {} <png> <external|list> <simple|none>", args[0]);
        std::process::exit(2);
    }
    let path = &args[1];
    let mode = match args[2].as_str() {
        "external" => RetrievalMode::External,
        "list" => RetrievalMode::List,
        _ => { eprintln!("bad mode"); std::process::exit(2); }
    };
    let method = match args[3].as_str() {
        "simple" => ContourApproximationMode::Simple,
        "none" => ContourApproximationMode::None,
        _ => { eprintln!("bad method"); std::process::exit(2); }
    };

    // Load via kornia, threshold via kornia, then find_contours.
    let rgb = kornia_io::png::read_image_png_rgb8(path)?;
    let (w, h) = (rgb.width(), rgb.height());
    let mut gray = Image::<u8, 1, _>::from_size_val(
        ImageSize { width: w, height: h }, 0, CpuAllocator,
    )?;
    kornia_imgproc::color::gray_from_rgb_u8(&rgb, &mut gray)?;
    let mut bw = Image::<u8, 1, _>::from_size_val(
        ImageSize { width: w, height: h }, 0, CpuAllocator,
    )?;
    kornia_imgproc::threshold::threshold_binary(&gray, &mut bw, 127, 1)?;

    let result = find_contours(&bw, mode, method)?;

    // Emit JSON: {"contours": [[[x,y], [x,y], ...], ...]}
    print!("{{\"contours\": [");
    for (i, c) in result.contours.iter().enumerate() {
        if i > 0 { print!(","); }
        print!("[");
        for (j, p) in c.iter().enumerate() {
            if j > 0 { print!(","); }
            print!("[{},{}]", p[0], p[1]);
        }
        print!("]");
    }
    println!("]}}");
    Ok(())
}
