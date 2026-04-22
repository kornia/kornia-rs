use argh::FromArgs;

use kornia::{
    image::{ops, Image},
    imgproc::{color, contours, threshold},
    tensor::CpuAllocator,
};

#[derive(FromArgs)]
/// Detect contours in an image file using the Suzuki-Abe algorithm and
/// visualise them depth-coloured with Rerun.
struct Args {
    /// path to the input image
    #[argh(option)]
    image: String,

    /// contour approximation: none | simple  [default: simple]
    #[argh(option, default = "String::from(\"simple\")")]
    approx: String,

    /// binarisation threshold 0-255  [default: 128]
    #[argh(option, default = "128u8")]
    threshold: u8,
}

/// Depth-keyed palette. Index = depth level
const DEPTH_COLORS: &[(u8, u8, u8)] = &[
    (0, 230, 80),   // 0 – green
    (255, 140, 0),  // 1 – orange
    (30, 144, 255), // 2 – blue
    (255, 220, 0),  // 3 – yellow
    (220, 30, 30),  // 4+ – red
];

fn depth_color(depth: usize) -> rerun::Color {
    let (r, g, b) = DEPTH_COLORS[depth.min(DEPTH_COLORS.len() - 1)];
    rerun::Color::from_rgb(r, g, b)
}

/// Walk parent links in the Tree hierarchy to get nesting depth of contour `i`.
fn contour_depth(hierarchy: &[[i32; 4]], i: usize) -> usize {
    let mut depth: usize = 0usize;
    let mut cur = hierarchy[i][3];
    while cur >= 0 {
        depth += 1;
        cur = hierarchy[cur as usize][3];
    }
    depth
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();
    let approx = match args.approx.to_lowercase().as_str() {
        "none" => Ok(contours::ContourApproximationMode::None),
        "simple" => Ok(contours::ContourApproximationMode::Simple),
        other => Err(format!("unknown approx '{other}'; valid: none | simple")),
    }?;

    // read the image
    let img = kornia::io::functional::read_image_any_rgb8(&args.image)?;
    let size = img.size();

    let mut img_f32 = Image::from_size_val(size, 0f32, CpuAllocator)?;
    let mut gray = Image::from_size_val(size, 0f32, CpuAllocator)?;

    // convert to gray scale
    ops::cast_and_scale(&img, &mut img_f32, 1.0 / 255.0)?;
    color::gray_from_rgb(&img_f32, &mut gray)?;

    // convert to binary image
    let mut binary_f32 = Image::from_size_val(size, 0f32, CpuAllocator)?;
    threshold::threshold_binary(
        &gray,
        &mut binary_f32,
        args.threshold as f32 / 255.0,
        1.0f32,
    )?;
    let mut binary = Image::from_size_val(size, 0u8, CpuAllocator)?;
    ops::cast_and_scale(&binary_f32, &mut binary, 1)?;

    // find contours
    let mut executor = contours::FindContoursExecutor::new();
    let result = executor.find_contours(&binary, contours::RetrievalMode::Tree, approx)?;

    // split contours into per-depth (strips, dots)
    let n_depths = DEPTH_COLORS.len();
    let mut strips: Vec<Vec<Vec<[f32; 2]>>> = vec![Vec::new(); n_depths];
    let mut dots: Vec<Vec<[f32; 2]>> = vec![Vec::new(); n_depths];

    for (i, contour) in result.contours.iter().enumerate() {
        let d = contour_depth(&result.hierarchy, i).min(n_depths - 1);
        match contour.len() {
            0 => {}
            1 => {
                dots[d].push([contour[0][0] as f32, contour[0][1] as f32]);
            }
            _ => {
                let mut pts: Vec<[f32; 2]> =
                    contour.iter().map(|&[x, y]| [x as f32, y as f32]).collect();
                pts.push(pts[0]);
                strips[d].push(pts);
            }
        }
    }

    // create a Rerun recording stream
    let rec = rerun::RecordingStreamBuilder::new("Kornia Contours").spawn()?;

    rec.log(
        "image/rgb",
        &rerun::Image::from_elements(img.as_slice(), img.size().into(), rerun::ColorModel::RGB),
    )?;

    for d in 0..n_depths {
        let color = depth_color(d);
        let path_base = format!("image/rgb/depth_{d}");

        if !strips[d].is_empty() {
            rec.log(
                format!("{path_base}/contours"),
                &rerun::LineStrips2D::new(strips[d].clone()).with_colors([color]),
            )?;
        }

        if !dots[d].is_empty() {
            rec.log(
                format!("{path_base}/isolated"),
                &rerun::Points2D::new(dots[d].clone())
                    .with_colors([color])
                    .with_radii([1.5f32]),
            )?;
        }
    }
    Ok(())
}
