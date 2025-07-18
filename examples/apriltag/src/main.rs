use argh::FromArgs;
use kornia::{
    image::{allocator::CpuAllocator, Image},
    imgproc::color::gray_from_rgb_u8,
    io::functional::read_image_any_rgb8,
};
use kornia_apriltag::{family::TagFamilyKind, AprilTagDecoder, DecodeTagsConfig};

/// Detects AprilTags in an image
#[derive(Debug, FromArgs)]
struct Args {
    /// image path
    #[argh(positional, short = 'p')]
    path: String,

    /// apriltag family kind to detect
    #[argh(
        option,
        short = 'k',
        default = "vec![TagFamilyKind::Tag36H11]",
        from_str_fn(to_tag_family_kind)
    )]
    kind: Vec<TagFamilyKind>,

    /// sharpening factor for decoding
    #[argh(option, short = 'd', default = "0.25")]
    decode_sharpening: f32,

    /// enable edge refinement during detection
    #[argh(switch, short = 'r')]
    refine_edges_enabled: bool,

    /// minimum difference between white and black for detection
    #[argh(option, short = 'm', default = "5")]
    min_white_black_difference: u8,
}

fn to_tag_family_kind(value: &str) -> Result<TagFamilyKind, String> {
    match value {
        "tag16_h5" => Ok(TagFamilyKind::Tag16H5),
        "tag36_h11" => Ok(TagFamilyKind::Tag36H11),
        "tag36_h10" => Ok(TagFamilyKind::Tag36H10),
        "tag25_h9" => Ok(TagFamilyKind::Tag25H9),
        "tagcircle21_h7" => Ok(TagFamilyKind::TagCircle21H7),
        "tagcircle49_h12" => Ok(TagFamilyKind::TagCircle49H12),
        "tagcustom48_h12" => Ok(TagFamilyKind::TagCustom48H12),
        "tagstandard41_h12" => Ok(TagFamilyKind::TagStandard41H12),
        "tagstandard52_h13" => Ok(TagFamilyKind::TagStandard52H13),
        _ => Err("Unsupported TagFamilyKind".to_string()),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    let rec = rerun::RecordingStreamBuilder::new("Kornia AprilTag Detection Example").spawn()?;

    let img = read_image_any_rgb8(args.path)?;

    let mut grayscale_img = Image::from_size_val(img.size(), 0, CpuAllocator)?;
    gray_from_rgb_u8(&img, &mut grayscale_img)?;

    let mut config = DecodeTagsConfig::new(args.kind);

    config.refine_edges_enabled = args.refine_edges_enabled;
    config.min_white_black_difference = args.min_white_black_difference;
    config.decode_sharpening = args.decode_sharpening;

    let mut decoder = AprilTagDecoder::new(config, grayscale_img.size())?;
    let detections = decoder.decode(&grayscale_img)?;

    let mut all_coords = Vec::new();
    let mut all_labels = Vec::new();
    let mut all_colors = Vec::new();

    for detection in detections {
        let coords = [
            [detection.quad.corners[0].x, detection.quad.corners[0].y],
            [detection.quad.corners[1].x, detection.quad.corners[1].y],
            [detection.quad.corners[2].x, detection.quad.corners[2].y],
            [detection.quad.corners[3].x, detection.quad.corners[3].y],
            [detection.quad.corners[0].x, detection.quad.corners[0].y],
        ];

        all_coords.push(coords);
        all_labels.push(detection.id.to_string());
        all_colors.push(tag_family_color(detection.tag_family_kind));
    }

    rec.log(
        "Image",
        &rerun::Image::from_elements(img.as_slice(), img.size().into(), rerun::ColorModel::RGB),
    )?;

    rec.log(
        "Image",
        &rerun::LineStrips2D::new(all_coords)
            .with_labels(all_labels)
            .with_colors(all_colors),
    )?;

    Ok(())
}

fn tag_family_color(kind: TagFamilyKind) -> [u8; 3] {
    match kind {
        TagFamilyKind::Tag16H5 => [255, 0, 0],            // Red
        TagFamilyKind::Tag36H11 => [0, 255, 0],           // Green
        TagFamilyKind::Tag36H10 => [0, 0, 255],           // Blue
        TagFamilyKind::Tag25H9 => [255, 255, 0],          // Yellow
        TagFamilyKind::TagCircle21H7 => [255, 0, 255],    // Magenta
        TagFamilyKind::TagCircle49H12 => [0, 255, 255],   // Cyan
        TagFamilyKind::TagCustom48H12 => [255, 128, 0],   // Orange
        TagFamilyKind::TagStandard41H12 => [128, 0, 255], // Purple
        TagFamilyKind::TagStandard52H13 => [0, 128, 255], // Sky Blue
        TagFamilyKind::Custom(_) => [128, 128, 128],      // Gray for custom/unknown
    }
}
