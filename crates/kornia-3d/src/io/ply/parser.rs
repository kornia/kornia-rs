use std::io::{BufRead, Read};
use std::path::Path;

use super::{properties::PlyType, PlyError, PlyPropertyTrait};
use crate::pointcloud::PointCloud;

/// Read a PLY file in binary format.
///
/// NOTE: This function only supports the OpenSplat and XYZRgbNormals PLY file format for now.
/// REF: <https://github.com/pierotofy/OpenSplat>
///
/// Args:
///     path: The path to the PLY file.
///
/// Returns:
///     A `PointCloud` struct containing the points, colors, and normals.
pub fn read_ply_binary(path: impl AsRef<Path>, property: PlyType) -> Result<PointCloud, PlyError> {
    // open the file
    let file = std::fs::File::open(path)?;
    let mut reader = std::io::BufReader::new(file);

    // read the header
    // TODO: parse automatically the header
    let mut header = String::new();
    loop {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        if line.starts_with("end_header") {
            header.push_str(&line);
            break;
        }
        header.push_str(&line);
    }

    // create a buffer for the points
    let mut buffer = vec![0u8; property.size_of()];

    // read the points and store them in a vector
    let mut points = Vec::new();
    let mut colors = Vec::new();
    let mut normals = Vec::new();

    while reader.read_exact(&mut buffer).is_ok() {
        let property_entry = property.deserialize(&buffer)?;
        points.push(property_entry.to_point());
        colors.push(property_entry.to_color());
        normals.push(property_entry.to_normal());
    }

    Ok(PointCloud::new(points, Some(colors), Some(normals)))
}
