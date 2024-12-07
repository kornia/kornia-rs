use std::io::{BufRead, Read};
use std::path::Path;

use super::properties::{OpenSplatProperty, PlyProperty};
use crate::pointcloud::{DVec3, PointCloud};

#[derive(Debug, thiserror::Error)]
pub enum PlyError {
    #[error("Failed to read PLY file")]
    Io(#[from] std::io::Error),

    #[error("Failed to deserialize PLY file")]
    Deserialize(#[from] bincode::Error),

    #[error("Unsupported PLY property")]
    UnsupportedProperty,
}

/// Read a PLY file in binary format.
///
/// NOTE: This function only supports the OpenSplat PLY file format format for now.
/// REF: https://github.com/pierotofy/OpenSplat
///
/// Args:
///     path: The path to the PLY file.
///
/// Returns:
///     A `PointCloud` struct containing the points, colors, and normals.
pub fn read_ply_binary(
    path: impl AsRef<Path>,
    property: PlyProperty,
) -> Result<PointCloud, PlyError> {
    if property != PlyProperty::OpenSplat {
        return Err(PlyError::UnsupportedProperty);
    }

    // open the file
    let file = std::fs::File::open(path)?;
    let mut reader = std::io::BufReader::new(file);

    // read the header
    // TODO support other formats headers
    let mut header = String::new();
    loop {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        if line.starts_with("end_header") {
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
        match property {
            PlyProperty::OpenSplat => {
                let property: OpenSplatProperty = bincode::deserialize(&buffer)?;
                points.push(DVec3 {
                    x: property.x as f64,
                    y: property.y as f64,
                    z: property.z as f64,
                });
                colors.push(DVec3 {
                    x: property.f_dc_0 as f64,
                    y: property.f_dc_1 as f64,
                    z: property.f_dc_2 as f64,
                });
                normals.push(DVec3 {
                    x: property.nx as f64,
                    y: property.ny as f64,
                    z: property.nz as f64,
                });
            }
        }
    }

    Ok(PointCloud::new(points, Some(colors), Some(normals)))
}
