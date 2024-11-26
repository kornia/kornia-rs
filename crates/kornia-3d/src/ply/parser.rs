use std::io::{BufRead, Read};
use std::path::Path;

use crate::ply::properties::OpenSplatProperty;
use crate::pointcloud::{PointCloud, Vec3};

#[derive(Debug, thiserror::Error)]
pub enum PlyError {
    #[error("Failed to read PLY file")]
    Io(#[from] std::io::Error),

    #[error("Failed to deserialize PLY file")]
    Deserialize(#[from] bincode::Error),
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
pub fn read_ply_binary(path: impl AsRef<Path>) -> Result<PointCloud, PlyError> {
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
    let mut buffer = vec![0u8; std::mem::size_of::<OpenSplatProperty>()];

    // read the points and store them in a vector
    let mut points = Vec::new();
    let mut colors = Vec::new();
    let mut normals = Vec::new();

    while reader.read_exact(&mut buffer).is_ok() {
        let property: OpenSplatProperty = bincode::deserialize(&buffer)?;
        points.push(Vec3 {
            x: property.x,
            y: property.y,
            z: property.z,
        });
        colors.push(Vec3 {
            x: property.f_dc_0,
            y: property.f_dc_1,
            z: property.f_dc_2,
        });
        normals.push(Vec3 {
            x: property.nx,
            y: property.ny,
            z: property.nz,
        });
    }

    Ok(PointCloud::new(points, Some(colors), Some(normals)))
}
