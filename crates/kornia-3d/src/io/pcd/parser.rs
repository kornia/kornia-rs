use serde::Deserialize;
use std::io::{BufRead, Read};
use std::path::Path;

use crate::pointcloud::PointCloud;

/// Error types for the PCD module.
#[derive(Debug, thiserror::Error)]
pub enum PcdError {
    /// Failed to read PCD file
    #[error("Failed to read PCD file")]
    Io(#[from] std::io::Error),

    /// Failed to deserialize PCD file
    #[error("Failed to deserialize PCD file")]
    Deserialize(#[from] bincode::Error),

    /// Unsupported PCD property
    #[error("Unsupported PCD property")]
    UnsupportedProperty,

    /// Invalid PCD file extension
    #[error("Invalid PCD file extension. Got:{0}")]
    InvalidFileExtension(String),
}

/// A property of a point in a PCD file.
#[derive(Debug, Deserialize)]
pub struct PcdPropertyXYZRGBNCurvature {
    /// x coordinate
    pub x: f32,
    /// y coordinate
    pub y: f32,
    /// z coordinate
    pub z: f32,
    /// rgb color
    pub rgb: u32,
    /// normal x coordinate
    pub nx: f32,
    /// normal y coordinate
    pub ny: f32,
    /// normal z coordinate
    pub nz: f32,
    /// curvature
    pub curvature: f32,
}

/// Read a PCD file in binary format.
///
/// Args:
///     path: The path to the PCD file.
///
/// Returns:
///     A `PointCloud` struct containing the points, colors, and normals.
pub fn read_pcd_binary(path: impl AsRef<Path>) -> Result<PointCloud, PcdError> {
    let Some(file_ext) = path.as_ref().extension() else {
        return Err(PcdError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "File extension is missing",
        )));
    };

    if file_ext != "pcd" {
        return Err(PcdError::InvalidFileExtension(
            file_ext.to_string_lossy().to_string(),
        ));
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
        if line.starts_with("DATA binary") {
            break;
        }
        header.push_str(&line);
    }

    // create a buffer for the points
    let mut buffer = vec![0u8; std::mem::size_of::<PcdPropertyXYZRGBNCurvature>()];

    // read the points and store them in a vector
    let mut points = Vec::new();
    let mut colors = Vec::new();
    let mut normals = Vec::new();

    while reader.read_exact(&mut buffer).is_ok() {
        let property: PcdPropertyXYZRGBNCurvature = bincode::deserialize(&buffer)?;
        points.push([property.x as f64, property.y as f64, property.z as f64]);
        let rgb = property.rgb as u32;
        colors.push([
            ((rgb >> 16) & 0xFF) as u8,
            ((rgb >> 8) & 0xFF) as u8,
            rgb as u8,
        ]);
        normals.push([property.nx as f64, property.ny as f64, property.nz as f64]);
    }

    Ok(PointCloud::new(points, Some(colors), Some(normals)))
}
