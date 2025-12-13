use std::collections::HashMap;
use std::io::{BufRead, Read};
use std::path::Path;

use crate::pointcloud::PointCloud;

/// Error types for the PCD module.
#[derive(Debug, thiserror::Error)]
pub enum PcdError {
    /// Failed to read PCD file
    #[error("Failed to read PCD file")]
    Io(#[from] std::io::Error),

    /// Unsupported or malformed PCD header
    #[error("Unsupported PCD header")]
    UnsupportedProperty,

    /// Invalid PCD file extension
    #[error("Invalid PCD file extension. Got:{0}")]
    InvalidFileExtension(String),
}

/// Describes a single field in a PCD point record
#[derive(Debug)]
struct PcdField {
    name: String,
    offset: usize, // byte offset within a point
    size: usize,   // size of one element (bytes)
    count: usize,  // number of elements
    kind: char,    // 'F', 'U', or 'I'
}

#[derive(Debug)]
struct PcdLayout {
    fields: HashMap<String, PcdField>,
    point_step: usize, // total bytes per point
    points: usize,     // number of points
}

/// Read a little-endian f32 from a byte buffer
#[inline]
fn read_f32(buf: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap())
}

/// Read a little-endian u32 from a byte buffer
#[inline]
fn read_u32(buf: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap())
}

/// Parse the ASCII PCD header and compute the binary layout.


fn parse_pcd_header<R: BufRead>(reader: &mut R) -> Result<PcdLayout, PcdError> {
    let mut field_names = Vec::new();
    let mut sizes = Vec::new();
    let mut types = Vec::new();
    let mut counts = Vec::new();
    let mut points = 0usize;

    loop {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        let line = line.trim();

        if line.starts_with("DATA") {
            if line != "DATA binary" {
                return Err(PcdError::UnsupportedProperty);
            }
            break;
        }

        let mut it = line.split_whitespace();
        match it.next() {
            Some("FIELDS") => field_names = it.map(String::from).collect(),
            Some("SIZE") => sizes = it.map(|v| v.parse().unwrap()).collect(),
            Some("TYPE") => types = it.map(|v| v.chars().next().unwrap()).collect(),
            Some("COUNT") => counts = it.map(|v| v.parse().unwrap()).collect(),
            Some("POINTS") => points = it.next().unwrap().parse().unwrap(),
            _ => {}
        }
    }

    if field_names.is_empty() {
        return Err(PcdError::UnsupportedProperty);
    }

    // Compute byte offsets for each field
    let mut offset = 0usize;
    let mut fields = HashMap::new();

    for i in 0..field_names.len() {
        let count = counts.get(i).copied().unwrap_or(1);
        let size = sizes[i];
        let field = PcdField {
            name: field_names[i].clone(),
            offset,
            size,
            count,
            kind: types[i],
        };
        offset += size * count;
        fields.insert(field.name.clone(), field);
    }

    Ok(PcdLayout {
        fields,
        point_step: offset,
        points,
    })
}

/// Read a PCD file in binary format.
///
/// - XYZ
/// - XYZRGB
/// - XYZ + normals
/// - XYZRGB + normals
pub fn read_pcd_binary(path: impl AsRef<Path>) -> Result<PointCloud, PcdError> {
    let Some(file_ext) = path.as_ref().extension() else {
        return Err(PcdError::InvalidFileExtension("".into()));
    };

    if file_ext != "pcd" {
        return Err(PcdError::InvalidFileExtension(
            file_ext.to_string_lossy().to_string(),
        ));
    }

    // ---- Open file ----
    let file = std::fs::File::open(path)?;
    let mut reader = std::io::BufReader::new(file);

    let layout = parse_pcd_header(&mut reader)?;

    // ---- Required fields ----
    let fx = layout.fields.get("x").ok_or(PcdError::UnsupportedProperty)?.offset;
    let fy = layout.fields.get("y").ok_or(PcdError::UnsupportedProperty)?.offset;
    let fz = layout.fields.get("z").ok_or(PcdError::UnsupportedProperty)?.offset;

    // ---- Optional fields ----
    let frgb = layout.fields.get("rgb").map(|f| f.offset);
    let fnx = layout.fields.get("normal_x").map(|f| f.offset);
    let fny = layout.fields.get("normal_y").map(|f| f.offset);
    let fnz = layout.fields.get("normal_z").map(|f| f.offset);

    let mut buffer = vec![0u8; layout.point_step];

    let mut points = Vec::with_capacity(layout.points);
    let mut colors = Vec::new();
    let mut normals = Vec::new();

    // ---- Read binary points ----
    while reader.read_exact(&mut buffer).is_ok() {
        let x = read_f32(&buffer, fx);
        let y = read_f32(&buffer, fy);
        let z = read_f32(&buffer, fz);
        points.push([x as f64, y as f64, z as f64]);

        if let Some(off) = frgb {
            let rgb = read_u32(&buffer, off);
            colors.push([
                ((rgb >> 16) & 0xFF) as u8,
                ((rgb >> 8) & 0xFF) as u8,
                rgb as u8,
            ]);
        }

        if let (Some(ox), Some(oy), Some(oz)) = (fnx, fny, fnz) {
            normals.push([
                read_f32(&buffer, ox) as f64,
                read_f32(&buffer, oy) as f64,
                read_f32(&buffer, oz) as f64,
            ]);
        }
    }

    Ok(PointCloud::new(
        points,
        if colors.is_empty() { None } else { Some(colors) },
        if normals.is_empty() { None } else { Some(normals) },
    ))
}
