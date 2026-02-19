use std::collections::HashMap;
use std::io::{BufRead, Read};
use std::path::Path;

use crate::pointcloud::PointCloud;

const MAX_POINT_STEP: usize = 1024;
const MAX_POINTS: usize = 50_000_000;

/// Error types for the PCD module.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PcdError {
    /// Failed to read PCD file
    #[error("Failed to read PCD file")]
    Io(#[from] std::io::Error),

    /// Unsupported header
    #[error("Unsupported PCD header")]
    UnsupportedProperty,
    
    /// Malformed PCD header
    #[error("Malformed PCD header")]
    MalformedHeader,

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
    kind: char, // PCD type: 'F' = float, 'U' = unsigned int, 'I' = signed int
}

#[derive(Debug)]
struct PcdLayout {
    fields: HashMap<String, PcdField>,
    point_step: usize, // total bytes per point
    num_points: usize,     // number of points
}

impl PcdLayout {
    fn get_field_offset(&self, name: &str) -> Result<usize, PcdError> {
        self.fields
            .get(name)
            .map(|f| f.offset)
            .ok_or(PcdError::UnsupportedProperty)
    }
}

/// Read a little-endian f32 from a byte buffer
#[inline]
fn read_f32(buf: &[u8], offset: usize) -> Result<f32, PcdError> {
    let slice = buf.get(offset..offset + 4).ok_or(PcdError::UnsupportedProperty)?;
    let mut bytes = [0u8; 4];
    bytes.copy_from_slice(slice);
    Ok(f32::from_le_bytes(bytes))
}

/// Read a little-endian u32 from a byte buffer
#[inline]
fn read_u32(buf: &[u8], offset: usize) -> Result<u32, PcdError> {
    let slice = buf.get(offset..offset + 4).ok_or(PcdError::UnsupportedProperty)?;
    let mut bytes = [0u8; 4];
    bytes.copy_from_slice(slice);
    Ok(u32::from_le_bytes(bytes))
}


fn parse_pcd_layout<R: BufRead>(reader: &mut R) -> Result<PcdLayout, PcdError> {
    let mut field_names: Vec<String> = Vec::new();
    let mut sizes = Vec::new();
    let mut types = Vec::new();
    let mut counts = Vec::new();
    let mut points = 0usize;

    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line)?;
        if n == 0 {
            return Err(PcdError::MalformedHeader);
        }
        let line = line.trim();

        if line.starts_with("DATA") {
            if line != "DATA binary" {
                return Err(PcdError::UnsupportedProperty);
            }
            break;
        }

        let mut it = line.split_whitespace();
        match it.next() {
            Some("SIZE") => {
                sizes = it
                    .map(|v| v.parse::<usize>().map_err(|_| PcdError::UnsupportedProperty))
                    .collect::<Result<Vec<_>, _>>()?;
            }
            Some("TYPE") => {
                types = it
                    .map(|v| v.chars().next().ok_or(PcdError::UnsupportedProperty))
                    .collect::<Result<Vec<_>, _>>()?;
            }
            Some("COUNT") => {
                counts = it
                    .map(|v| v.parse::<usize>().map_err(|_| PcdError::UnsupportedProperty))
                    .collect::<Result<Vec<_>, _>>()?;
            }
            Some("POINTS") => {
                let token = it.next().ok_or(PcdError::UnsupportedProperty)?;
                points = token
                    .parse::<usize>()
                    .map_err(|_| PcdError::UnsupportedProperty)?;
            }
            Some("FIELDS") => field_names = it.map(String::from).collect(),
            _ => {}
        }
    }

    if field_names.is_empty()
        || sizes.len() != field_names.len()
        || types.len() != field_names.len()
        || (!counts.is_empty() && counts.len() != field_names.len())
    {
        return Err(PcdError::UnsupportedProperty);
    }

    // Compute byte offsets for each field
    let mut offset = 0usize;
    let mut fields = HashMap::new();

    for i in 0..field_names.len() {
        // If COUNT is omitted, PCD spec defines default count as 1
        let count = counts.get(i).copied().unwrap_or(1);
        let size = sizes[i];
        let field = PcdField {
            name: field_names[i].clone(),
            offset,
            size,
            count,
            kind: types[i],
        };

        match field_names[i].as_str() {
            "x" | "y" | "z" | "normal_x" | "normal_y" | "normal_z" | "nx" | "ny" | "nz" => {
                if !(size == 4 && count == 1 && types[i] == 'F') {
                    return Err(PcdError::UnsupportedProperty);
                }
            }
            "rgb" => {
                if !(size == 4 && count == 1 && (types[i] == 'U' || types[i] == 'I' || types[i] == 'F')) {
                    return Err(PcdError::UnsupportedProperty);
                }
            }
            _ => {}
        }

        let field_bytes = size
            .checked_mul(count)
            .ok_or(PcdError::MalformedHeader)?;

        offset = offset
            .checked_add(field_bytes)
            .ok_or(PcdError::MalformedHeader)?;

        if offset > MAX_POINT_STEP {
            return Err(PcdError::MalformedHeader);
        }

        if fields.contains_key(&field.name) {
            return Err(PcdError::MalformedHeader);
        }
        fields.insert(field.name.clone(), field);
    }

    Ok(PcdLayout {
        fields,
        point_step: offset,
        num_points: points,
    })
}

/// Read a binary PCD file.
///
/// # Arguments
/// * `path` - Path to a `.pcd` file.
///
/// # Returns
/// A [`PointCloud`] containing:
/// - 3D points (always)
/// - RGB colors (if present)
/// - Normals (if present)
///
/// # Supported formats
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

    // Open file
    let file = std::fs::File::open(path)?;
    let mut reader = std::io::BufReader::new(file);

    let layout = parse_pcd_layout(&mut reader)?;

    if layout.num_points == 0 {
        return Err(PcdError::MalformedHeader);
    }

    if layout.num_points > MAX_POINTS {
        return Err(PcdError::MalformedHeader);
    }

    // Required fields
    let fx = layout.get_field_offset("x")?;
    let fy = layout.get_field_offset("y")?;
    let fz = layout.get_field_offset("z")?;


    // Optional fields
    let frgb = layout.fields.get("rgb").map(|f| f.offset);
    let fnx = layout.fields.get("normal_x").or_else(|| layout.fields.get("nx")).map(|f| f.offset);
    let fny = layout.fields.get("normal_y").or_else(|| layout.fields.get("ny")).map(|f| f.offset);
    let fnz = layout.fields.get("normal_z").or_else(|| layout.fields.get("nz")).map(|f| f.offset);

    if layout.point_step == 0 || layout.point_step > MAX_POINT_STEP {
        return Err(PcdError::MalformedHeader);
    }

    let mut buffer = vec![0u8; layout.point_step];

    let mut points = Vec::with_capacity(layout.num_points);
    let mut colors = if frgb.is_some() {
        Vec::with_capacity(layout.num_points)
    } else {
        Vec::new()
    };

    let mut normals = if fnx.is_some() && fny.is_some() && fnz.is_some() {
        Vec::with_capacity(layout.num_points)
    } else {
        Vec::new()
    };

    // Read binary points
    for _ in 0..layout.num_points {
        reader.read_exact(&mut buffer)?;

        let x = read_f32(&buffer, fx)?;
        let y = read_f32(&buffer, fy)?;
        let z = read_f32(&buffer, fz)?;
        points.push([x as f64, y as f64, z as f64]);

        if let Some(off) = frgb {
            let rgb = read_u32(&buffer, off)?;
            colors.push([
                ((rgb >> 16) & 0xFF) as u8,
                ((rgb >> 8) & 0xFF) as u8,
                (rgb & 0xFF) as u8,
            ]);
        }

        if let (Some(ox), Some(oy), Some(oz)) = (fnx, fny, fnz) {
            normals.push([
                read_f32(&buffer, ox)? as f64,
                read_f32(&buffer, oy)? as f64,
                read_f32(&buffer, oz)? as f64,
            ]);
        }
    }

    Ok(PointCloud::new(
        points,
        (!colors.is_empty()).then_some(colors),
        (!normals.is_empty()).then_some(normals),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn fails_on_ascii_or_non_binary() {
        let data = b"FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
POINTS 1
DATA ascii";
        let mut reader = Cursor::new(&data[..]);
        assert!(parse_pcd_layout(&mut reader).is_err());
    }

    #[test]
    fn parses_valid_binary_header() {
        let data = b"FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
POINTS 10
DATA binary";
        let mut reader = Cursor::new(&data[..]);
        let layout = parse_pcd_layout(&mut reader).expect("valid binary header should parse");
        assert_eq!(layout.num_points, 10);
        assert!(layout.fields.contains_key("x"));
    }

    #[test]
    fn rejects_wrong_type_for_xyz() {
        let data = b"FIELDS x y z
SIZE 4 4 4
TYPE I I I
COUNT 1 1 1
POINTS 5
DATA binary";
        let mut reader = Cursor::new(&data[..]);
        assert!(parse_pcd_layout(&mut reader).is_err());
    }
}

