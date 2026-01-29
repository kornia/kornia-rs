use std::io::{BufRead, Read};
use std::path::Path;

use super::{
    properties::{PlyDataType, PlyPropertyDefinition, PlyType},
    PlyError, PlyPropertyTrait,
};
use crate::pointcloud::PointCloud;

struct PlyHeader {
    pub vertex_count: usize,
    pub properties: Vec<PlyPropertyDefinition>,
    pub format: PlyType,
}

fn parse_header<R: BufRead>(reader: &mut R) -> Result<PlyHeader, PlyError> {
    let mut line = String::new();
    let mut vertex_count = None;
    let mut is_binary_little_endian = false;
    let mut is_ply = false;
    let mut properties = Vec::new();

    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        let trimmed = line.trim();

        if trimmed == "ply" {
            is_ply = true;
            continue;
        }

        if trimmed == "end_header" {
            break;
        }

        if trimmed.starts_with("format binary_little_endian") {
            is_binary_little_endian = true;
        } else if trimmed.starts_with("element vertex") {
            vertex_count = Some(
                trimmed
                    .split_whitespace()
                    .last()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0),
            );
        } else if trimmed.starts_with("property") {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() >= 3 {
                let data_type = parse_data_type(parts[1])?;
                let name = parts[2].to_string();
                properties.push(PlyPropertyDefinition { name, data_type });
            }
        }
    }

    if !is_ply || !is_binary_little_endian {
        return Err(PlyError::UnsupportedProperty);
    }

    let vertex_count = vertex_count.ok_or(PlyError::UnsupportedProperty)?;
    let format = PlyType::detect_format(&properties)?;

    Ok(PlyHeader {
        vertex_count,
        properties,
        format,
    })
}

fn parse_data_type(type_str: &str) -> Result<PlyDataType, PlyError> {
    match type_str {
        "float" | "float32" => Ok(PlyDataType::Float32),
        "double" | "float64" => Ok(PlyDataType::Float64),
        "char" | "int8" => Ok(PlyDataType::Int8),
        "uchar" | "uint8" => Ok(PlyDataType::UInt8),
        "short" | "int16" => Ok(PlyDataType::Int16),
        "ushort" | "uint16" => Ok(PlyDataType::UInt16),
        "int" | "int32" => Ok(PlyDataType::Int32),
        "uint" | "uint32" => Ok(PlyDataType::UInt32),
        _ => Err(PlyError::UnsupportedProperty),
    }
}

/// Read a PLY file in binary format with automatic format detection.
pub fn read_ply_binary(path: impl AsRef<Path>) -> Result<PointCloud, PlyError> {
    let file = std::fs::File::open(path)?;
    let mut reader = std::io::BufReader::new(file);
    let header = parse_header(&mut reader)?;
    let mut buffer = vec![0u8; header.format.size_of()];

    let mut points = Vec::with_capacity(header.vertex_count);
    let mut colors = Vec::with_capacity(header.vertex_count);
    let mut normals = Vec::with_capacity(header.vertex_count);

    for _ in 0..header.vertex_count {
        reader.read_exact(&mut buffer)?;
        let property_entry = header.format.deserialize(&buffer)?;
        points.push(property_entry.to_point());
        colors.push(property_entry.to_color());
        normals.push(property_entry.to_normal());
    }

    Ok(PointCloud::new(points, Some(colors), Some(normals)))
}

/// Read a PLY file in binary format with explicit format specification.
pub fn read_ply_binary_with_format(
    path: impl AsRef<Path>,
    property: PlyType,
) -> Result<PointCloud, PlyError> {
    let file = std::fs::File::open(path)?;
    let mut reader = std::io::BufReader::new(file);
    let header = parse_header(&mut reader)?;
    let mut buffer = vec![0u8; property.size_of()];

    let mut points = Vec::with_capacity(header.vertex_count);
    let mut colors = Vec::with_capacity(header.vertex_count);
    let mut normals = Vec::with_capacity(header.vertex_count);

    for _ in 0..header.vertex_count {
        reader.read_exact(&mut buffer)?;
        let property_entry = property.deserialize(&buffer)?;
        points.push(property_entry.to_point());
        colors.push(property_entry.to_color());
        normals.push(property_entry.to_normal());
    }

    Ok(PointCloud::new(points, Some(colors), Some(normals)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_parse_header_basic() {
        let header_text = "ply\nformat binary_little_endian 1.0\nelement vertex 10\nproperty float x\nproperty float y\nproperty float z\nend_header\n";
        let mut reader = std::io::BufReader::new(header_text.as_bytes());
        let header = parse_header(&mut reader).unwrap();
        assert_eq!(header.vertex_count, 10);
        assert_eq!(header.properties.len(), 3);
        assert_eq!(header.properties[0].name, "x");
        assert_eq!(header.properties[0].data_type, PlyDataType::Float32);
    }

    #[test]
    fn test_parse_header_xyz_rgb_normals() {
        let header_text = "ply\nformat binary_little_endian 1.0\nelement vertex 5\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty float nx\nproperty float ny\nproperty float nz\nend_header\n";
        let mut reader = std::io::BufReader::new(header_text.as_bytes());
        let header = parse_header(&mut reader).unwrap();
        assert_eq!(header.vertex_count, 5);
        assert_eq!(header.properties.len(), 9);
        assert_eq!(header.format, PlyType::XYZRgbNormals);
    }

    #[test]
    fn test_parse_header_dynamic() {
        let header_text = "ply\nformat binary_little_endian 1.0\nelement vertex 5\nproperty float x\nproperty float y\nproperty float z\nproperty float intensity\nend_header\n";
        let mut reader = std::io::BufReader::new(header_text.as_bytes());
        let header = parse_header(&mut reader).unwrap();
        assert_eq!(header.vertex_count, 5);
        assert_eq!(header.properties.len(), 4);
        match header.format {
            PlyType::Dynamic(ref props) => {
                assert_eq!(props.len(), 4);
                assert_eq!(props[0].name, "x");
                assert_eq!(props[3].name, "intensity");
            }
            _ => panic!("Expected dynamic format"),
        }
    }

    #[test]
    fn test_data_type_parsing() {
        assert_eq!(parse_data_type("float").unwrap(), PlyDataType::Float32);
        assert_eq!(parse_data_type("uchar").unwrap(), PlyDataType::UInt8);
        assert_eq!(parse_data_type("double").unwrap(), PlyDataType::Float64);
        assert!(parse_data_type("invalid").is_err());
    }

    #[test]
    fn test_read_ply_binary_auto_detection() {
        let mut file = NamedTempFile::new().unwrap();
        let header = "ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty float nx\nproperty float ny\nproperty float nz\nend_header\n";
        file.write_all(header.as_bytes()).unwrap();

        let mut data = Vec::new();
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&2.0f32.to_le_bytes());
        data.extend_from_slice(&3.0f32.to_le_bytes());
        data.push(255);
        data.push(128);
        data.push(0);
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        file.write_all(&data).unwrap();

        let pointcloud = read_ply_binary(file.path()).unwrap();
        assert_eq!(pointcloud.len(), 1);
        assert_eq!(pointcloud.points()[0], [1.0, 2.0, 3.0]);
        assert_eq!(pointcloud.colors().unwrap()[0], [255, 128, 0]);
        assert_eq!(pointcloud.normals().unwrap()[0], [0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_read_ply_binary_with_format() {
        let mut file = NamedTempFile::new().unwrap();
        let header = "ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty float nx\nproperty float ny\nproperty float nz\nend_header\n";
        file.write_all(header.as_bytes()).unwrap();

        let mut data = Vec::new();
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&2.0f32.to_le_bytes());
        data.extend_from_slice(&3.0f32.to_le_bytes());
        data.push(255);
        data.push(128);
        data.push(0);
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        file.write_all(&data).unwrap();

        let pointcloud = read_ply_binary_with_format(file.path(), PlyType::XYZRgbNormals).unwrap();
        assert_eq!(pointcloud.len(), 1);
        assert_eq!(pointcloud.points()[0], [1.0, 2.0, 3.0]);
        assert_eq!(pointcloud.colors().unwrap()[0], [255, 128, 0]);
        assert_eq!(pointcloud.normals().unwrap()[0], [0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_read_ply_binary_dynamic_format() {
        let mut file = NamedTempFile::new().unwrap();
        let header = "ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nproperty float intensity\nend_header\n";
        file.write_all(header.as_bytes()).unwrap();

        let mut data = Vec::new();
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&2.0f32.to_le_bytes());
        data.extend_from_slice(&3.0f32.to_le_bytes());
        data.extend_from_slice(&0.5f32.to_le_bytes());
        file.write_all(&data).unwrap();

        let pointcloud = read_ply_binary(file.path()).unwrap();
        assert_eq!(pointcloud.len(), 1);
        assert_eq!(pointcloud.points()[0], [1.0, 2.0, 3.0]);
        assert_eq!(pointcloud.colors().unwrap()[0], [0, 0, 0]);
        assert_eq!(pointcloud.normals().unwrap()[0], [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_format_detection_xyz_rgb_normals() {
        let props = vec![
            PlyPropertyDefinition {
                name: "x".to_string(),
                data_type: PlyDataType::Float32,
            },
            PlyPropertyDefinition {
                name: "y".to_string(),
                data_type: PlyDataType::Float32,
            },
            PlyPropertyDefinition {
                name: "z".to_string(),
                data_type: PlyDataType::Float32,
            },
            PlyPropertyDefinition {
                name: "red".to_string(),
                data_type: PlyDataType::UInt8,
            },
            PlyPropertyDefinition {
                name: "green".to_string(),
                data_type: PlyDataType::UInt8,
            },
            PlyPropertyDefinition {
                name: "blue".to_string(),
                data_type: PlyDataType::UInt8,
            },
            PlyPropertyDefinition {
                name: "nx".to_string(),
                data_type: PlyDataType::Float32,
            },
            PlyPropertyDefinition {
                name: "ny".to_string(),
                data_type: PlyDataType::Float32,
            },
            PlyPropertyDefinition {
                name: "nz".to_string(),
                data_type: PlyDataType::Float32,
            },
        ];

        let detected = PlyType::detect_format(&props).unwrap();
        assert_eq!(detected, PlyType::XYZRgbNormals);
    }
}
