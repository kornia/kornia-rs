use serde::Deserialize;

use super::PlyError;

#[derive(Debug, PartialEq, Clone)]
pub enum PlyType {
    OpenSplat,
    XYZRgbNormals,
    Dynamic(Vec<PlyPropertyDefinition>),
}

#[derive(Debug, PartialEq, Clone)]
pub struct PlyPropertyDefinition {
    pub name: String,
    pub data_type: PlyDataType,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum PlyDataType {
    Float32,
    Float64,
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
}

impl PlyDataType {
    pub fn size(&self) -> usize {
        match self {
            PlyDataType::Float32 | PlyDataType::Int32 | PlyDataType::UInt32 => 4,
            PlyDataType::Float64 => 8,
            PlyDataType::Int16 | PlyDataType::UInt16 => 2,
            PlyDataType::Int8 | PlyDataType::UInt8 => 1,
        }
    }
}

pub trait PlyPropertyTrait {
    fn to_point(&self) -> [f64; 3];
    fn to_color(&self) -> [u8; 3];
    fn to_normal(&self) -> [f64; 3];
}

#[repr(C, packed)]
#[derive(Debug, Deserialize, bincode::Decode)]
pub struct XYZRgbNormalsProperty {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub red: u8,
    pub green: u8,
    pub blue: u8,
    pub nx: f32,
    pub ny: f32,
    pub nz: f32,
}

impl PlyPropertyTrait for XYZRgbNormalsProperty {
    fn to_point(&self) -> [f64; 3] {
        [self.x as f64, self.y as f64, self.z as f64]
    }

    fn to_color(&self) -> [u8; 3] {
        [self.red, self.green, self.blue]
    }

    fn to_normal(&self) -> [f64; 3] {
        [self.nx as f64, self.ny as f64, self.nz as f64]
    }
}

#[repr(C, packed)]
#[derive(Debug, Deserialize, bincode::Decode)]
pub struct OpenSplatProperty {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub nx: f32,
    pub ny: f32,
    pub nz: f32,
    pub f_dc_0: f32,
    pub f_dc_1: f32,
    pub f_dc_2: f32,
    pub f_rest_0: f32,
    pub f_rest_1: f32,
    pub f_rest_2: f32,
    pub f_rest_3: f32,
    pub f_rest_4: f32,
    pub f_rest_5: f32,
    pub f_rest_6: f32,
    pub f_rest_7: f32,
    pub f_rest_8: f32,
    pub f_rest_9: f32,
    pub f_rest_10: f32,
    pub f_rest_11: f32,
    pub f_rest_12: f32,
    pub f_rest_13: f32,
    pub f_rest_14: f32,
    pub f_rest_15: f32,
    pub f_rest_16: f32,
    pub f_rest_17: f32,
    pub f_rest_18: f32,
    pub f_rest_19: f32,
    pub f_rest_20: f32,
    pub f_rest_21: f32,
    pub f_rest_22: f32,
    pub f_rest_23: f32,
    pub f_rest_24: f32,
    pub f_rest_25: f32,
    pub f_rest_26: f32,
    pub f_rest_27: f32,
    pub f_rest_28: f32,
    pub f_rest_29: f32,
    pub f_rest_30: f32,
    pub f_rest_31: f32,
    pub f_rest_32: f32,
    pub f_rest_33: f32,
    pub f_rest_34: f32,
    pub f_rest_35: f32,
    pub f_rest_36: f32,
    pub f_rest_37: f32,
    pub f_rest_38: f32,
    pub f_rest_39: f32,
    pub f_rest_40: f32,
    pub f_rest_41: f32,
    pub f_rest_42: f32,
    pub f_rest_43: f32,
    pub f_rest_44: f32,
    pub opacity: f32,
    pub scale_0: f32,
    pub scale_1: f32,
    pub scale_2: f32,
    pub rot_0: f32,
    pub rot_1: f32,
    pub rot_2: f32,
    pub rot_3: f32,
}

impl PlyPropertyTrait for OpenSplatProperty {
    fn to_point(&self) -> [f64; 3] {
        [self.x as f64, self.y as f64, self.z as f64]
    }

    fn to_color(&self) -> [u8; 3] {
        [
            (self.f_dc_0 * 255.0) as u8,
            (self.f_dc_1 * 255.0) as u8,
            (self.f_dc_2 * 255.0) as u8,
        ]
    }

    fn to_normal(&self) -> [f64; 3] {
        [self.nx as f64, self.ny as f64, self.nz as f64]
    }
}

/// Dynamic PLY property that can handle arbitrary schemas
#[derive(Debug)]
pub struct DynamicProperty {
    pub properties: Vec<(String, DynamicPropertyValue)>,
}

#[derive(Debug, Clone, Copy)]
pub enum DynamicPropertyValue {
    Float32(f32),
    Float64(f64),
    Int8(i8),
    UInt8(u8),
    Int16(i16),
    UInt16(u16),
    Int32(i32),
    UInt32(u32),
}

impl DynamicProperty {
    fn parse_from_buffer(buffer: &[u8], schema: &[PlyPropertyDefinition]) -> Result<Self, PlyError> {
        let mut properties = Vec::new();
        let mut offset = 0;

        for prop_def in schema {
            let size = prop_def.data_type.size();
            if offset + size > buffer.len() {
                return Err(PlyError::UnsupportedProperty);
            }

            let value = match prop_def.data_type {
                PlyDataType::Float32 => {
                    let bytes = buffer[offset..offset + 4].try_into().unwrap();
                    DynamicPropertyValue::Float32(f32::from_le_bytes(bytes))
                }
                PlyDataType::Float64 => {
                    let bytes = buffer[offset..offset + 8].try_into().unwrap();
                    DynamicPropertyValue::Float64(f64::from_le_bytes(bytes))
                }
                PlyDataType::Int8 => DynamicPropertyValue::Int8(buffer[offset] as i8),
                PlyDataType::UInt8 => DynamicPropertyValue::UInt8(buffer[offset]),
                PlyDataType::Int16 => {
                    let bytes = buffer[offset..offset + 2].try_into().unwrap();
                    DynamicPropertyValue::Int16(i16::from_le_bytes(bytes))
                }
                PlyDataType::UInt16 => {
                    let bytes = buffer[offset..offset + 2].try_into().unwrap();
                    DynamicPropertyValue::UInt16(u16::from_le_bytes(bytes))
                }
                PlyDataType::Int32 => {
                    let bytes = buffer[offset..offset + 4].try_into().unwrap();
                    DynamicPropertyValue::Int32(i32::from_le_bytes(bytes))
                }
                PlyDataType::UInt32 => {
                    let bytes = buffer[offset..offset + 4].try_into().unwrap();
                    DynamicPropertyValue::UInt32(u32::from_le_bytes(bytes))
                }
            };

            properties.push((prop_def.name.clone(), value));
            offset += size;
        }

        Ok(DynamicProperty { properties })
    }

    fn get_float(&self, name: &str) -> f64 {
        self.properties
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| match v {
                DynamicPropertyValue::Float32(v) => *v as f64,
                DynamicPropertyValue::Float64(v) => *v,
                DynamicPropertyValue::Int8(v) => *v as f64,
                DynamicPropertyValue::UInt8(v) => *v as f64,
                DynamicPropertyValue::Int16(v) => *v as f64,
                DynamicPropertyValue::UInt16(v) => *v as f64,
                DynamicPropertyValue::Int32(v) => *v as f64,
                DynamicPropertyValue::UInt32(v) => *v as f64,
            })
            .unwrap_or(0.0)
    }

    fn get_u8(&self, name: &str) -> u8 {
        self.properties
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| match v {
                DynamicPropertyValue::UInt8(v) => *v,
                DynamicPropertyValue::Int8(v) => *v as u8,
                DynamicPropertyValue::Float32(v) => (*v * 255.0) as u8,
                _ => 0,
            })
            .unwrap_or(0)
    }

    pub fn to_point(&self) -> [f64; 3] {
        [self.get_float("x"), self.get_float("y"), self.get_float("z")]
    }

    pub fn to_color(&self) -> [u8; 3] {
        [self.get_u8("red"), self.get_u8("green"), self.get_u8("blue")]
    }

    pub fn to_normal(&self) -> [f64; 3] {
        [self.get_float("nx"), self.get_float("ny"), self.get_float("nz")]
    }
}

pub enum PlyProperty {
    OpenSplat(Box<OpenSplatProperty>),
    XYZRgbNormals(XYZRgbNormalsProperty),
    Dynamic(DynamicProperty),
}

impl PlyType {
    pub fn deserialize(&self, buffer: &[u8]) -> Result<PlyProperty, PlyError> {
        match self {
            PlyType::OpenSplat => {
                let (property, _): (OpenSplatProperty, usize) =
                    bincode::decode_from_slice(buffer, bincode::config::standard())?;
                Ok(PlyProperty::OpenSplat(Box::new(property)))
            }
            PlyType::XYZRgbNormals => {
                let (property, _): (XYZRgbNormalsProperty, usize) =
                    bincode::decode_from_slice(buffer, bincode::config::standard())?;
                Ok(PlyProperty::XYZRgbNormals(property))
            }
            PlyType::Dynamic(ref schema) => {
                let dynamic_property = DynamicProperty::parse_from_buffer(buffer, schema)?;
                Ok(PlyProperty::Dynamic(dynamic_property))
            }
        }
    }

    pub fn size_of(&self) -> usize {
        match self {
            PlyType::OpenSplat => std::mem::size_of::<OpenSplatProperty>(),
            PlyType::XYZRgbNormals => std::mem::size_of::<XYZRgbNormalsProperty>(),
            PlyType::Dynamic(ref props) => props.iter().map(|p| p.data_type.size()).sum(),
        }
    }

    pub fn detect_format(properties: &[PlyPropertyDefinition]) -> Result<Self, PlyError> {
        if properties.len() == 9 {
            let expected_names = ["x", "y", "z", "red", "green", "blue", "nx", "ny", "nz"];
            if properties.iter().zip(expected_names.iter()).all(|(p, expected)| &p.name == expected) {
                return Ok(PlyType::XYZRgbNormals);
            }
        }

        if properties.len() == 62 {
             // Simplified check for OpenSplat
             if properties[0].name == "x" && properties[6].name == "f_dc_0" {
                 return Ok(PlyType::OpenSplat);
             }
        }

        Ok(PlyType::Dynamic(properties.to_vec()))
    }
}

impl PlyPropertyTrait for PlyProperty {
    fn to_point(&self) -> [f64; 3] {
        match self {
            PlyProperty::OpenSplat(property) => property.to_point(),
            PlyProperty::XYZRgbNormals(property) => property.to_point(),
            PlyProperty::Dynamic(property) => property.to_point(),
        }
    }

    fn to_color(&self) -> [u8; 3] {
        match self {
            PlyProperty::OpenSplat(property) => property.to_color(),
            PlyProperty::XYZRgbNormals(property) => property.to_color(),
            PlyProperty::Dynamic(property) => property.to_color(),
        }
    }

    fn to_normal(&self) -> [f64; 3] {
        match self {
            PlyProperty::OpenSplat(property) => property.to_normal(),
            PlyProperty::XYZRgbNormals(property) => property.to_normal(),
            PlyProperty::Dynamic(property) => property.to_normal(),
        }
    }
}
