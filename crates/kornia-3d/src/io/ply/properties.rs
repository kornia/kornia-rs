use serde::Deserialize;

use super::PlyError;

/// Supported PLY properties.
#[derive(Debug, PartialEq)]
pub enum PlyType {
    OpenSplat,
    XYZRgbNormals,
}

/// Trait to represent the PLY property.
pub trait PlyPropertyTrait {
    /// Convert the property to a point.
    fn to_point(&self) -> [f64; 3];

    /// Convert the property to a color.
    fn to_color(&self) -> [u8; 3];

    /// Convert the property to a normal.
    fn to_normal(&self) -> [f64; 3];
}

/// Header of the XYZRgbNormals PLY file format.
///
/// Contains points, colors, and normals.
#[derive(Debug, Deserialize)]
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
        //println!("x: {}, y: {}, z: {}", self.x, self.y, self.z);
        [self.x as f64, self.y as f64, self.z as f64]
    }

    fn to_color(&self) -> [u8; 3] {
        [self.red, self.green, self.blue]
    }

    fn to_normal(&self) -> [f64; 3] {
        [self.nx as f64, self.ny as f64, self.nz as f64]
    }
}

///// Header of the OpenSplat PLY file format.
/// REF: https://github.com/pierotofy/OpenSplat
#[derive(Debug, Deserialize)]
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

/// Enum to represent the PLY property.
pub enum PlyProperty {
    OpenSplat(OpenSplatProperty),
    XYZRgbNormals(XYZRgbNormalsProperty),
}

impl PlyType {
    /// Deserialize the property from the buffer.
    pub fn deserialize(&self, buffer: &[u8]) -> Result<PlyProperty, PlyError> {
        match self {
            PlyType::OpenSplat => {
                let property: OpenSplatProperty = bincode::deserialize(buffer)?;
                Ok(PlyProperty::OpenSplat(property))
            }
            PlyType::XYZRgbNormals => {
                let property: XYZRgbNormalsProperty = bincode::deserialize(buffer)?;
                Ok(PlyProperty::XYZRgbNormals(property))
            }
        }
    }

    pub fn size_of(&self) -> usize {
        match self {
            PlyType::OpenSplat => std::mem::size_of::<OpenSplatProperty>(),
            PlyType::XYZRgbNormals => std::mem::size_of::<XYZRgbNormalsProperty>(),
        }
    }
}

impl PlyPropertyTrait for PlyProperty {
    fn to_point(&self) -> [f64; 3] {
        match self {
            PlyProperty::OpenSplat(property) => property.to_point(),
            PlyProperty::XYZRgbNormals(property) => property.to_point(),
        }
    }

    fn to_color(&self) -> [u8; 3] {
        match self {
            PlyProperty::OpenSplat(property) => property.to_color(),
            PlyProperty::XYZRgbNormals(property) => property.to_color(),
        }
    }

    fn to_normal(&self) -> [f64; 3] {
        match self {
            PlyProperty::OpenSplat(property) => property.to_normal(),
            PlyProperty::XYZRgbNormals(property) => property.to_normal(),
        }
    }
}
