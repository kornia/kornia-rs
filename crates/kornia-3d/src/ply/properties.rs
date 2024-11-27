use serde::Deserialize;

/// Supported PLY properties.
#[derive(Debug, PartialEq)]
pub enum PlyProperty {
    OpenSplat,
}

impl PlyProperty {
    /// Get the size of the property in bytes.
    pub fn size_of(&self) -> usize {
        match self {
            PlyProperty::OpenSplat => std::mem::size_of::<OpenSplatProperty>(),
        }
    }
}

/// Header of the OpenSplat PLY file format.
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
