/// Simple 3D vector with x, y, and z coordinates as single precision floats.
#[derive(Debug, Clone)]
pub struct Vec3 {
    /// x coordinate
    pub x: f32,
    /// y coordinate
    pub y: f32,
    /// z coordinate
    pub z: f32,
}

impl Vec3 {
    /// Create a new Vec3 from an array of 3 f32 values.
    pub fn from_array(array: &[f32; 3]) -> Self {
        Self {
            x: array[0],
            y: array[1],
            z: array[2],
        }
    }
}

/// Simple 3D vector with x, y, and z coordinates as double precision floats.
#[derive(Debug, Clone)]
pub struct DVec3 {
    /// x coordinate
    pub x: f64,
    /// y coordinate
    pub y: f64,
    /// z coordinate
    pub z: f64,
}

impl DVec3 {
    /// Create a new DVec3 from an array of 3 f64 values.
    pub fn from_array(array: &[f64; 3]) -> Self {
        Self {
            x: array[0],
            y: array[1],
            z: array[2],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec3_from_array() {
        let array = [1.0, 2.0, 3.0];
        let vec = Vec3::from_array(&array);
        assert_eq!(vec.x, 1.0);
        assert_eq!(vec.y, 2.0);
        assert_eq!(vec.z, 3.0);
    }

    #[test]
    fn test_dvec3_from_array() {
        let array = [1.0, 2.0, 3.0];
        let vec = DVec3::from_array(&array);
        assert_eq!(vec.x, 1.0);
        assert_eq!(vec.y, 2.0);
        assert_eq!(vec.z, 3.0);
    }
}
