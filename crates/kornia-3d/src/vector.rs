/// Simple 3D vector with x, y, and z coordinates as single precision floats.
#[derive(Debug, Clone, PartialEq)]
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

    /// Add two Vec3 vectors
    pub fn add(&self, other: &Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }

    /// Scale Vec3 vector with a factor
    pub fn scale(&self, factor: f32) -> Self {
        Self {
            x: self.x * factor,
            y: self.y * factor,
            z: self.z * factor,
        }
    }
}

/// Simple 3D vector with x, y, and z coordinates as double precision floats.
#[derive(Debug, Clone, PartialEq)]
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

    /// Add two DVec3 vectors.
    pub fn add(&self, other: &Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }

    /// Scale a DVec3 vector by a scalar.
    pub fn scale(&self, factor: f64) -> Self {
        Self {
            x: self.x * factor,
            y: self.y * factor,
            z: self.z * factor,
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
    fn test_vec3_add() {
        let v1 = Vec3::from_array(&[1.0, 2.0, 3.0]);
        let v2 = Vec3::from_array(&[4.0, 5.0, 6.0]);
        let result = v1.add(&v2);
        assert_eq!(result.x, 5.0);
        assert_eq!(result.y, 7.0);
        assert_eq!(result.z, 9.0);
    }

    #[test]
    fn test_vec3_scale() {
        let v = Vec3::from_array(&[1.0, 2.0, 3.0]);
        let result = v.scale(2.0);
        assert_eq!(result.x, 2.0);
        assert_eq!(result.y, 4.0);
        assert_eq!(result.z, 6.0);
    }

    #[test]
    fn test_dvec3_from_array() {
        let array = [1.0, 2.0, 3.0];
        let vec = DVec3::from_array(&array);
        assert_eq!(vec.x, 1.0);
        assert_eq!(vec.y, 2.0);
        assert_eq!(vec.z, 3.0);
    }

    #[test]
    fn test_dvec3_add() {
        let v1 = DVec3::from_array(&[1.0, 2.0, 3.0]);
        let v2 = DVec3::from_array(&[4.0, 5.0, 6.0]);
        let result = v1.add(&v2);
        assert_eq!(result.x, 5.0);
        assert_eq!(result.y, 7.0);
        assert_eq!(result.z, 9.0);
    }

    #[test]
    fn test_dvec3_scale() {
        let v = DVec3::from_array(&[1.0, 2.0, 3.0]);
        let result = v.scale(2.0);
        assert_eq!(result.x, 2.0);
        assert_eq!(result.y, 4.0);
        assert_eq!(result.z, 6.0);
    }
    
}