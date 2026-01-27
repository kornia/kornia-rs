use crate::{
    param::{Param, ParamError},
    SE2F32, SE3F32, SO2F32, SO3F32,
};

/// Type of variable in the optimization problem.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VariableType {
    /// Euclidean space variable (e.g., 3 for Vec3F32, 2 for Vec2F32)
    Euclidean(usize),
    /// SE(3) - 3D rigid transformation (7 params, 6-dim tangent)
    SE3,
    /// SE(2) - 2D rigid transformation (4 params, 3-dim tangent)
    SE2,
    /// SO(3) - 3D rotation (4 params quaternion, 3-dim tangent)
    SO3,
    /// SO(2) - 2D rotation (2 params complex, 1-dim tangent)
    SO2,
}

impl VariableType {
    /// Global storage dimension (length of the parameter block).
    pub fn global_dim(&self) -> usize {
        match self {
            VariableType::Euclidean(n) => *n,
            VariableType::SE3 => 7,
            VariableType::SE2 => 4,
            VariableType::SO3 => 4,
            VariableType::SO2 => 2,
        }
    }

    /// Local update (tangent space) dimension used by the optimizer.
    pub fn local_dim(&self) -> usize {
        match self {
            VariableType::Euclidean(n) => *n,
            VariableType::SE3 => 6,
            VariableType::SE2 => 3,
            VariableType::SO3 => 3,
            VariableType::SO2 => 1,
        }
    }

    /// Apply a local update to a global parameter block.
    pub fn plus(&self, x: &[f32], delta: &[f32], out: &mut [f32]) -> Result<(), ParamError> {
        match self {
            VariableType::Euclidean(n) => {
                if x.len() < *n {
                    return Err(ParamError::WrongGlobalSize {
                        expected: *n,
                        got: x.len(),
                    });
                }
                if delta.len() < *n {
                    return Err(ParamError::WrongLocalSize {
                        expected: *n,
                        got: delta.len(),
                    });
                }
                if out.len() < *n {
                    return Err(ParamError::WrongOutSize {
                        expected: *n,
                        got: out.len(),
                    });
                }

                for i in 0..*n {
                    out[i] = x[i] + delta[i];
                }
                Ok(())
            }
            VariableType::SE3 => SE3F32::plus(x, delta, out),
            VariableType::SE2 => SE2F32::plus(x, delta, out),
            VariableType::SO3 => SO3F32::plus(x, delta, out),
            VariableType::SO2 => SO2F32::plus(x, delta, out),
        }
    }

    /// Apply a local update in-place.
    pub fn apply_plus(&self, values: &mut Vec<f32>, delta: &[f32]) -> Result<(), ParamError> {
        match self {
            VariableType::Euclidean(n) => {
                if values.len() < *n {
                    return Err(ParamError::WrongGlobalSize {
                        expected: *n,
                        got: values.len(),
                    });
                }
                if delta.len() < *n {
                    return Err(ParamError::WrongLocalSize {
                        expected: *n,
                        got: delta.len(),
                    });
                }
                for i in 0..*n {
                    values[i] += delta[i];
                }
                Ok(())
            }
            _ => {
                let mut out = vec![0.0f32; self.global_dim()];
                self.plus(values, delta, &mut out)?;
                *values = out;
                Ok(())
            }
        }
    }
}

/// A variable in an optimization problem.
#[derive(Debug, Clone)]
pub struct Variable {
    /// Name of the variable (used for referencing in factors)
    pub name: String,
    /// Type of the variable
    pub var_type: VariableType,
    /// Current parameter values
    pub values: Vec<f32>,
}

impl Variable {
    /// Create a new variable.
    #[inline]
    pub fn new(name: impl Into<String>, var_type: VariableType, values: Vec<f32>) -> Self {
        Self {
            name: name.into(),
            var_type,
            values,
        }
    }

    /// Create a new Euclidean variable.
    pub fn euclidean(name: impl Into<String>, dim: usize) -> Self {
        Self {
            name: name.into(),
            var_type: VariableType::Euclidean(dim),
            values: vec![0.0; dim],
        }
    }

    #[inline]
    pub fn se3(name: impl Into<String>, values: Vec<f32>) -> Self {
        Self::new(name, VariableType::SE3, values)
    }

    #[inline]
    pub fn global_dim(&self) -> usize {
        self.var_type.global_dim()
    }

    #[inline]
    pub fn local_dim(&self) -> usize {
        self.var_type.local_dim()
    }
}
