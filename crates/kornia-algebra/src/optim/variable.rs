//! Variable types for factor graph optimization
//!
//! Variables represent parameters to be optimized in an optimization problem.
//! They can be Euclidean (e.g., 3D points) or Lie group elements (e.g., SE(3) poses).

/// Type of variable in the optimization problem.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VariableType {
    /// Euclidean space variable (e.g., 3 for Vec3F32, 2 for Vec2F32)
    Euclidean(usize),
    /// SE(3) - 3D rigid transformation (7 DOF: quaternion + translation)
    SE3,
    /// SE(2) - 2D rigid transformation (4 DOF: angle + translation)
    SE2,
    /// SO(3) - 3D rotation (3 DOF: axis-angle)
    SO3,
    /// SO(2) - 2D rotation (1 DOF: angle)
    SO2,
}

impl VariableType {
    /// Get the dimension (degrees of freedom) for this variable type.
    pub fn dim(&self) -> usize {
        match self {
            VariableType::Euclidean(n) => *n,
            VariableType::SE3 => 7,
            VariableType::SE2 => 4,
            VariableType::SO3 => 3,
            VariableType::SO2 => 1,
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
    /// Create a new Euclidean variable.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the variable
    /// * `dim` - Dimension (e.g., 3 for a 3D point)
    pub fn euclidean(name: impl Into<String>, dim: usize) -> Self {
        Self {
            name: name.into(),
            var_type: VariableType::Euclidean(dim),
            values: vec![0.0; dim],
        }
    }

    /// Get the dimension (degrees of freedom) of this variable.
    pub fn dim(&self) -> usize {
        self.var_type.dim()
    }
}
