pub mod core;
pub mod losses;
pub mod solvers;

#[cfg(test)]
mod tests_l2_baseline;

#[cfg(test)]
mod tests_robust_losses;

pub use core::{Factor, FactorError, FactorResult, LinearizationResult, PriorFactor};
pub use core::{Problem, ProblemError, Variable, VariableType};
pub use losses::{CauchyLoss, HuberLoss, IdentityLoss, RobustLoss};
pub use solvers::{
    LevenbergMarquardt, LinearSystemBuilder, OptimizerError, OptimizerResult, OptimizerState,
    TerminationReason, VariableLayout,
};
