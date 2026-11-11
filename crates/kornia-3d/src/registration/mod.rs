//! Registration algorithms (e.g. ICP).

mod icp_vanilla;
pub use icp_vanilla::*;

mod ops;

mod rgbd;
pub use rgbd::*;

mod icp;
pub use icp::{
    icp_projective_plane, DepthNoise, IcpPlaneCriteria, IcpPlaneResult, RotationPrior,
    DEFAULT_DEGENERACY_RTOL,
};

#[cfg(test)]
mod synth;
