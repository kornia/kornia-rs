//! Registration algorithms (e.g. ICP).

mod icp_vanilla;
pub use icp_vanilla::*;

mod ops;

mod rgbd;
pub use rgbd::*;

mod icp;
pub use icp::{
    icp_projective_plane, Association, DepthNoise, IcpPlaneCriteria, IcpPlaneResult, PosePrior,
    RotationPrior, DEFAULT_DEGENERACY_RTOL, HYBRID_SWITCH_NORM, MAX_BILINEAR_DEPTH_JUMP,
};

#[cfg(test)]
mod synth;
