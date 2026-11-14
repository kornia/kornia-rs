//! Registration algorithms (e.g. ICP).

mod icp_vanilla;
pub use icp_vanilla::*;

mod ops;

mod rgbd;
pub use rgbd::*;

mod icp;
pub use icp::{
    icp_projective_plane, icp_projective_plane_with, Association, DepthNoise, IcpKernel,
    IcpPlaneCriteria, IcpPlaneResult, IcpWorkspace, PosePrior, RotationPrior,
    DEFAULT_DEGENERACY_RTOL, HYBRID_SWITCH_NORM, MAX_BILINEAR_DEPTH_JUMP,
};

#[cfg(test)]
mod synth;
