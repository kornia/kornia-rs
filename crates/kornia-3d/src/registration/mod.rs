//! Registration algorithms (e.g. ICP).

mod icp_vanilla;
pub use icp_vanilla::*;

mod icp;
pub use icp::*;

mod rgbd;
pub use rgbd::*;

#[cfg(test)]
pub(crate) mod synth;

mod ops;
