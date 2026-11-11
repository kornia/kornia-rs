//! Registration algorithms (e.g. ICP).

mod icp_vanilla;
pub use icp_vanilla::*;

mod ops;

mod rgbd;
pub use rgbd::*;

#[cfg(test)]
mod synth;
