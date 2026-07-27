mod responses;
pub use responses::*;

mod fast;
pub use fast::*;

mod orb;
pub use orb::*;

mod r#match;
pub use r#match::*;

mod cells;
pub use cells::{
    fast_detect_cells_u8, fast_detect_pyramid_u8, fast_detect_rect_u8, CellDetectConfig,
    CellKeypoint, FastCorner, PyramidKeypoint, Rect,
};

mod sift;
/// The SIFT API: the two entry points, the matcher, and the types they exchange.
///
/// The per-stage kernels (blur, extremum search, orientation assignment,
/// descriptor) are deliberately *not* re-exported. They take raw slices plus
/// explicit strides and only make sense in the order the pipeline calls them,
/// so publishing them would commit this crate to signatures that exist to serve
/// one caller.
///
/// `SiftConfigError` does belong here: it is the error half of both entry
/// points' `Result`, so without it a caller can neither match on a failure nor
/// implement `From` to carry one through `?`.
pub use sift::{
    l2_sq, l2_sq_scalar, sift_detect_and_compute, sift_detect_with, sift_match_descriptors,
    sift_match_descriptors_scalar, FirstOctave, SiftConfig, SiftConfigError, SiftFeatures,
    SiftKeypoint, SiftWorkspace, DESCR_LEN, ORI_HIST_BINS, SIFT_IMG_BORDER, SIFT_MAX_INTERP_STEPS,
};

/// Numerics both backends must agree on bit for bit.
///
/// `cuda::sift` names these instead of re-declaring the literals. The copies
/// drifted once already — see `SiftCudaConfig::shared_config` for the
/// `powi`/`powf` divergence that motivated the shared module — so a constant
/// that appears in both pipelines has exactly one definition, here.
#[cfg(feature = "cuda")]
pub(crate) use sift::{
    gaussian_kernel_f32, gaussian_ksize, sift_final_order, DESCR_HIST_BINS, DESCR_MAG_THR,
    DESCR_SCL_FCTR, DESCR_WIDTH, INT_DESCR_FCTR, ORI_PEAK_RATIO, ORI_RADIUS, ORI_SIG_FCTR,
};
