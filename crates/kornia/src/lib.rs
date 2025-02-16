#![doc = include_str!(concat!("../", env!("CARGO_PKG_README")))]

/// Core tensor library.
#[doc(inline)]
pub use kornia_tensor as tensor;

/// Core tensor operations.
#[doc(inline)]
pub use kornia_tensor_ops as tensor_ops;

/// Image processing.
#[doc(inline)]
pub use kornia_image as image;

#[doc(inline)]
pub use kornia_imgproc as imgproc;

#[doc(inline)]
pub use kornia_io as io;

#[doc(inline)]
pub use kornia_3d as k3d;

#[doc(inline)]
pub use kornia_icp as icp;
