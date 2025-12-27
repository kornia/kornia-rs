mod apriltag;
mod color;
mod dlpack;
mod enhance;
mod histogram;
mod icp;
mod image;
mod io;
mod pointcloud;
mod resize;
mod warp;

use crate::icp::{PyICPConvergenceCriteria, PyICPResult};
use crate::image::{PyImageLayout, PyImageSize, PyPixelFormat};
use crate::io::jpegturbo::{PyImageDecoder, PyImageEncoder};
use pyo3::prelude::*;

pub fn get_version() -> String {
 let version = env!("CARGO_PKG_VERSION").to_string();
 // cargo uses "1.0-alpha1" etc. while python uses "1.0.0a1", this is not full compatibility,
 // but it's good enough for now
 // see https://docs.rs/semver/1.0.9/semver/struct.Version.html#method.parse for rust spec
 // see https://peps.python.org/pep-0440/ for python spec
 // it seems the dot after "alpha/beta" e.g. "-alpha.1" is not necessary, hence why this works
 version.replace("-alpha", "a").replace("-beta", "b")
}

#[pymodule(gil_used = false)]
pub fn kornia_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
 m.add("__version__", get_version())?;
 
 // Add backward compatibility functions at top level
 m.add_function(wrap_pyfunction!(color::rgb_from_gray, m)?)?;
 m.add_function(wrap_pyfunction!(color::rgb_from_rgba, m)?)?;
 m.add_function(wrap_pyfunction!(color::rgb_from_bgra, m)?)?;
 m.add_function(wrap_pyfunction!(color::bgr_from_rgb, m)?)?;
 m.add_function(wrap_pyfunction!(color::gray_from_rgb, m)?)?;
 m.add_function(wrap_pyfunction!(enhance::add_weighted, m)?)?;
 m.add_function(wrap_pyfunction!(histogram::compute_histogram, m)?)?;
 m.add_function(wrap_pyfunction!(icp::icp_vanilla, m)?)?;
 m.add_function(wrap_pyfunction!(io::functional::read_image_any, m)?)?;
 m.add_function(wrap_pyfunction!(io::functional::read_image, m)?)?;
 m.add_function(wrap_pyfunction!(io::png::decode_image_png_u8, m)?)?;
 m.add_function(wrap_pyfunction!(io::png::decode_image_png_u16, m)?)?;
 m.add_function(wrap_pyfunction!(io::png::read_image_png_u8, m)?)?;
 m.add_function(wrap_pyfunction!(io::png::read_image_png_u16, m)?)?;
 m.add_function(wrap_pyfunction!(io::png::write_image_png_u8, m)?)?;
 m.add_function(wrap_pyfunction!(io::png::write_image_png_u16, m)?)?;
 m.add_function(wrap_pyfunction!(io::jpeg::decode_image_jpeg, m)?)?;
 m.add_function(wrap_pyfunction!(io::jpeg::read_image_jpeg, m)?)?;
 m.add_function(wrap_pyfunction!(io::jpeg::write_image_jpeg, m)?)?;
 m.add_function(wrap_pyfunction!(io::jpeg::encode_image_jpeg, m)?)?;
 m.add_function(wrap_pyfunction!(io::tiff::read_image_tiff_f32, m)?)?;
 m.add_function(wrap_pyfunction!(io::tiff::read_image_tiff_u8, m)?)?;
 m.add_function(wrap_pyfunction!(io::tiff::read_image_tiff_u16, m)?)?;
 m.add_function(wrap_pyfunction!(io::tiff::write_image_tiff_f32, m)?)?;
 m.add_function(wrap_pyfunction!(io::tiff::write_image_tiff_u8, m)?)?;
 m.add_function(wrap_pyfunction!(io::tiff::write_image_tiff_u16, m)?)?;
 m.add_function(wrap_pyfunction!(io::jpegturbo::decode_image_jpegturbo, m)?)?;
 m.add_function(wrap_pyfunction!(io::jpegturbo::read_image_jpegturbo, m)?)?;
 m.add_function(wrap_pyfunction!(io::jpegturbo::write_image_jpegturbo, m)?)?;
 m.add_function(wrap_pyfunction!(resize::resize, m)?)?;
 m.add_function(wrap_pyfunction!(warp::warp_affine, m)?)?;
 m.add_function(wrap_pyfunction!(warp::warp_perspective, m)?)?;
 m.add_class::<PyImageDecoder>()?;
 m.add_class::<PyImageEncoder>()?;
 m.add_class::<PyICPConvergenceCriteria>()?;
 m.add_class::<PyICPResult>()?;
 
 // Image submodule
 let image_mod = PyModule::new(m.py(), "image")?;
 image_mod.add_class::<PyImageSize>()?;
 image_mod.add_class::<PyPixelFormat>()?;
 image_mod.add_class::<PyImageLayout>()?;
 m.add_submodule(&image_mod)?;
 
 // IO submodule
 let io_mod = PyModule::new(m.py(), "io")?;
 
 // IO.functional submodule
 let io_functional_mod = PyModule::new(io_mod.py(), "functional")?;
 io_functional_mod.add_function(wrap_pyfunction!(io::functional::read_image_any, io_functional_mod.py())?)?;
 io_functional_mod.add_function(wrap_pyfunction!(io::functional::read_image, io_functional_mod.py())?)?;
 io_mod.add_submodule(&io_functional_mod)?;
 
 // IO.png submodule
 let io_png_mod = PyModule::new(io_mod.py(), "png")?;
 io_png_mod.add_function(wrap_pyfunction!(io::png::decode_image_png_u8, io_png_mod.py())?)?;
 io_png_mod.add_function(wrap_pyfunction!(io::png::decode_image_png_u16, io_png_mod.py())?)?;
 io_png_mod.add_function(wrap_pyfunction!(io::png::read_image_png_u8, io_png_mod.py())?)?;
 io_png_mod.add_function(wrap_pyfunction!(io::png::read_image_png_u16, io_png_mod.py())?)?;
 io_png_mod.add_function(wrap_pyfunction!(io::png::write_image_png_u8, io_png_mod.py())?)?;
 io_png_mod.add_function(wrap_pyfunction!(io::png::write_image_png_u16, io_png_mod.py())?)?;
 io_mod.add_submodule(&io_png_mod)?;
 
 // IO.jpeg submodule
 let io_jpeg_mod = PyModule::new(io_mod.py(), "jpeg")?;
 io_jpeg_mod.add_function(wrap_pyfunction!(io::jpeg::decode_image_jpeg, io_jpeg_mod.py())?)?;
 io_jpeg_mod.add_function(wrap_pyfunction!(io::jpeg::read_image_jpeg, io_jpeg_mod.py())?)?;
 io_jpeg_mod.add_function(wrap_pyfunction!(io::jpeg::write_image_jpeg, io_jpeg_mod.py())?)?;
 io_jpeg_mod.add_function(wrap_pyfunction!(io::jpeg::encode_image_jpeg, io_jpeg_mod.py())?)?;
 io_mod.add_submodule(&io_jpeg_mod)?;
 
 // IO.tiff submodule
 let io_tiff_mod = PyModule::new(io_mod.py(), "tiff")?;
 io_tiff_mod.add_function(wrap_pyfunction!(io::tiff::read_image_tiff_f32, io_tiff_mod.py())?)?;
 io_tiff_mod.add_function(wrap_pyfunction!(io::tiff::read_image_tiff_u8, io_tiff_mod.py())?)?;
 io_tiff_mod.add_function(wrap_pyfunction!(io::tiff::read_image_tiff_u16, io_tiff_mod.py())?)?;
 io_tiff_mod.add_function(wrap_pyfunction!(io::tiff::write_image_tiff_f32, io_tiff_mod.py())?)?;
 io_tiff_mod.add_function(wrap_pyfunction!(io::tiff::write_image_tiff_u8, io_tiff_mod.py())?)?;
 io_tiff_mod.add_function(wrap_pyfunction!(io::tiff::write_image_tiff_u16, io_tiff_mod.py())?)?;
 io_mod.add_submodule(&io_tiff_mod)?;
 
 // IO.jpegturbo submodule
 let io_jpegturbo_mod = PyModule::new(io_mod.py(), "jpegturbo")?;
 io_jpegturbo_mod.add_function(wrap_pyfunction!(io::jpegturbo::decode_image_jpegturbo, io_jpegturbo_mod.py())?)?;
 io_jpegturbo_mod.add_function(wrap_pyfunction!(io::jpegturbo::read_image_jpegturbo, io_jpegturbo_mod.py())?)?;
 io_jpegturbo_mod.add_function(wrap_pyfunction!(io::jpegturbo::write_image_jpegturbo, io_jpegturbo_mod.py())?)?;
 io_mod.add_submodule(&io_jpegturbo_mod)?;
 m.add_submodule(&io_mod)?;
 
 // Color submodule
 let color_mod = PyModule::new(m.py(), "color")?;
 color_mod.add_function(wrap_pyfunction!(color::rgb_from_gray, color_mod.py())?)?;
 color_mod.add_function(wrap_pyfunction!(color::rgb_from_rgba, color_mod.py())?)?;
 color_mod.add_function(wrap_pyfunction!(color::rgb_from_bgra, color_mod.py())?)?;
 color_mod.add_function(wrap_pyfunction!(color::bgr_from_rgb, color_mod.py())?)?;
 color_mod.add_function(wrap_pyfunction!(color::gray_from_rgb, color_mod.py())?)?;
 m.add_submodule(&color_mod)?;
 
 // Enhance submodule
 let enhance_mod = PyModule::new(m.py(), "enhance")?;
 enhance_mod.add_function(wrap_pyfunction!(enhance::add_weighted, enhance_mod.py())?)?;
 m.add_submodule(&enhance_mod)?;
 
 // Histogram submodule
 let histogram_mod = PyModule::new(m.py(), "histogram")?;
 histogram_mod.add_function(wrap_pyfunction!(histogram::compute_histogram, histogram_mod.py())?)?;
 m.add_submodule(&histogram_mod)?;
 
 // Resize submodule
 let resize_mod = PyModule::new(m.py(), "resize")?;
 resize_mod.add_function(wrap_pyfunction!(resize::resize, resize_mod.py())?)?;
 m.add_submodule(&resize_mod)?;
 
 // Warp submodule
 let warp_mod = PyModule::new(m.py(), "warp")?;
 warp_mod.add_function(wrap_pyfunction!(warp::warp_affine, warp_mod.py())?)?;
 warp_mod.add_function(wrap_pyfunction!(warp::warp_perspective, warp_mod.py())?)?;
 m.add_submodule(&warp_mod)?;
 
 // ICP submodule
 let icp_mod = PyModule::new(m.py(), "icp")?;
 icp_mod.add_function(wrap_pyfunction!(icp::icp_vanilla, icp_mod.py())?)?;
 icp_mod.add_class::<PyICPConvergenceCriteria>()?;
 icp_mod.add_class::<PyICPResult>()?;
 m.add_submodule(&icp_mod)?;
 
 // AprilTag submodule
 let apriltag_mod = PyModule::new(m.py(), "apriltag")?;
 apriltag_mod.add_class::<apriltag::PyDecodeTagsConfig>()?;
 apriltag_mod.add_class::<apriltag::PyFitQuadConfig>()?;
 apriltag_mod.add_class::<apriltag::PyAprilTagDecoder>()?;
 apriltag_mod.add_class::<apriltag::PyApriltagDetection>()?;
 apriltag_mod.add_class::<apriltag::PyQuad>()?;
 let apriltag_family_mod = PyModule::new(apriltag_mod.py(), "family")?;
 apriltag_family_mod.add_class::<apriltag::family::PyTagFamily>()?;
 apriltag_family_mod.add_class::<apriltag::family::PyTagFamilyKind>()?;
 apriltag_family_mod.add_class::<apriltag::family::PyQuickDecode>()?;
 apriltag_family_mod.add_class::<apriltag::family::PySharpeningBuffer>()?;
 apriltag_mod.add_submodule(&apriltag_family_mod)?;
 m.add_submodule(&apriltag_mod)?;
 
 Ok(())
}
