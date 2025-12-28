mod apriltag;
mod color;
mod enhance;
mod histogram;
mod icp;
mod image;
mod io;
mod pointcloud;
mod resize;
mod warp;

use pyo3::prelude::*;

pub fn get_version() -> String {
    let version = env!("CARGO_PKG_VERSION").to_string();
    version.replace("-alpha", "a").replace("-beta", "b")
}

#[pymodule(gil_used = false)]
pub fn kornia_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", get_version())?;

    // --- 1. IMAGE SUBMODULE ---
    let image_mod = PyModule::new(m.py(), "image")?;
    image_mod.add_class::<image::PyImageSize>()?;
    image_mod.add_class::<image::PyPixelFormat>()?;
    image_mod.add_class::<image::PyImageLayout>()?;
    m.add_submodule(&image_mod)?;

    // --- 2. IO SUBMODULE ---
    let io_mod = PyModule::new(m.py(), "io")?;
    // Functional
    io_mod.add_function(wrap_pyfunction!(io::functional::read_image, &io_mod)?)?;
    io_mod.add_function(wrap_pyfunction!(io::functional::read_image_any, &io_mod)?)?;
    // JPEG
    io_mod.add_function(wrap_pyfunction!(io::jpeg::read_image_jpeg, &io_mod)?)?;
    io_mod.add_function(wrap_pyfunction!(io::jpeg::write_image_jpeg, &io_mod)?)?;
    io_mod.add_function(wrap_pyfunction!(io::jpeg::decode_image_jpeg, &io_mod)?)?;
    io_mod.add_function(wrap_pyfunction!(io::jpeg::encode_image_jpeg, &io_mod)?)?;
    // PNG
    io_mod.add_function(wrap_pyfunction!(io::png::read_image_png_u8, &io_mod)?)?;
    io_mod.add_function(wrap_pyfunction!(io::png::read_image_png_u16, &io_mod)?)?;
    io_mod.add_function(wrap_pyfunction!(io::png::write_image_png_u8, &io_mod)?)?;
    io_mod.add_function(wrap_pyfunction!(io::png::write_image_png_u16, &io_mod)?)?;
    io_mod.add_function(wrap_pyfunction!(io::png::decode_image_png_u8, &io_mod)?)?;
    io_mod.add_function(wrap_pyfunction!(io::png::decode_image_png_u16, &io_mod)?)?;
    // TIFF
    io_mod.add_function(wrap_pyfunction!(io::tiff::read_image_tiff_u8, &io_mod)?)?;
    io_mod.add_function(wrap_pyfunction!(io::tiff::read_image_tiff_u16, &io_mod)?)?;
    io_mod.add_function(wrap_pyfunction!(io::tiff::read_image_tiff_f32, &io_mod)?)?;
    io_mod.add_function(wrap_pyfunction!(io::tiff::write_image_tiff_u8, &io_mod)?)?;
    io_mod.add_function(wrap_pyfunction!(io::tiff::write_image_tiff_u16, &io_mod)?)?;
    io_mod.add_function(wrap_pyfunction!(io::tiff::write_image_tiff_f32, &io_mod)?)?;
    // JpegTurbo
    io_mod.add_class::<io::jpegturbo::PyImageDecoder>()?;
    io_mod.add_class::<io::jpegturbo::PyImageEncoder>()?;
    io_mod.add_function(wrap_pyfunction!(io::jpegturbo::read_image_jpegturbo, &io_mod)?)?;
    io_mod.add_function(wrap_pyfunction!(io::jpegturbo::write_image_jpegturbo, &io_mod)?)?;
    io_mod.add_function(wrap_pyfunction!(io::jpegturbo::decode_image_jpegturbo, &io_mod)?)?;
    m.add_submodule(&io_mod)?;

    // --- 3. IMGPROC SUBMODULE (Color, Enhance, Histogram, Resize) ---
    let imgproc_mod = PyModule::new(m.py(), "imgproc")?;
    imgproc_mod.add_function(wrap_pyfunction!(color::rgb_from_gray, &imgproc_mod)?)?;
    imgproc_mod.add_function(wrap_pyfunction!(color::rgb_from_rgba, &imgproc_mod)?)?;
    imgproc_mod.add_function(wrap_pyfunction!(color::rgb_from_bgra, &imgproc_mod)?)?;
    imgproc_mod.add_function(wrap_pyfunction!(color::bgr_from_rgb, &imgproc_mod)?)?;
    imgproc_mod.add_function(wrap_pyfunction!(color::gray_from_rgb, &imgproc_mod)?)?;
    imgproc_mod.add_function(wrap_pyfunction!(enhance::add_weighted, &imgproc_mod)?)?;
    imgproc_mod.add_function(wrap_pyfunction!(histogram::compute_histogram, &imgproc_mod)?)?;
    imgproc_mod.add_function(wrap_pyfunction!(resize::resize, &imgproc_mod)?)?;
    m.add_submodule(&imgproc_mod)?;

    // --- 4. GEOMETRY SUBMODULE (Warp, ICP) ---
    let geometry_mod = PyModule::new(m.py(), "geometry")?;
    geometry_mod.add_function(wrap_pyfunction!(warp::warp_affine, &geometry_mod)?)?;
    geometry_mod.add_function(wrap_pyfunction!(warp::warp_perspective, &geometry_mod)?)?;
    geometry_mod.add_function(wrap_pyfunction!(icp::icp_vanilla, &geometry_mod)?)?;
    geometry_mod.add_class::<icp::PyICPConvergenceCriteria>()?;
    geometry_mod.add_class::<icp::PyICPResult>()?;
    m.add_submodule(&geometry_mod)?;

    // --- 5. APRILTAG SUBMODULE ---
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