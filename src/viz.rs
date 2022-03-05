use image;
use pyo3::prelude::*; //TODO: import what you use
                      // internal libs
use crate::tensor::cv;
use crate::viz::image::RgbImage;

#[pyclass(unsendable)]
pub struct VizManager {
    pub manager: vviz::manager::Manager,
}

impl Default for VizManager {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl VizManager {
    #[new]
    pub fn new() -> Self {
        VizManager {
            manager: vviz::manager::Manager::new_remote(),
        }
    }

    pub fn add_image(&mut self, window_name: String, image: cv::Tensor) {
        let (data, shape) = (image.data, image.shape);
        let (_w, _h, _ch) = (shape[0], shape[1], shape[2]);
        let buf: image::RgbImage =
            image::ImageBuffer::from_raw(_w as u32, _h as u32, data).unwrap();
        let img = image::DynamicImage::ImageRgb8(buf);
        self.manager.add_widget2(window_name, img.into_rgba8());
    }

    pub fn show(&mut self) {
        loop {
            self.manager.sync_with_gui();
        }
    }
}

#[pyfunction]
pub fn show_image_from_file(window_name: String, file_path: String) {
    vviz::app::spawn(
        vviz::app::VVizMode::Local,
        move |mut manager: vviz::manager::Manager| {
            let img: image::DynamicImage = image::open(file_path.clone()).unwrap();
            manager.add_widget2(window_name, img.into_rgba8());
            manager.sync_with_gui();
        },
    );
}

#[pyfunction]
pub fn show_image_from_tensor(window_name: String, image: cv::Tensor) {
    vviz::app::spawn(
        vviz::app::VVizMode::Local,
        move |mut manager: vviz::manager::Manager| {
            let (data, shape) = (image.data, image.shape);
            let (_h, _w, _ch) = (shape[0], shape[1], shape[2]);
            let buf: RgbImage = image::ImageBuffer::from_raw(_w as u32, _h as u32, data).unwrap();
            let img = image::DynamicImage::ImageRgb8(buf);
            manager.add_widget2(window_name, img.into_rgba8());
            manager.sync_with_gui();
        },
    );
}
