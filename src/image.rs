use sophus_rs::image::view::ImageSize as ImageSizeInner;

pub type ImageSize = ImageSizeInner;

#[cfg(test)]
mod tests {
    #[test]
    fn image_size() {
        use crate::image::ImageSize;
        let image_size = ImageSize {
            width: 10,
            height: 20,
        };
        assert_eq!(image_size.width, 10);
        assert_eq!(image_size.height, 20);
    }
}
