#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};

    #[test]
    fn test_gaussian_pyramid() {
        let img = GrayImage::new(100, 100);
        let pyramid = build_gaussian_pyramid(&img, 3, 1.5);
        assert_eq!(pyramid.len(), 3);
    }

    #[test]
    fn test_lucas_kanade() {
        let img1 = GrayImage::new(100, 100);
        let img2 = GrayImage::new(100, 100);
        let flow = lucas_kanade_optical_flow(&img1, &img2, 5);
        assert!(flow.len() > 0);
    }
}
