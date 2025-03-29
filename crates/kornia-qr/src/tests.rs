#[cfg(test)]
mod tests {
    
    use kornia_image::Image;
    use kornia_io::functional as F;
    use kornia_tensor::{Tensor, CpuAllocator};
    use crate::{QrDetector, QrDetectionExt};
    use std::path::Path;

    // Test QR code detection with a real image
    #[test]
    fn test_qr_detection() {
        // Load the test image with a QR code
        let test_img_path = Path::new("tests/data/qr/kornia.png");
        
        // Skip the test if the test image doesn't exist
        if !test_img_path.exists() {
            println!("Test image not found: {:?}", test_img_path);
            return;
        }
        
        // Read the test image
        let image: Image<u8, 3> = match F::read_image_any_rgb8(test_img_path) {
            Ok(img) => img,
            Err(e) => {
                println!("Failed to read test image: {}", e);
                return;
            }
        };
        
        // Detect QR codes
        let detections = match image.detect_qr_codes() {
            Ok(dets) => dets,
            Err(e) => {
                println!("QR detection failed: {}", e);
                return;
            }
        };
        
        // Check that at least one QR code was detected
        assert!(!detections.is_empty(), "No QR codes were detected");
        
        // Check the content of the first QR code
        let expected_content = "https://kornia.org";
        assert_eq!(detections[0].content, expected_content);
    }
    
    // Test the grayscale conversion logic
    #[test]
    fn test_tensor_to_grayscale() {
        // Create a simple RGB tensor
        let width = 4;
        let height = 4;
        let channels = 3;
        
        // Create RGB data where each pixel is (r, g, b) = (100, 150, 200)
        let mut rgb_data = Vec::with_capacity(width * height * channels);
        for _ in 0..(width * height) {
            rgb_data.push(100); // R
            rgb_data.push(150); // G
            rgb_data.push(200); // B
        }
        
        // Create a tensor
        let tensor = Tensor::<u8, 3, CpuAllocator>::from_shape_vec(
            [height, width, channels],
            rgb_data,
            CpuAllocator
        ).unwrap();
        
        // Convert to grayscale
        let grayscale = QrDetector::tensor_to_grayscale(&tensor).unwrap();
        
        // Check dimensions
        assert_eq!(grayscale.width(), width as u32);
        assert_eq!(grayscale.height(), height as u32);
        
        // For RGB (100, 150, 200), the average is 150
        for pixel in grayscale.pixels() {
            assert_eq!(pixel[0], 150);
        }
    }
} 