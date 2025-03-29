/// A struct representing an RGB-D image.
#[derive(Debug, Clone)]
pub struct RGBDImage {
    /// The width of the image
    pub width: usize,
    /// The height of the image
    pub height: usize,
    /// The RGB image as a 2D array of RGB values
    pub rgb: Vec<[u8; 3]>,
    /// The depth image as a 2D array of depth values
    pub depth: Vec<f64>,
}

impl RGBDImage {
    /// Creates a new RGBDImage with the given RGB and depth arrays.
    pub fn new(rgb: Vec<[u8; 3]>, depth: Vec<f64>, width: usize, height: usize) -> Self {
        assert_eq!(rgb.len(), depth.len(), "RGB and depth images must have the same dimensions");
        Self { rgb, depth, width, height }
    }

    /// Returns the dimensions of the image (height, width)
    pub fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    
    /// Get the depth value at a specific pixel.
    #[inline]
    pub fn get_depth(&self, x: usize, y: usize) -> f64 {
        self.depth[y * self.width + x]
    }

    /// Get the color value at a specific pixel.
    #[inline]
    pub fn get_color(&self, x: usize, y: usize) -> [u8; 3] {
        self.rgb[y * self.width + x]
    }
} 