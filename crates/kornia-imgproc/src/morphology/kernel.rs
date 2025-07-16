pub enum MorphShape {
    Rect,
    Ellipse,
    Cross,
}

/// Creates kernels for erosion
pub fn kernel_shape(shape: MorphShape, ksize: (usize, usize)) -> Vec<bool> {
    let (rows, cols) = ksize;
    let mut kernel = vec![false; rows * cols];
    let cy = rows / 2;
    let cx = cols / 2;

    for r in 0..rows {
        for c in 0..cols {
            let idx = r * cols + c;
            kernel[idx] = match shape {
                MorphShape::Rect => true,
                MorphShape::Cross => r == cy || c == cx,
                MorphShape::Ellipse => {
                    let dy = (r as f64 - cy as f64) / (rows as f64 / 2.0);
                    let dx = (c as f64 - cx as f64) / (cols as f64 / 2.0);
                    dx * dx + dy * dy <= 1.0
                }
            };
        }
    }

    kernel
}

#[cfg(test)]
mod tests {
    use super::*;

    fn print_kernel(kernel: &[bool], rows: usize, cols: usize) {
        for r in 0..rows {
            for c in 0..cols {
                let idx = r * cols + c;
                print!("{}", if kernel[idx] { "1 " } else { "0 " });
            }
            println!();
        }
    }

    #[test]
    fn test_rect_kernel_3x3() {
        let rows = 3;
        let cols = 3;
        let kernel = kernel_shape(MorphShape::Rect, (rows, cols));

        assert_eq!(kernel.len(), rows * cols);
        assert!(kernel.iter().all(|&v| v));

        println!("\nRect 3x3:");
        print_kernel(&kernel, rows, cols);
    }

    #[test]
    fn test_cross_kernel_3x3() {
        let rows = 3;
        let cols = 3;
        let kernel = kernel_shape(MorphShape::Cross, (rows, cols));

        assert_eq!(kernel.len(), rows * cols);

        println!("\nCross 3x3:");
        print_kernel(&kernel, rows, cols);

        assert!(kernel[1 * cols + 1]);
        assert!(!kernel[0]);
        assert!(!kernel[2]);
        assert!(!kernel[6]);
        assert!(!kernel[8]);
    }

    #[test]
    fn test_ellipse_kernel_5x5() {
        let rows = 5;
        let cols = 5;
        let kernel = kernel_shape(MorphShape::Ellipse, (rows, cols));

        assert_eq!(kernel.len(), rows * cols);

        println!("\nEllipse 5x5:");
        print_kernel(&kernel, rows, cols);

        assert!(kernel[2 * cols + 2]);
    }
}
