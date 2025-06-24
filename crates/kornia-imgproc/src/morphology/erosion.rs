use super::utils::{GrayImage, BorderMode};
use super::MorphologyError;

/// Applies erosion using a custom binary kernel.
/// Returns an error if the image or kernel is invalid.
pub fn erode_with_kernel(input: &GrayImage, kernel: &[Vec<bool>]) -> Result<GrayImage, MorphologyError> {
    if kernel.is_empty() || kernel[0].is_empty() {
        return Err(MorphologyError::EmptyKernel);
    }

    if kernel.iter().all(|row| row.iter().all(|&b| !b)) {
        return Err(MorphologyError::AllKernelElementsInactive);
    }

    validate_kernel_and_image(input, kernel)?;

    let (width, height) = input.dimensions();
    let kh = kernel.len();
    let kw = kernel[0].len();
    let k_center_y = kh / 2;
    let k_center_x = kw / 2;

    let mut output = GrayImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let mut min_val = 255u8;
            let mut found_active_pixel = false;

            for (ky, row) in kernel.iter().enumerate() {
                for (kx, &is_active) in row.iter().enumerate() {
                    if !is_active {
                        continue;
                    }

                    let ny = y as i32 + ky as i32 - k_center_y as i32;
                    let nx = x as i32 + kx as i32 - k_center_x as i32;

                    if ny >= 0 && ny < height as i32 && nx >= 0 && nx < width as i32 {
                        let val = input.get_pixel(nx as u32, ny as u32);
                        min_val = min_val.min(val);
                        found_active_pixel = true;
                    }
                }
            }

            output.put_pixel(x, y, if found_active_pixel { min_val } else { 0 });
        }
    }
    

    Ok(output)
}

/// Applies standard 3x3 erosion using a default full kernel.
/// Errors on empty image.
pub fn erode(input: &GrayImage) -> Result<GrayImage, MorphologyError> {
    if input.width == 0 || input.height == 0 {
        return Err(MorphologyError::EmptyImage);
    }

    let (width, height) = input.dimensions();
    let mut output = GrayImage::new(width, height);

    let neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1), ( 0, 0), ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1),
    ];

    for y in 0..height {
        for x in 0..width {
            let mut min_val = 255u8;

            for (dy, dx) in neighbors.iter() {
                let ny = y as i32 + dy;
                let nx = x as i32 + dx;

                if ny >= 0 && ny < height as i32 && nx >= 0 && nx < width as i32 {
                    let val = input.get_pixel(nx as u32, ny as u32);
                    min_val = min_val.min(val);
                }
            }

            output.put_pixel(x, y, min_val);
        }
    }

    Ok(output)
}

/// Applies erosion with custom kernel and configurable border handling.
pub fn erode_with_kernel_border(
    input: &GrayImage,
    kernel: &[Vec<bool>],
    border_mode: BorderMode,
) -> Result<GrayImage, MorphologyError> {
    validate_kernel_and_image(input, kernel)?;

    let (width, height) = input.dimensions();
    let kh = kernel.len();
    let kw = kernel[0].len();
    let k_center_y = kh / 2;
    let k_center_x = kw / 2;

    let mut output = GrayImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let mut min_val = 255u8;
            let mut considered = false;

            for (ky, row) in kernel.iter().enumerate() {
                for (kx, &is_active) in row.iter().enumerate() {
                    if !is_active {
                        continue;
                    }

                    let ny = y as i32 + ky as i32 - k_center_y as i32;
                    let nx = x as i32 + kx as i32 - k_center_x as i32;

                    let val_opt = if ny >= 0 && ny < height as i32 && nx >= 0 && nx < width as i32 {
                        Some(input.get_pixel(nx as u32, ny as u32))
                    } else {
                        match border_mode {
                            BorderMode::Ignore => None,
                            BorderMode::Clamp => {
                                let cy = ny.clamp(0, (height - 1) as i32) as u32;
                                let cx = nx.clamp(0, (width - 1) as i32) as u32;
                                Some(input.get_pixel(cx, cy))
                            }
                            BorderMode::Constant(v) => Some(v),
                        }
                    };

                    if let Some(val) = val_opt {
                        min_val = min_val.min(val);
                        considered = true;
                    }
                }
            }

            output.put_pixel(x, y, if considered { min_val } else { 0 });
        }
    }

    Ok(output)
}

/// Shared kernel/image validation logic.
fn validate_kernel_and_image(
    input: &GrayImage,
    kernel: &[Vec<bool>],
) -> Result<(), MorphologyError> {
    if input.width == 0 || input.height == 0 {
        return Err(MorphologyError::EmptyImage);
    }

    if kernel.is_empty() || kernel[0].is_empty() {
        return Err(MorphologyError::EmptyKernel);
    }

    let row_len = kernel[0].len();
    if !kernel.iter().all(|row| row.len() == row_len) {
        return Err(MorphologyError::NonRectangularKernel);
    }

    if row_len % 2 == 0 || kernel.len() % 2 == 0 {
        return Err(MorphologyError::EvenSizedKernel);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::utils::BorderMode;
    use super::*;

    fn raw_gray(data: &[u8], width: u32, height: u32) -> GrayImage {
        GrayImage {
            width,
            height,
            pixels: data.to_vec(),
        }
    }

    #[test]
    fn test_erode() {
        let input = raw_gray(&[255, 255, 255, 255, 0, 255, 255, 255, 255], 3, 3);
        let expected = raw_gray(&[0, 0, 0, 0, 0, 0, 0, 0, 0], 3, 3);
        assert_eq!(erode(&input), Ok(expected));
    }

    #[test]
    fn test_erode_with_kernel_identity() {
        let input = raw_gray(&[10, 20, 30, 40, 50, 60, 70, 80, 90], 3, 3);
        let kernel = vec![
            vec![false, false, false],
            vec![false, true, false],
            vec![false, false, false],
        ];
        let output = erode_with_kernel(&input, &kernel);
        assert_eq!(output, Ok(input));
    }

    #[test]
    fn test_erode_with_kernel_smaller() {
        let input = raw_gray(&[5, 3, 7, 6, 2, 8, 9, 4, 1], 3, 3);
        let kernel = vec![
            vec![false, true, false],
            vec![true, true, true],
            vec![false, true, false],
        ];
        let expected = raw_gray(&[3, 2, 3, 2, 2, 1, 4, 1, 1], 3, 3);
        let output = erode_with_kernel(&input, &kernel);
        assert_eq!(output, Ok(expected));
    }

    #[test]
    fn test_erode_with_kernel_border_ignore() {
        let input = raw_gray(&[5, 3, 7, 6, 2, 8, 9, 4, 1], 3, 3);
        let kernel = vec![
            vec![true, true, true],
            vec![true, true, true],
            vec![true, true, true],
        ];
        let expected = raw_gray(&[2, 2, 2, 2, 1, 1, 2, 1, 1], 3, 3);
        let output = erode_with_kernel_border(&input, &kernel, BorderMode::Ignore);
        assert_eq!(output, Ok(expected));
    }

    #[test]
    fn test_erode_with_kernel_border_clamp() {
        let input = raw_gray(&[5, 3, 7, 6, 2, 8, 9, 4, 1], 3, 3);
        let kernel = vec![
            vec![true, true, true],
            vec![true, true, true],
            vec![true, true, true],
        ];
        let expected = raw_gray(&[2, 2, 2, 2, 1, 1, 2, 1, 1], 3, 3);
        let output = erode_with_kernel_border(&input, &kernel, BorderMode::Clamp);
        assert_eq!(output, Ok(expected));
    }

    #[test]
    fn test_erode_with_kernel_border_constant() {
        let input = raw_gray(&[5, 3, 7, 6, 2, 8, 9, 4, 1], 3, 3);
        let kernel = vec![
            vec![true, true, true],
            vec![true, true, true],
            vec![true, true, true],
        ];
        let border_value = 10;
        let expected = raw_gray(&[2, 2, 2, 2, 1, 1, 2, 1, 1], 3, 3);
        let output = erode_with_kernel_border(&input, &kernel, BorderMode::Constant(border_value));
        assert_eq!(output, Ok(expected));
    }

    #[test]
    fn test_erode_with_kernel_all_false() {
        let input = raw_gray(&[5, 3, 7, 6, 2, 8, 9, 4, 1], 3, 3);
        let kernel = vec![
            vec![false, false, false],
            vec![false, false, false],
            vec![false, false, false],
        ];
        let output = erode_with_kernel(&input, &kernel);
        assert_eq!(output, Err(MorphologyError::AllKernelElementsInactive));
    }

    #[test]
    fn test_empty_kernel() {
        let input = GrayImage::new(3, 3);
        let result = erode_with_kernel(&input, &[]);
        assert_eq!(result, Err(MorphologyError::EmptyKernel));
    }
}
