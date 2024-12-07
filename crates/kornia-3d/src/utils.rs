/// Utility function to convert a 3D array to a faer column vector.
/// # Arguments
///
/// * `array` - A 3D array.
///
/// # Returns
///
/// A faer column vector.
pub fn array3_to_faer_col(array: &[f64; 3]) -> faer::ColRef<'_, f64> {
    let array_slice = unsafe { std::slice::from_raw_parts(array.as_ptr(), array.len()) };
    faer::col::from_slice(array_slice)
}

/// Utility function to convert a 3x3 array to a faer matrix 3x3.
///
/// # Arguments
///
/// * `array` - A 3x3 array.
///
/// # Returns
///
/// A faer matrix 3x3.
pub fn array33_to_faer_mat33(array: &[[f64; 3]; 3]) -> faer::MatRef<'_, f64> {
    let array_slice =
        unsafe { std::slice::from_raw_parts(array.as_ptr() as *const f64, array.len() * 3) };
    faer::mat::from_row_major_slice(array_slice, 3, 3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array3_to_col() {
        let array = [1.0, 2.0, 3.0];
        let col = array3_to_faer_col(&array);
        assert_eq!(col.read(0), 1.0);
        assert_eq!(col.read(1), 2.0);
        assert_eq!(col.read(2), 3.0);
    }

    #[test]
    fn test_array33_to_mat33() {
        let array = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let mat = array33_to_faer_mat33(&array);
        assert_eq!(mat.read(0, 0), 1.0);
        assert_eq!(mat.read(0, 1), 2.0);
        assert_eq!(mat.read(0, 2), 3.0);
        assert_eq!(mat.read(1, 0), 4.0);
        assert_eq!(mat.read(1, 1), 5.0);
        assert_eq!(mat.read(1, 2), 6.0);
        assert_eq!(mat.read(2, 0), 7.0);
        assert_eq!(mat.read(2, 1), 8.0);
        assert_eq!(mat.read(2, 2), 9.0);
    }
}
