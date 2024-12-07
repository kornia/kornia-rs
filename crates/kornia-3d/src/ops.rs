/// Utility function to compute the Euclidean distance between two points.
///
/// # Arguments
///
/// * `a` - A point in 3D space.
/// * `b` - Another point in 3D space.
///
/// # Returns
///
/// The Euclidean distance between the two points.
///
/// Example:
/// ```
/// use kornia_3d::ops::euclidean_distance;
///
/// let a = [1.0, 2.0, 3.0];
/// let b = [4.0, 5.0, 6.0];
/// let dst = euclidean_distance(&a, &b);
/// ```
pub fn euclidean_distance(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    assert_eq!(a.len(), b.len());
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    #[test]
    fn test_euclidean_distance() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        assert_relative_eq!(euclidean_distance(&a, &b), 5.196152, epsilon = 1e-6);
    }
}
