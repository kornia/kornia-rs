/// Generate a square binary kernel of given odd size (e.g., 3 -> 3x3 all true)
pub fn generate_box_kernel(size: usize) -> Vec<Vec<bool>> {
    assert!(size % 2 == 1, "Kernel size must be odd");
    vec![vec![true; size]; size]
}
