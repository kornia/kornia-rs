use super::*;

#[test]
fn select_rejects_bad_index_and_attributes() {
    let cloud = PointCloud::new(vec![[0.0; 3]; 2], Some(vec![[0; 3]]), None);
    assert!(matches!(
        select_points(&cloud, &[0]),
        Err(FilterError::AttributeLengthMismatch {
            attribute: "colors",
            len: 1,
            expected: 2
        })
    ));
    let cloud = PointCloud::new(vec![[0.0; 3]; 2], None, None);
    assert!(matches!(
        select_points(&cloud, &[2]),
        Err(FilterError::IndexOutOfRange { index: 2, len: 2 })
    ));
}
