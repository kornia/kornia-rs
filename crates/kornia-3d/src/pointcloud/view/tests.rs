use std::borrow::Cow;

use super::*;
use crate::pointcloud::filter::FilterError;

fn cloud() -> PointCloud {
    PointCloud::new(
        vec![[0.0; 3], [1.0; 3], [2.0; 3], [3.0; 3]],
        Some(vec![[0; 3], [10; 3], [20; 3], [30; 3]]),
        Some(vec![
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5; 3],
        ]),
    )
}

#[test]
fn select_borrows_or_owns_and_matches_the_copy() -> Result<(), Box<dyn std::error::Error>> {
    let cloud = cloud();
    let indices = vec![3u32, 1, 1, 0];
    let borrowed = cloud.select(&indices)?;
    assert!(matches!(borrowed.indices, Cow::Borrowed(_)));
    let owned = cloud.select(indices.clone())?;
    assert!(matches!(owned.indices, Cow::Owned(_)));
    for view in [&borrowed, &owned] {
        assert_eq!(view.len(), 4);
        assert!(!view.is_empty());
        assert!(std::ptr::eq(view.cloud(), &cloud));
        assert_eq!(view.indices(), &indices[..]);
        assert_eq!(view.point(1), Some(&[1.0; 3]));
        assert_eq!(view.color(0), Some(&[30; 3]));
        assert_eq!(view.normal(3), Some(&[0.0, 0.0, 1.0]));
        assert_eq!(view.point(4), None);
        assert_eq!(view.color(4), None);
        assert_eq!(view.normal(4), None);
        assert_eq!(view.points().len(), 4);
        let copy = view.to_point_cloud();
        assert_eq!(copy.points(), &vec![[3.0; 3], [1.0; 3], [1.0; 3], [0.0; 3]]);
        assert_eq!(
            copy.colors(),
            Some(&vec![[30; 3], [10; 3], [10; 3], [0; 3]])
        );
        let normals: Vec<[f64; 3]> = view.normals().ok_or("normals")?.copied().collect();
        assert_eq!(copy.normals(), Some(&normals));
    }
    assert_eq!(owned.into_indices(), indices);
    Ok(())
}

#[test]
fn select_without_attributes_and_empty() -> Result<(), Box<dyn std::error::Error>> {
    let cloud = PointCloud::new(vec![[0.0; 3], [1.0; 3]], None, None);
    let view = cloud.select(&[1])?;
    assert!(view.colors().is_none());
    assert!(view.normals().is_none());
    assert_eq!(view.color(0), None);
    let copy = view.to_point_cloud();
    assert!(copy.colors().is_none() && copy.normals().is_none());
    let empty = cloud.select(Vec::new())?;
    assert!(empty.is_empty());
    assert!(empty.to_point_cloud().is_empty());
    Ok(())
}

#[test]
fn select_rejects_bad_index_and_attributes() {
    let bad = PointCloud::new(vec![[0.0; 3]; 2], None, Some(vec![[0.0; 3]]));
    assert!(matches!(
        bad.select(&[0]),
        Err(FilterError::AttributeLengthMismatch {
            attribute: "normals",
            len: 1,
            expected: 2
        })
    ));
    let cloud = cloud();
    assert!(matches!(
        cloud.select(vec![0, 7, 9]),
        Err(FilterError::IndexOutOfRange { index: 7, len: 4 })
    ));
}
