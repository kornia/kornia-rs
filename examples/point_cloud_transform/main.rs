use kornia_icp::{compute_centroids, fit_transformation};
use kornia_linalg::{Mat3, Vec3};

fn main() {
    // Create a source point cloud
    let points_src = vec![
        Vec3::new(1.0, 2.0, 3.0),
        Vec3::new(4.0, 5.0, 6.0),
        Vec3::new(7.0, 8.0, 9.0),
    ];

    // Create a transformation
    let rotation = Mat3::from_rotation_y(std::f32::consts::PI / 4.0);
    let translation = Vec3::new(2.0, -1.0, 3.0);
    
    // Apply transformation to create destination point cloud
    let points_dst: Vec<Vec3> = points_src.iter()
        .map(|p| rotation * *p + translation)
        .collect();

    // Compute centroids and transformation
    let (src_centroid, dst_centroid) = compute_centroids(&points_src, &points_dst);
    println!("Source centroid: {:?}", src_centroid);
    println!("Destination centroid: {:?}", dst_centroid);

    // Compute the transformation
    let (r, t) = fit_transformation(&points_src, &points_dst);
    println!("Computed rotation matrix:\n{:?}", r);
    println!("Expected rotation matrix:\n{:?}", rotation);
    println!("Computed translation vector: {:?}", t);
    println!("Expected translation vector: {:?}", translation);

    // Measure error
    let rot_error = (r - rotation).abs().max_element();
    let trans_error = (t - translation).length();
    println!("Rotation error: {}", rot_error);
    println!("Translation error: {}", trans_error);
} 