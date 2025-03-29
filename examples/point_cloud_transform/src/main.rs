use glam::{Vec3, Mat3};
use kornia_icp::fit_transformation;

fn main() {
    println!("Point Cloud Transformation Example");
    println!("=================================\n");
    
    // Create a point cloud with a known pattern
    let points_src = vec![
        Vec3::new(0.0, 0.0, 0.0), // Origin
        Vec3::new(1.0, 0.0, 0.0), // Unit X
        Vec3::new(0.0, 1.0, 0.0), // Unit Y
        Vec3::new(0.0, 0.0, 1.0), // Unit Z
    ];

    // Define a reference transformation: 45-deg rotation around Y-axis + translation
    let rotation = Mat3::from_rotation_y(std::f32::consts::PI / 4.0);
    let translation = Vec3::new(2.0, -1.0, 3.0);

    println!("Reference rotation matrix:");
    println!("{:?}", rotation);
    println!("Reference translation: {:?}", translation);
    println!("Reference matrix determinant: {}", rotation.determinant());

    // Apply transformation to create destination point cloud
    let points_dst: Vec<Vec3> = points_src
        .iter()
        .map(|&p| rotation * p + translation)
        .collect();

    // Print source and destination points
    println!("\nSource and destination points:");
    for i in 0..points_src.len() {
        println!("Point {}: src={:?} -> dst={:?}", i, points_src[i], points_dst[i]);
    }

    // Compute transformation using our SVD algorithm
    let (est_rotation, est_translation) = fit_transformation(&points_src, &points_dst);

    println!("\nEstimated transformation:");
    println!("Rotation matrix:\n{:?}", est_rotation);
    println!("Translation vector: {:?}", est_translation);

    // Compute error metrics
    let rot_error_mat = est_rotation * rotation.transpose();
    let rot_error = (rot_error_mat - Mat3::IDENTITY)
        .to_cols_array()
        .iter()
        .fold(0.0f32, |max_val, &val| max_val.max(val.abs()));
    let trans_error = (est_translation - translation).length();

    println!("\nError metrics:");
    println!("Rotation error: {}", rot_error);
    println!("Translation error: {}", trans_error);

    // Apply estimated transformation to source points for verification
    println!("\nVerification of transformation:");
    for i in 0..points_src.len() {
        let transformed = est_rotation * points_src[i] + est_translation;
        let error = (transformed - points_dst[i]).length();
        println!("Point {}: transformed={:?}, dst={:?}, error={:e}", 
                 i, transformed, points_dst[i], error);
    }
} 