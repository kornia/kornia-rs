use kornia_pnp as kpnp;
use rand::Rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let rec = rerun::RecordingStreamBuilder::new("PnP with Distortion Demo").spawn()?;

    // Camera intrinsics (pinhole, fx=fy=800, cx=640, cy=480)
    let intrinsics = kpnp::CameraIntrinsics::new(800.0, 800.0, 640.0, 480.0);
    
    // Add some radial distortion (k1=0.1, k2=0.01)
    let distortion = kpnp::PolynomialDistortion::radial(0.1, 0.01);
    
    // Create camera model with distortion
    let camera = kpnp::CameraModel::with_distortion(intrinsics, distortion);

    // Build a dense cube (corners + 200 random interior points)
    let cube_size = 1.0;
    let half = cube_size / 2.0;
    let mut world_pts = vec![
        [-half, -half, 0.0],
        [half, -half, 0.0],
        [half, half, 0.0],
        [-half, half, 0.0],
        [-half, -half, cube_size],
        [half, -half, cube_size],
        [half, half, cube_size],
        [-half, half, cube_size],
    ];

    // Add evenly spaced samples along the 12 cube edges for better structure.
    const EDGES: &[(usize, usize)] = &[
        (0, 1), (1, 2), (2, 3), (3, 0), // bottom face
        (4, 5), (5, 6), (6, 7), (7, 4), // top face
        (0, 4), (1, 5), (2, 6), (3, 7), // vertical edges
    ];

    let num_div = 9; // points per edge excluding endpoints
    for &(a, b) in EDGES {
        let p0 = world_pts[a];
        let p1 = world_pts[b];
        for i in 1..num_div {
            let t = i as f32 / num_div as f32;
            let x = p0[0] * (1.0 - t) + p1[0] * t;
            let y = p0[1] * (1.0 - t) + p1[1] * t;
            let z = p0[2] * (1.0 - t) + p1[2] * t;
            world_pts.push([x, y, z]);
        }
    }

    // Log the cube points (yellow)
    let p3d = world_pts
        .iter()
        .map(|p| rerun::Position3D::new(p[0], p[1], p[2]))
        .collect::<Vec<_>>();
    let color_cube = rerun::Color::from_rgb(255, 215, 0); // gold
    rec.log(
        "cube",
        &rerun::Points3D::new(p3d).with_colors(vec![color_cube; world_pts.len()]),
    )?;

    // Ground-truth pose: small rotation + translation along +Z
    let yaw = 30.0_f32.to_radians(); // around Z
    let pitch = -15.0_f32.to_radians(); // around Y
    let roll = 10.0_f32.to_radians(); // around X

    let (cy, sy) = (yaw.cos(), yaw.sin());
    let (cp, sp) = (pitch.cos(), pitch.sin());
    let (cr, sr) = (roll.cos(), roll.sin());

    // R = Rz * Ry * Rx
    let gt_r = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ];
    let gt_t = [0.7, -0.4, 5.0];

    // Project points to image plane (undistorted) + add small noise
    let mut rng = rand::rng();
    let sigma_px = 0.5; // pixel noise
    let undistorted_image_pts: Vec<[f32; 2]> = world_pts
        .iter()
        .map(|p| {
            let x_c = gt_r[0][0] * p[0] + gt_r[0][1] * p[1] + gt_r[0][2] * p[2] + gt_t[0];
            let y_c = gt_r[1][0] * p[0] + gt_r[1][1] * p[1] + gt_r[1][2] * p[2] + gt_t[1];
            let z_c = gt_r[2][0] * p[0] + gt_r[2][1] * p[1] + gt_r[2][2] * p[2] + gt_t[2];
            let u = camera.intrinsics.fx * x_c / z_c + camera.intrinsics.cx + rng.random::<f32>() * sigma_px;
            let v = camera.intrinsics.fy * y_c / z_c + camera.intrinsics.cy + rng.random::<f32>() * sigma_px;
            [u, v]
        })
        .collect();

    // Apply distortion to get the "observed" distorted points
    let distorted_image_pts: Vec<[f32; 2]> = undistorted_image_pts
        .iter()
        .map(|&[u, v]| {
            let (u_dist, v_dist) = camera.distort_point(u, v).unwrap();
            [u_dist, v_dist]
        })
        .collect();

    // Log observed distorted 2D points (green)
    let img_obs = distorted_image_pts
        .iter()
        .map(|uv| (uv[0], uv[1]))
        .collect::<Vec<_>>();
    rec.log_static(
        "image/distorted_observed",
        &rerun::Points2D::new(img_obs).with_colors([[0, 255, 0]]), // green
    )?;

    // Log undistorted 2D points (blue) for comparison
    let img_undist = undistorted_image_pts
        .iter()
        .map(|uv| (uv[0], uv[1]))
        .collect::<Vec<_>>();
    rec.log_static(
        "image/undistorted_ground_truth",
        &rerun::Points2D::new(img_undist).with_colors([[0, 0, 255]]), // blue
    )?;

    println!("=== PnP with Distortion Demo ===");
    println!("Camera intrinsics: fx={}, fy={}, cx={}, cy={}", 
             camera.intrinsics.fx, camera.intrinsics.fy, camera.intrinsics.cx, camera.intrinsics.cy);
    println!("Distortion: k1={}, k2={}", 
             camera.distortion.as_ref().unwrap().k1, 
             camera.distortion.as_ref().unwrap().k2);
    println!("Number of points: {}", world_pts.len());

    // Test 1: Solve PnP with distorted points using the new camera model interface
    println!("\n--- Test 1: PnP with camera model (handles distortion automatically) ---");
    let result_with_camera = kpnp::solve_pnp_with_camera(
        &world_pts, 
        &distorted_image_pts, 
        &camera, 
        kpnp::PnPMethod::EPnPDefault
    )?;

    println!("Ground truth translation: {:?}", gt_t);
    println!("Estimated translation  : {:?}", result_with_camera.translation);
    if let Some(rmse) = result_with_camera.reproj_rmse {
        println!("Reprojection RMSE: {:.3} px", rmse);
    }

    // Test 2: Solve PnP with undistorted points (traditional approach)
    println!("\n--- Test 2: PnP with undistorted points (traditional approach) ---");
    let result_undistorted = kpnp::solve_pnp(
        &world_pts, 
        &undistorted_image_pts, 
        &camera.intrinsics_matrix(), 
        kpnp::PnPMethod::EPnPDefault
    )?;

    println!("Ground truth translation: {:?}", gt_t);
    println!("Estimated translation  : {:?}", result_undistorted.translation);
    if let Some(rmse) = result_undistorted.reproj_rmse {
        println!("Reprojection RMSE: {:.3} px", rmse);
    }

    // Test 3: Solve PnP with distorted points using old interface (should fail or give poor results)
    println!("\n--- Test 3: PnP with distorted points using old interface (incorrect) ---");
    let result_incorrect = kpnp::solve_pnp(
        &world_pts, 
        &distorted_image_pts, 
        &camera.intrinsics_matrix(), 
        kpnp::PnPMethod::EPnPDefault
    )?;

    println!("Ground truth translation: {:?}", gt_t);
    println!("Estimated translation  : {:?}", result_incorrect.translation);
    if let Some(rmse) = result_incorrect.reproj_rmse {
        println!("Reprojection RMSE: {:.3} px", rmse);
    }

    // Reproject world points with estimated pose from Test 1
    let r_est = result_with_camera.rotation;
    let t_est = result_with_camera.translation;
    let mut img_reproj = Vec::with_capacity(world_pts.len());
    for p in &world_pts {
        let x_c = r_est[0][0] * p[0] + r_est[0][1] * p[1] + r_est[0][2] * p[2] + t_est[0];
        let y_c = r_est[1][0] * p[0] + r_est[1][1] * p[1] + r_est[1][2] * p[2] + t_est[1];
        let z_c = r_est[2][0] * p[0] + r_est[2][1] * p[1] + r_est[2][2] * p[2] + t_est[2];
        let u = camera.intrinsics.fx * x_c / z_c + camera.intrinsics.cx;
        let v = camera.intrinsics.fy * y_c / z_c + camera.intrinsics.cy;
        img_reproj.push((u, v));
    }
    rec.log_static(
        "image/reprojected_undistorted",
        &rerun::Points2D::new(img_reproj).with_colors([[255, 0, 0]]), // red
    )?;

    // Compute camera center in world coordinates: C = -R^T * t
    let r = result_with_camera.rotation;
    let t = result_with_camera.translation;
    let camera_center = [
        -(r[0][0] * t[0] + r[1][0] * t[1] + r[2][0] * t[2]),
        -(r[0][1] * t[0] + r[1][1] * t[1] + r[2][1] * t[2]),
        -(r[0][2] * t[0] + r[1][2] * t[1] + r[2][2] * t[2]),
    ];
    rec.log(
        "camera_center",
        &rerun::Points3D::new(vec![rerun::Position3D::new(
            camera_center[0],
            camera_center[1],
            camera_center[2],
        )])
        .with_colors(vec![rerun::Color::from_rgb(255, 0, 0)]),
    )?;

    println!("\n=== Summary ===");
    println!("Test 1 (with distortion handling): RMSE = {:.3} px", 
             result_with_camera.reproj_rmse.unwrap_or(0.0));
    println!("Test 2 (undistorted points): RMSE = {:.3} px", 
             result_undistorted.reproj_rmse.unwrap_or(0.0));
    println!("Test 3 (incorrect, distorted points): RMSE = {:.3} px", 
             result_incorrect.reproj_rmse.unwrap_or(0.0));

    Ok(())
}
