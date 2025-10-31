use argh::FromArgs;
use kornia_imgproc::calibration::{
    distortion::{distort_point_polynomial, PolynomialDistortion},
    CameraIntrinsic,
};
use kornia_pnp as kpnp;
use rand::Rng;

#[derive(FromArgs, Debug)]
/// PnP demo application
struct Args {
    /// use RANSAC (robust EPnP). If not set, use direct EPnP
    #[argh(switch)]
    use_ransac: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    // Parse CLI arguments
    let args: Args = argh::from_env();

    let rec = rerun::RecordingStreamBuilder::new("PnP Demo").spawn()?;

    // Camera intrinsics (pinhole, fx=fy=800, cx=640, cy=480)
    let k = [[800.0, 0.0, 640.0], [0.0, 800.0, 480.0], [0.0, 0.0, 1.0]];

    // Intrinsics for distortion API (f64)
    let cam_intr = CameraIntrinsic {
        fx: 800.0,
        fy: 800.0,
        cx: 640.0,
        cy: 480.0,
    };

    // Example Brown-Conrady distortion coefficients
    let distortion = PolynomialDistortion {
        k1: 1.0e-4,
        k2: -5.0e-7,
        k3: 0.0,
        k4: 0.0,
        k5: 0.0,
        k6: 0.0,
        p1: 1.0e-5,
        p2: -1.0e-5,
    };

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
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0), // bottom face
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4), // top face
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7), // vertical edges
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
        &rerun::Points3D::new(p3d).with_colors(vec![color_cube; 8]),
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

    // Project points to image plane + apply distortion + add small noise
    let mut rng = rand::rng();
    let sigma_px = 0.5; // pixel noise
    let image_pts: Vec<[f32; 2]> = world_pts
        .iter()
        .map(|p| {
            let x_c = gt_r[0][0] * p[0] + gt_r[0][1] * p[1] + gt_r[0][2] * p[2] + gt_t[0];
            let y_c = gt_r[1][0] * p[0] + gt_r[1][1] * p[1] + gt_r[1][2] * p[2] + gt_t[1];
            let z_c = gt_r[2][0] * p[0] + gt_r[2][1] * p[1] + gt_r[2][2] * p[2] + gt_t[2];
            let u = k[0][0] * x_c / z_c + k[0][2];
            let v = k[1][1] * y_c / z_c + k[1][2];
            let (ud, vd) = distort_point_polynomial(u as f64, v as f64, &cam_intr, &distortion);
            let ud = ud as f32 + rng.random::<f32>() * sigma_px;
            let vd = vd as f32 + rng.random::<f32>() * sigma_px;
            [ud, vd]
        })
        .collect();

    println!("Dataset: {} points", world_pts.len());
    println!("Using RANSAC: {}", args.use_ransac);

    // Run pose estimation
    let result = if args.use_ransac {
        // RANSAC mode: robust estimation
        let ransac_params = kpnp::RansacParams {
            max_iterations: 100,
            reproj_threshold_px: 2.0,
            confidence: 0.99,
            random_seed: Some(42),
            refine: true,
        };

        let ransac_result = kpnp::solve_pnp_ransac(
            &world_pts,
            &image_pts,
            &k,
            Some(&distortion),
            kpnp::PnPMethod::EPnPDefault,
            &ransac_params,
        )?;
        println!("RANSAC EPnP:");
        println!(
            "  Inliers found: {}/{}",
            ransac_result.inliers.len(),
            world_pts.len()
        );
        if let Some(rmse) = ransac_result.pose.reproj_rmse {
            println!("  RMSE: {rmse:.3} px");
        }
        ransac_result.pose
    } else {
        // Direct mode: standard EPnP
        let direct_result = kpnp::solve_pnp(
            &world_pts,
            &image_pts,
            &k,
            Some(&distortion),
            kpnp::PnPMethod::EPnPDefault,
        )?;
        println!("Direct EPnP:");
        if let Some(rmse) = direct_result.reproj_rmse {
            println!("  RMSE: {rmse:.3} px");
        }
        direct_result
    };

    // Log observed 2D points (all green since we're using clean data)
    let img_obs = image_pts
        .iter()
        .map(|uv| (uv[0], uv[1]))
        .collect::<Vec<_>>();
    rec.log_static(
        "image/observed",
        &rerun::Points2D::new(img_obs).with_colors([[0, 255, 0]]), // Green for all points
    )?;

    // Reproject original world points with estimated pose (not the outliers)
    let r_est = result.rotation;
    let t_est = result.translation;
    let mut img_reproj = Vec::with_capacity(world_pts.len());
    for p in &world_pts {
        let x_c = r_est[0][0] * p[0] + r_est[0][1] * p[1] + r_est[0][2] * p[2] + t_est[0];
        let y_c = r_est[1][0] * p[0] + r_est[1][1] * p[1] + r_est[1][2] * p[2] + t_est[1];
        let z_c = r_est[2][0] * p[0] + r_est[2][1] * p[1] + r_est[2][2] * p[2] + t_est[2];
        let u = k[0][0] * x_c / z_c + k[0][2];
        let v = k[1][1] * y_c / z_c + k[1][2];
        let (ud, vd) = distort_point_polynomial(u as f64, v as f64, &cam_intr, &distortion);
        img_reproj.push((ud as f32, vd as f32));
    }
    rec.log_static(
        "image/reprojected",
        &rerun::Points2D::new(img_reproj).with_colors([[0, 0, 255]]), // blue
    )?;

    // Compute camera center in world coordinates: C = -R^T * t
    let r = result.rotation;
    let t = result.translation;
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
        .with_colors(vec![rerun::Color::from_rgb(255, 165, 0)]), // orange
    )?;

    // Compute pose accuracy
    let trans_error = ((result.translation[0] - gt_t[0]).powi(2)
        + (result.translation[1] - gt_t[1]).powi(2)
        + (result.translation[2] - gt_t[2]).powi(2))
    .sqrt();

    println!("\n=== Results ===");
    println!("Translation error: {trans_error:.3} units");
    if let Some(rmse) = result.reproj_rmse {
        println!("Reprojection RMSE: {rmse:.3} px");
    }

    println!("\n=== Configuration ===");
    println!("To use RANSAC: run with --use-ransac");
    println!("To use direct EPnP: run without --use-ransac");

    println!("\n=== Visualization ===");
    println!("- Green points: Observed 2D points");
    println!("- Blue points: Reprojected 3D points using estimated pose");
    println!("- Yellow points: Original 3D cube structure");
    println!("- Orange point: Estimated camera center");
    Ok(())
}
