use argh::FromArgs;
use kornia_3d::pnp as kpnp;
use kornia_algebra::{Mat3AF32, Vec2F32, Vec3AF32};
use kornia_imgproc::calibration::{
    distortion::{distort_point_polynomial, PolynomialDistortion},
    CameraIntrinsic,
};
use rand::Rng;

#[derive(FromArgs, Debug)]
/// PnP demo application
struct Args {
    /// use RANSAC (robust EPnP). If not set, use direct EPnP
    #[argh(switch)]
    use_ransac: bool,
    /// enable LM refinement after initial EPnP solution
    #[argh(switch)]
    refine: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    // Parse CLI arguments
    let args: Args = argh::from_env();

    let rec = rerun::RecordingStreamBuilder::new("PnP Demo").spawn()?;

    // Camera intrinsics (pinhole, fx=fy=800, cx=640, cy=480)
    let k = Mat3AF32::from_cols(
        Vec3AF32::new(800.0, 0.0, 0.0),
        Vec3AF32::new(0.0, 800.0, 0.0),
        Vec3AF32::new(640.0, 480.0, 1.0),
    );

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
        Vec3AF32::new(-half, -half, 0.0),
        Vec3AF32::new(half, -half, 0.0),
        Vec3AF32::new(half, half, 0.0),
        Vec3AF32::new(-half, half, 0.0),
        Vec3AF32::new(-half, -half, cube_size),
        Vec3AF32::new(half, -half, cube_size),
        Vec3AF32::new(half, half, cube_size),
        Vec3AF32::new(-half, half, cube_size),
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
            let x = p0.x * (1.0 - t) + p1.x * t;
            let y = p0.y * (1.0 - t) + p1.y * t;
            let z = p0.z * (1.0 - t) + p1.z * t;
            world_pts.push(Vec3AF32::new(x, y, z));
        }
    }

    // Log the cube points (yellow)
    let p3d = world_pts
        .iter()
        .map(|p| rerun::Position3D::new(p.x, p.y, p.z))
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
    let r00 = cy * cp;
    let r01 = cy * sp * sr - sy * cr;
    let r02 = cy * sp * cr + sy * sr;
    let r10 = sy * cp;
    let r11 = sy * sp * sr + cy * cr;
    let r12 = sy * sp * cr - cy * sr;
    let r20 = -sp;
    let r21 = cp * sr;
    let r22 = cp * cr;
    let gt_r = Mat3AF32::from_cols(
        Vec3AF32::new(r00, r10, r20),
        Vec3AF32::new(r01, r11, r21),
        Vec3AF32::new(r02, r12, r22),
    );
    let gt_t = Vec3AF32::new(0.7, -0.4, 5.0);

    // Project points to image plane + apply distortion + add small noise
    let mut rng = rand::rng();
    let sigma_px = 0.5; // pixel noise
    let fx = k.x_axis().x;
    let fy = k.y_axis().y;
    let cx = k.z_axis().x;
    let cy = k.z_axis().y;
    let image_pts: Vec<Vec2F32> = world_pts
        .iter()
        .map(|p| {
            let pc = gt_r * *p + gt_t;
            let u = fx * pc.x / pc.z + cx;
            let v = fy * pc.y / pc.z + cy;
            let (ud, vd) = distort_point_polynomial(u as f64, v as f64, &cam_intr, &distortion);
            let ud = ud as f32 + rng.random::<f32>() * sigma_px;
            let vd = vd as f32 + rng.random::<f32>() * sigma_px;
            Vec2F32::new(ud, vd)
        })
        .collect();

    println!("Dataset: {} points", world_pts.len());
    println!("Using RANSAC: {}", args.use_ransac);
    println!("Using LM refinement: {}", args.refine);

    // Create EPnP parameters with optional refinement
    let epnp_params = if args.refine {
        kpnp::EPnPParams {
            tol: kpnp::NumericTol::default(),
            refine_lm: Some(kpnp::LMRefineParams::default()),
        }
    } else {
        kpnp::EPnPParams::default()
    };

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
            kpnp::PnPMethod::EPnP(epnp_params),
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
        if let Some(iterations) = ransac_result.pose.num_iterations {
            println!("  LM iterations: {}", iterations);
        }
        if let Some(converged) = ransac_result.pose.converged {
            println!("  LM converged: {}", converged);
        }
        ransac_result.pose
    } else {
        // Direct mode: standard EPnP
        let direct_result = kpnp::solve_pnp(
            &world_pts,
            &image_pts,
            &k,
            Some(&distortion),
            kpnp::PnPMethod::EPnP(epnp_params),
        )?;
        println!("Direct EPnP:");
        if let Some(rmse) = direct_result.reproj_rmse {
            println!("  RMSE: {rmse:.3} px");
        }
        if let Some(iterations) = direct_result.num_iterations {
            println!("  LM iterations: {}", iterations);
        }
        if let Some(converged) = direct_result.converged {
            println!("  LM converged: {}", converged);
        }
        direct_result
    };

    // Log observed 2D points (all green since we're using clean data)
    let img_obs = image_pts.iter().map(|uv| (uv.x, uv.y)).collect::<Vec<_>>();
    rec.log_static(
        "image/observed",
        &rerun::Points2D::new(img_obs).with_colors([[0, 255, 0]]), // Green for all points
    )?;

    // Reproject original world points with estimated pose (not the outliers)
    let r_est = result.rotation;
    let t_est = result.translation;
    let mut img_reproj = Vec::with_capacity(world_pts.len());
    for p in &world_pts {
        let pc = r_est * *p + t_est;
        let u = fx * pc.x / pc.z + cx;
        let v = fy * pc.y / pc.z + cy;
        let (ud, vd) = distort_point_polynomial(u as f64, v as f64, &cam_intr, &distortion);
        img_reproj.push((ud as f32, vd as f32));
    }
    rec.log_static(
        "image/reprojected",
        &rerun::Points2D::new(img_reproj).with_colors([[0, 0, 255]]), // blue
    )?;

    // Compute camera center in world coordinates: C = -R^T * t
    let camera_center = -(result.rotation.transpose() * result.translation);
    rec.log(
        "camera_center",
        &rerun::Points3D::new(vec![rerun::Position3D::new(
            camera_center.x,
            camera_center.y,
            camera_center.z,
        )])
        .with_colors(vec![rerun::Color::from_rgb(255, 165, 0)]), // orange
    )?;

    // Compute pose accuracy
    let trans_error = (result.translation - gt_t).length();

    println!("\n=== Results ===");
    println!("Translation error: {trans_error:.3} units");
    if let Some(rmse) = result.reproj_rmse {
        println!("Reprojection RMSE: {rmse:.3} px");
    }

    println!("\n=== Configuration ===");
    println!("To use RANSAC: run with --use-ransac");
    println!("To use direct EPnP: run without --use-ransac");
    println!("To enable LM refinement: run with --refine");

    println!("\n=== Visualization ===");
    println!("- Green points: Observed 2D points");
    println!("- Blue points: Reprojected 3D points using estimated pose");
    println!("- Yellow points: Original 3D cube structure");
    println!("- Orange point: Estimated camera center");
    Ok(())
}
