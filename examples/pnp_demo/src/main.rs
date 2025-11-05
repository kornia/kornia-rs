use argh::FromArgs;
use kornia_imgproc::calibration::{
    distortion::{distort_point_polynomial, PolynomialDistortion},
    CameraIntrinsic,
};
use kornia_pnp as kpnp;
use rand::Rng;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
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

    // Run EPnP (baseline) and EPnP+LM (refined)
    let result_epnp = kpnp::solve_pnp(&world_pts, &image_pts, &k, kpnp::PnPMethod::EPnPDefault)?;
    let params_lm = kpnp::EPnPParams {
        refine_lm: Some(kpnp::LMParams::default()),
        ..Default::default()
    };
    let result_lm = kpnp::solve_pnp(&world_pts, &image_pts, &k, kpnp::PnPMethod::EPnP(params_lm))?;

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

    // Helper to reproject with a given pose
    let reproject = |r: [[f32; 3]; 3], t: [f32; 3]| -> Vec<(f32, f32)> {
        let mut out = Vec::with_capacity(world_pts.len());
        for p in &world_pts {
            let x_c = r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2] + t[0];
            let y_c = r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2] + t[1];
            let z_c = r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2] + t[2];
            let u = k[0][0] * x_c / z_c + k[0][2];
            let v = k[1][1] * y_c / z_c + k[1][2];
            out.push((u, v));
        }
        out
    };

    let img_reproj_epnp = reproject(result_epnp.rotation, result_epnp.translation);
    let img_reproj_lm = reproject(result_lm.rotation, result_lm.translation);

    rec.log_static(
        "image/reprojected_epnp",
        &rerun::Points2D::new(img_reproj_epnp).with_colors([[255, 0, 0]]), // red
    )?;
    rec.log_static(
        "image/reprojected_lm",
        &rerun::Points2D::new(img_reproj_lm).with_colors([[0, 128, 255]]), // blue
    )?;

    // Compute camera center in world coordinates: C = -R^T * t
    let r = result_lm.rotation;
    let t = result_lm.translation;
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

    println!("Ground truth translation: {:?}", gt_t);
    println!("Estimated translation (LM): {:?}", result_lm.translation);
    println!("Ground truth rotation   : {:?}", gt_r);
    println!("Estimated rotation (EPnP):\n{:?}", result_epnp.rotation);
    println!("Estimated rotation (LM):\n{:?}", result_lm.rotation);
    if let Some(rmse) = result_epnp.reproj_rmse {
        println!("EPnP RMSE: {:.3} px", rmse);
    }
    if let Some(rmse) = result_lm.reproj_rmse {
        println!("EPnP+LM RMSE: {:.3} px", rmse);
    }
    if let Some(it) = result_lm.num_iterations {
        println!("LM iterations: {}", it);
    }
    if let Some(c) = result_lm.converged {
        println!("LM converged: {}", c);
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
