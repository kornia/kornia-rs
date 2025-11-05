use kornia_pnp::IPPE;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Set up rerun
    let rec = rerun::RecordingStreamBuilder::new("IPPE Demo").spawn()?;

    // Define square in normalized coords and estimate pose
    let square_len = 1.0f32;
    let norm = [[-0.5f32, 0.5], [0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]];
    let res = IPPE::solve_square(&norm, square_len)?;

    // Build simple intrinsics for visualization (pixels)
    let k = [[800.0f32, 0.0, 640.0], [0.0, 800.0, 480.0], [0.0, 0.0, 1.0]];

    // Project the canonical 3D square with estimated pose
    let obj3d = [
        [-0.5, 0.5, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, -0.5, 0.0],
        [-0.5, -0.5, 0.0],
    ];

    let r = res.first.rotation;
    let t = res.first.translation;

    let fx = k[0][0];
    let fy = k[1][1];
    let cx = k[0][2];
    let cy = k[1][2];

    let mut proj = [[0.0f32; 2]; 4];
    for (i, p) in obj3d.iter().enumerate() {
        let x = r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2] + t[0];
        let y = r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2] + t[1];
        let z = r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2] + t[2];
        let u = fx * x / z + cx;
        let v = fy * y / z + cy;
        proj[i] = [u, v];
    }

    // Convert normalized inputs to pixels for overlay
    let mut meas = [[0.0f32; 2]; 4];
    for (i, p) in norm.iter().enumerate() {
        meas[i] = [fx * p[0] + cx, fy * p[1] + cy];
    }

    // Log both sets of points
    rec.log(
        "image/points_measured",
        &rerun::Points2D::new(meas).with_colors([[255, 0, 0]]), // red
    )?;
    rec.log(
        "image/points_projected",
        &rerun::Points2D::new(proj).with_colors([[0, 255, 0]]), // green
    )?;

    // Also draw lines forming the projected square
    let lines = vec![[0, 1], [1, 2], [2, 3], [3, 0]];
    rec.log(
        "image/square_edges",
        &rerun::LineStrips2D::new(vec![proj]).with_colors([[0, 255, 0]]),
    )?;
    // Show measured as a line loop too
    rec.log(
        "image/square_measured",
        &rerun::LineStrips2D::new(vec![meas]).with_colors([[255, 0, 0]]),
    )?;

    Ok(())
}
