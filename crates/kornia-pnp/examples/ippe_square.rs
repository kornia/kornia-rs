use kornia_pnp::IPPE;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Normalized image coordinates for a 1.0-side square centered at origin
    let norm = [[-0.5f32, 0.5], [0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]];

    // Use identity intrinsics; image points are already normalized.
    let k = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    let res = IPPE::solve_square(&norm, &k, 1.0)?;
    println!("R (best): {:?}", res.first.rotation);
    println!("t (best): {:?}", res.first.translation);
    println!("rvec (best): {:?}", res.first.rvec);
    println!("rmse (normalized): {:?}", res.first.reproj_rmse);

    Ok(())
}
