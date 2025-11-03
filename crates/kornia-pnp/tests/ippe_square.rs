use approx::assert_relative_eq;
use kornia_pnp::IPPE;

#[test]
fn ippe_square_identity_pose() {
    // Square of side 1.0 centered at origin on z=0.
    // Normalized image coordinates match the object plane (camera at z=1).
    let norm = [
        [-0.5f32, 0.5],
        [0.5, 0.5],
        [0.5, -0.5],
        [-0.5, -0.5],
    ];

    let res = IPPE::solve_square(&norm, 1.0).expect("IPPE solve_square should succeed");
    let r = res.first.rotation;
    let t = res.first.translation;

    // Expect approximately identity rotation and t.z ~ 1
    assert_relative_eq!(r[0][0], 1.0, epsilon = 1e-2);
    assert_relative_eq!(r[1][1], 1.0, epsilon = 1e-2);
    assert_relative_eq!(r[2][2], 1.0, epsilon = 1e-2);
    assert_relative_eq!(t[0], 0.0, epsilon = 5e-2);
    assert_relative_eq!(t[1], 0.0, epsilon = 5e-2);
    assert_relative_eq!(t[2], 1.0, epsilon = 5e-2);
}
