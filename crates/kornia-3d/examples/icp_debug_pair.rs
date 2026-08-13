//! Debug harness: run projective point-to-plane ICP on two captured RAW1 depth frames
//! (`RAW1` magic + u32 w,h + u16 mm) and print association statistics per configuration.
//!
//! Usage: icp_debug_pair <frame0.raw1> <frame1.raw1> <fx> <fy> <cx> <cy>
//! Intrinsics are for the RGB grid the camera_info describes; they are scaled per-axis
//! onto the depth grid exactly like flux-odom does.

use kornia_3d::registration::{
    icp_projective_plane, DepthIntrinsics, IcpPlaneCriteria, RgbdPyramid,
};

fn load_raw1(path: &str) -> (Vec<u16>, u32, u32) {
    let b = std::fs::read(path).expect("read");
    assert_eq!(&b[0..4], b"RAW1");
    let w = u32::from_le_bytes(b[4..8].try_into().unwrap());
    let h = u32::from_le_bytes(b[8..12].try_into().unwrap());
    let px: Vec<u16> = b[12..12 + (w * h) as usize * 2]
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    (px, w, h)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (d0, w, h) = load_raw1(&a[1]);
    let (d1, w1, h1) = load_raw1(&a[2]);
    assert_eq!((w, h), (w1, h1));
    let (rfx, rfy, rcx, rcy): (f64, f64, f64, f64) = (
        a[3].parse().unwrap(),
        a[4].parse().unwrap(),
        a[5].parse().unwrap(),
        a[6].parse().unwrap(),
    );
    // camera_info is on the 2x RGB grid (640x360) — scale to the depth grid like flux-odom.
    let (sx, sy) = (w as f64 / (w as f64 * 2.0), h as f64 / (h as f64 * 2.0));
    let intr = DepthIntrinsics {
        fx: rfx * sx,
        fy: rfy * sy,
        cx: rcx * sx,
        cy: rcy * sy,
        width: w as usize,
        height: h as usize,
    };
    let valid0 = d0.iter().filter(|&&d| d != 0).count();
    let valid1 = d1.iter().filter(|&&d| d != 0).count();
    println!(
        "dims {w}x{h} valid0 {:.1}% valid1 {:.1}% intr fx={:.1} cx={:.1}",
        100.0 * valid0 as f64 / (w * h) as f64,
        100.0 * valid1 as f64 / (w * h) as f64,
        intr.fx,
        intr.cx
    );
    let p0 = RgbdPyramid::from_depth_mm(&d0, &intr, 3).expect("pyr0");
    let p1 = RgbdPyramid::from_depth_mm(&d1, &intr, 3).expect("pyr1");
    let ident = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for (label, crit) in [
        ("default", IcpPlaneCriteria::default()),
        (
            "loose-gates",
            IcpPlaneCriteria {
                max_dist_m: 0.5,
                max_normal_angle_rad: 1.2,
                ..Default::default()
            },
        ),
    ] {
        match icp_projective_plane(&p1, &p0, ident, [0.0; 3], crit) {
            Ok(r) => println!(
                "{label}: iters {} rmse {:.2}mm inliers {:.2}% t=({:.1},{:.1},{:.1})mm",
                r.iterations,
                r.rmse * 1000.0,
                r.inlier_fraction * 100.0,
                r.translation[0] * 1000.0,
                r.translation[1] * 1000.0,
                r.translation[2] * 1000.0
            ),
            Err(e) => println!("{label}: ERR {e}"),
        }
    }
    // Frame vs ITSELF — must be ~100% inliers at identity if association is sane.
    match icp_projective_plane(&p0, &p0, ident, [0.0; 3], IcpPlaneCriteria::default()) {
        Ok(r) => println!(
            "self: iters {} rmse {:.4}mm inliers {:.2}%",
            r.iterations,
            r.rmse * 1000.0,
            r.inlier_fraction * 100.0
        ),
        Err(e) => println!("self: ERR {e}"),
    }
}
