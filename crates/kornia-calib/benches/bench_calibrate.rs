use criterion::{criterion_group, criterion_main, Criterion};
use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, Vec2F64, Vec3F64};
use kornia_calib::{calibrate_apriltag, CalibConfig, TagObservation};

fn pinhole(fx: f64, fy: f64, cx: f64, cy: f64) -> PinholeCamera {
    PinholeCamera {
        fx,
        fy,
        cx,
        cy,
        k1: 0.0,
        k2: 0.0,
        p1: 0.0,
        p2: 0.0,
    }
}

fn project(pw: Vec3F64, t_world_cam: &Pose3d, k: &PinholeCamera) -> Vec2F64 {
    let pc = t_world_cam.inverse().transform_point(&pw);
    Vec2F64::new(k.fx * pc.x / pc.z + k.cx, k.fy * pc.y / pc.z + k.cy)
}

fn yaw_pose(yaw: f64, t: Vec3F64) -> Pose3d {
    let r = Mat3F64::from_cols(
        Vec3F64::new(yaw.cos(), 0.0, -yaw.sin()),
        Vec3F64::new(0.0, 1.0, 0.0),
        Vec3F64::new(yaw.sin(), 0.0, yaw.cos()),
    );
    Pose3d::new(r, t)
}

fn square(s: f64) -> [Vec3F64; 4] {
    [
        Vec3F64::new(-s, s, 0.0),
        Vec3F64::new(s, s, 0.0),
        Vec3F64::new(s, -s, 0.0),
        Vec3F64::new(-s, -s, 0.0),
    ]
}

fn bench_calibrate(c: &mut Criterion) {
    let ks = [
        pinhole(400.0, 400.0, 320.0, 240.0),
        pinhole(450.0, 450.0, 320.0, 240.0),
        pinhole(500.0, 500.0, 320.0, 240.0),
    ];
    let poses = [
        Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.0, 0.0, -2.5)),
        yaw_pose(0.10, Vec3F64::new(-0.40, 0.0, -2.4)),
        yaw_pose(-0.08, Vec3F64::new(0.35, -0.05, -2.6)),
    ];
    let s = 0.08;
    let sq = square(s);
    let off = |p: Vec3F64| Vec3F64::new(p.x + 0.4, p.y + 0.1, p.z + 0.15);
    let b_world = [off(sq[0]), off(sq[1]), off(sq[2]), off(sq[3])];
    let mk = |w: &[Vec3F64; 4], id: u16| TagObservation {
        tag_id: id,
        per_camera: (0..3)
            .map(|i| {
                (
                    i,
                    [
                        project(w[0], &poses[i], &ks[i]),
                        project(w[1], &poses[i], &ks[i]),
                        project(w[2], &poses[i], &ks[i]),
                        project(w[3], &poses[i], &ks[i]),
                    ],
                )
            })
            .collect(),
    };
    let tags = [mk(&sq, 0), mk(&b_world, 1)];
    let cfg = CalibConfig::new(2.0 * s);

    c.bench_function("calibrate_apriltag_3cam_2tag", |b| {
        b.iter(|| calibrate_apriltag(&ks, &tags, &[], &cfg).unwrap());
    });
}

criterion_group!(benches, bench_calibrate);
criterion_main!(benches);
