//! Aria Fisheye624 camera model (radial-tangential thin-prism fisheye).
//!
//! Reference: projectaria-tools `FisheyeRadTanThinPrism` with one focal length
//! (`FISHEYE624`, `projectaria_tools/core/calibration/camera_projections`). The
//! projection matches it to 1e-9 px on the LaMAria `R_11_5cp` SLAM cameras.
//! Unprojection inverts the tangential terms by Newton steps and solves the
//! radial polynomial on `[0, π/2]` only (Newton with a bisection fallback, as
//! [`super::FisheyeCamera`] does), stopping earlier at the polynomial's first
//! fold, so pixels past `θ = 90°` or past the fold return `None`.

use super::CameraModel;
use kornia_algebra::{Vec2F64, Vec3F64};

/// Projectaria's `FisheyeRadTanThinPrism` with one focal length (`FISHEYE624`).
///
/// With `θ = atan2(r, z)`, `θ_d = θ(1 + Σ k_i θ^(2i+2))` and `(a, b) = θ_d/r · (x, y)`:
/// `u = f(a(1 + t) + ρ²p0 + s0ρ² + s1ρ⁴) + cx`, `v = f(b(1 + t) + ρ²p1 + s2ρ² + s3ρ⁴) + cy`,
/// where `ρ² = a² + b²` and `t = 2(p0·a + p1·b)`. Pixels are raw image pixels.
///
/// # Example
///
/// ```rust
/// use kornia_3d::camera::{CameraModel, Fisheye624Camera};
/// use kornia_algebra::Vec3F64;
///
/// let camera = Fisheye624Camera {
///     f: 241.6,
///     cx: 322.9,
///     cy: 240.4,
///     k: [-0.0248, 0.0963, -0.0633, 0.0062, 0.0035, -0.0007],
///     p: [0.0009, 0.0004],
///     s: [-0.0014, -0.0001, -0.0005, -0.0001],
/// };
/// let ray = Vec3F64::new(0.5, 0.5, 1.0);
/// let pixel = camera.project(&ray).unwrap();
/// let back = camera.unproject(&pixel).unwrap();
/// assert!((back - ray.normalize()).length() < 1e-9);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fisheye624Camera {
    /// Focal length (pixels), shared by both axes.
    pub f: f64,
    /// Principal point x (pixels).
    pub cx: f64,
    /// Principal point y (pixels).
    pub cy: f64,
    /// Radial coefficients of `θ³ … θ¹³`.
    pub k: [f64; 6],
    /// Tangential coefficients.
    pub p: [f64; 2],
    /// Thin-prism coefficients `[s0, s1]` for u, `[s2, s3]` for v.
    pub s: [f64; 4],
}

const EPSILON: f64 = 1e-12;
const TANGENTIAL_MAX_ITER: usize = 10;
const TANGENTIAL_EPS: f64 = 1e-12;
/// Enough for bisection alone to reach `RADIAL_EPS` on `[0, π/2]`.
const RADIAL_MAX_ITER: usize = 60;
const RADIAL_EPS: f64 = 1e-12;
/// Samples of `dθ_d/dθ` on `[0, π/2]` when looking for the first fold.
const FOLD_SAMPLES: usize = 64;

impl Fisheye624Camera {
    /// `(θ_d, dθ_d/dθ)`.
    fn radial(&self, theta: f64) -> (f64, f64) {
        let theta2 = theta * theta;
        let mut power = theta2;
        let mut poly = 1.0;
        let mut dpoly = 1.0;
        for (i, k) in self.k.iter().enumerate() {
            poly += k * power;
            dpoly += (2 * i + 3) as f64 * k * power;
            power *= theta2;
        }
        (theta * poly, dpoly)
    }

    /// Tangential and thin-prism terms on the normalised radial point `(a, b)`:
    /// the distorted normalised point and its 2×2 Jacobian `[[∂g0/∂a, ∂g0/∂b], [∂g1/∂a, ∂g1/∂b]]`.
    fn tangential(&self, a: f64, b: f64) -> ([f64; 2], [[f64; 2]; 2]) {
        let [p0, p1] = self.p;
        let [s0, s1, s2, s3] = self.s;
        let rho2 = a * a + b * b;
        let t = 2.0 * (p0 * a + p1 * b);
        let g = [
            a * (1.0 + t) + rho2 * (p0 + s0 + s1 * rho2),
            b * (1.0 + t) + rho2 * (p1 + s2 + s3 * rho2),
        ];
        let prism_u = 2.0 * (s0 + 2.0 * rho2 * s1);
        let prism_v = 2.0 * (s2 + 2.0 * rho2 * s3);
        let jacobian = [
            [
                1.0 + t + 4.0 * a * p0 + a * prism_u,
                2.0 * a * p1 + 2.0 * b * p0 + b * prism_u,
            ],
            [
                2.0 * b * p0 + 2.0 * a * p1 + a * prism_v,
                1.0 + t + 4.0 * b * p1 + b * prism_v,
            ],
        ];
        (g, jacobian)
    }

    /// Inverts [`Self::tangential`] by Newton steps from the distorted point.
    fn untangential(&self, m: [f64; 2]) -> Option<[f64; 2]> {
        let mut ab = m;
        for _ in 0..TANGENTIAL_MAX_ITER {
            let (g, j) = self.tangential(ab[0], ab[1]);
            let e = [g[0] - m[0], g[1] - m[1]];
            let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
            if det.abs() < EPSILON {
                return None;
            }
            let delta = [
                (j[1][1] * e[0] - j[0][1] * e[1]) / det,
                (j[0][0] * e[1] - j[1][0] * e[0]) / det,
            ];
            ab = [ab[0] - delta[0], ab[1] - delta[1]];
            if delta[0].hypot(delta[1]) < TANGENTIAL_EPS {
                return Some(ab);
            }
        }
        None
    }

    /// Bracket `[lo, hi]` holding the solution of `θ_d(θ) = rho` on the part of
    /// `[0, π/2]` before the first fold (`dθ_d/dθ = 0`), or `None` if `θ_d`
    /// never reaches `rho` there. Samples are walked only up to the solution.
    fn radial_bracket(&self, rho: f64) -> Option<(f64, f64)> {
        let step = std::f64::consts::FRAC_PI_2 / FOLD_SAMPLES as f64;
        for i in 1..=FOLD_SAMPLES {
            let theta = step * i as f64;
            let (theta_d, dtheta_d) = self.radial(theta);
            if dtheta_d <= 0.0 {
                // Fold in this step: bisect the sign change of dθ_d/dθ.
                let (mut lo, mut hi) = (theta - step, theta);
                for _ in 0..RADIAL_MAX_ITER {
                    let mid = 0.5 * (lo + hi);
                    if self.radial(mid).1 > 0.0 {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                return (self.radial(lo).0 >= rho).then_some((theta - step, lo));
            }
            if theta_d >= rho {
                return Some((theta - step, theta));
            }
        }
        None
    }

    /// Solves `θ_d(θ) = rho` on `[0, min(fold, π/2)]` by Newton steps, bisecting
    /// whenever a step would leave the bracket (as kornia-3d's
    /// `FisheyeCamera::unproject`). Past a fold `θ_d` falls again, so a bound at
    /// `π/2` alone would reject pixels of pre-fold bearings and break bisection.
    fn unradial(&self, rho: f64) -> Option<f64> {
        let (mut lo, mut hi) = self.radial_bracket(rho)?;
        let mut theta = rho.clamp(lo, hi);
        for _ in 0..RADIAL_MAX_ITER {
            let (theta_d, dtheta_d) = self.radial(theta);
            let residual = theta_d - rho;
            if residual > 0.0 {
                hi = theta;
            } else {
                lo = theta;
            }
            let newton = theta - residual / dtheta_d;
            let next = if newton > lo && newton < hi {
                newton
            } else {
                0.5 * (lo + hi)
            };
            let delta = next - theta;
            theta = next;
            if delta.abs() < RADIAL_EPS {
                return Some(theta);
            }
        }
        None
    }

    /// Raw-to-pinhole pixel magnification at incidence angle `theta`, radial terms only.
    ///
    /// The ratio of the virtual pinhole image `f·tanθ` to the raw image, along
    /// the radius and along the circle of constant `θ`. Tangential and
    /// thin-prism terms are ignored; on the LaMAria cameras they change the
    /// ratios by less than 0.5% up to 79°.
    ///
    /// # Arguments
    ///
    /// * `theta` - Angle between the ray and the optical axis, in radians.
    ///
    /// # Returns
    ///
    /// `(radial, tangential) = (sec²θ / θ_d'(θ), tanθ / θ_d(θ))`, and `(1, 1)`
    /// at `θ = 0`.
    ///
    /// # Errors
    ///
    /// This method is infallible.
    pub fn pinhole_magnification(&self, theta: f64) -> (f64, f64) {
        if theta.abs() < EPSILON {
            return (1.0, 1.0);
        }
        let (theta_d, dtheta_d) = self.radial(theta);
        let cos = theta.cos();
        (1.0 / (cos * cos * dtheta_d), theta.tan() / theta_d)
    }
}

impl CameraModel for Fisheye624Camera {
    /// `None` for `z <= 0`.
    fn project(&self, p_cam: &Vec3F64) -> Option<Vec2F64> {
        self.project_with_jacobian(p_cam).map(|(pixel, _)| pixel)
    }

    fn project_with_jacobian(&self, p_cam: &Vec3F64) -> Option<(Vec2F64, [Vec3F64; 2])> {
        let (x, y, z) = (p_cam.x, p_cam.y, p_cam.z);
        if z <= 0.0 {
            return None;
        }
        // hypot and a relative axis test keep the result independent of the
        // point's scale (x² + y² under- or overflows for tiny or huge points).
        let r = x.hypot(y);
        if r <= EPSILON * z {
            return Some((
                Vec2F64::new(self.cx, self.cy),
                [
                    Vec3F64::new(self.f / z, 0.0, 0.0),
                    Vec3F64::new(0.0, self.f / z, 0.0),
                ],
            ));
        }
        let norm = r.hypot(z);
        let (theta_d, dtheta_d) = self.radial(r.atan2(z));
        // (a, b) = s · (x, y) with s = θ_d / r, as kornia-3d's FisheyeCamera.
        // ∂(a, b)/∂p is written in the unit ratios (x/r, y/r, r/|p|, z/|p|) over
        // |p| so no intermediate under- or overflows for tiny or huge points.
        let s = theta_d / r;
        let (ex, ey) = (x / r, y / r);
        let k = (dtheta_d * (z / norm) - theta_d * (norm / r)) / norm;
        let radial_rows = [
            [s + ex * ex * k, ex * ey * k, -dtheta_d * (x / norm) / norm],
            [ex * ey * k, s + ey * ey * k, -dtheta_d * (y / norm) / norm],
        ];
        let (g, d) = self.tangential(theta_d * ex, theta_d * ey);
        let row = |i: usize| {
            let c = |j: usize| self.f * (d[i][0] * radial_rows[0][j] + d[i][1] * radial_rows[1][j]);
            Vec3F64::new(c(0), c(1), c(2))
        };
        Some((
            Vec2F64::new(self.f * g[0] + self.cx, self.f * g[1] + self.cy),
            [row(0), row(1)],
        ))
    }

    /// `None` when the tangential inversion does not converge or the pixel lies
    /// past the image of `θ = 90°` or of the radial polynomial's first fold.
    fn unproject(&self, pixel: &Vec2F64) -> Option<Vec3F64> {
        let m = [(pixel.x - self.cx) / self.f, (pixel.y - self.cy) / self.f];
        let [a, b] = self.untangential(m)?;
        let rho = a.hypot(b);
        if rho < EPSILON {
            return Some(Vec3F64::new(0.0, 0.0, 1.0));
        }
        let theta = self.unradial(rho)?;
        let sin = theta.sin();
        Some(Vec3F64::new(sin * a / rho, sin * b / rho, theta.cos()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::FisheyeCamera;

    /// LaMAria R_11_5cp `cam0` (`camera-slam-left`), Aria JSON order
    /// `[fx, fy, cx, cy, k0..k5, p0, p1, s0..s3]`.
    fn r11_cam0() -> Fisheye624Camera {
        Fisheye624Camera {
            f: 241.6048925567284,
            cx: 322.895555771704,
            cy: 240.4445599560633,
            k: [
                -0.02484243376501498,
                0.09628938752134535,
                -0.063325466945624,
                0.006200726341569274,
                0.003488761399963396,
                -0.0007297520056856479,
            ],
            p: [0.00089532032215904, 0.000361066580790014],
            s: [
                -0.00136879244387001,
                -0.0001054083415853531,
                -0.0004635611298488213,
                -0.0001049856461262643,
            ],
        }
    }

    fn r11_cam1() -> Fisheye624Camera {
        Fisheye624Camera {
            f: 241.3272717603338,
            cx: 321.6189294210335,
            cy: 239.6221110493106,
            k: [
                -0.02540327399199527,
                0.09654066714494701,
                -0.06397303271827755,
                0.006223153894704157,
                0.003741072868202308,
                -0.0007998563476126205,
            ],
            p: [0.001725615932096635, -0.0009989475964546615],
            s: [
                -0.002626656620305424,
                4.621087294408381e-05,
                0.001412977671638857,
                -8.439958970429877e-05,
            ],
        }
    }

    fn radial_only(camera: Fisheye624Camera) -> Fisheye624Camera {
        Fisheye624Camera {
            p: [0.0; 2],
            s: [0.0; 4],
            ..camera
        }
    }

    fn bearing(theta: f64, phi: f64) -> Vec3F64 {
        Vec3F64::new(
            theta.sin() * phi.cos(),
            theta.sin() * phi.sin(),
            theta.cos(),
        )
    }

    #[test]
    fn radial_only_model_matches_kannala_brandt() {
        let camera = Fisheye624Camera {
            k: [0.0344, -0.0216, 0.0031, -0.0005, 0.0, 0.0],
            ..radial_only(r11_cam0())
        };
        let oracle = FisheyeCamera {
            fx: camera.f,
            fy: camera.f,
            cx: camera.cx,
            cy: camera.cy,
            k1: camera.k[0],
            k2: camera.k[1],
            k3: camera.k[2],
            k4: camera.k[3],
        };
        for p in [
            Vec3F64::new(0.0, 0.0, 2.0),
            Vec3F64::new(0.4, -0.3, 1.5),
            Vec3F64::new(3.0, 1.0, 0.5),
            Vec3F64::new(-1.0, 2.0, 0.1),
        ] {
            let ours = camera.project(&p).unwrap();
            let theirs = oracle.project(&p).unwrap();
            assert!(
                (ours - theirs).length() < 1e-12,
                "{p:?}: {ours:?} vs {theirs:?}"
            );
        }
    }

    /// projectaria-tools 2.4.0 `project_no_checks` on `camera-slam-left` and
    /// `camera-slam-right` of `R_11_5cp.vrs`.
    #[test]
    fn matches_projectaria_tools_on_r11() {
        let cases: [(Fisheye624Camera, [[f64; 5]; 6]); 2] = [
            (
                r11_cam0(),
                [
                    [0.0, 0.0, 1.0, 322.895555771704, 240.4445599560633],
                    [0.2, 0.0, 1.0, 370.5602591385664, 240.4435582668154],
                    [0.0, -0.4, 1.0, 322.8785217456931, 148.6959264719726],
                    [0.5, 0.5, 1.0, 428.2300164353083, 345.8130644648975],
                    [-1.0, 0.3, 1.0, 134.68987051297754, 296.85264441051964],
                    [1.5, -0.8, 1.0, 548.5535397721308, 119.94886366169236],
                ],
            ),
            (
                r11_cam1(),
                [
                    [0.0, 0.0, 1.0, 321.6189294210335, 239.6221110493106],
                    [0.2, 0.0, 1.0, 369.2394677780801, 239.62596689407238],
                    [0.0, -0.4, 1.0, 321.58779467203146, 147.90944636700155],
                    [0.5, 0.5, 1.0, 426.72793338563383, 344.8470155707467],
                    [-1.0, 0.3, 1.0, 134.01639189725068, 295.9183713407115],
                    [1.5, -0.8, 1.0, 547.4155443447492, 119.1613110868152],
                ],
            ),
        ];
        for (camera, rows) in cases {
            for [x, y, z, u, v] in rows {
                let pixel = camera.project(&Vec3F64::new(x, y, z)).unwrap();
                assert!(
                    (pixel - Vec2F64::new(u, v)).length() < 1e-9,
                    "({x}, {y}, {z}): {pixel:?} vs ({u}, {v})"
                );
            }
        }
    }

    #[test]
    fn jacobian_matches_central_differences() {
        let camera = r11_cam0();
        let h = 1e-6;
        let axes = [
            Vec3F64::new(1.0, 0.0, 0.0),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(0.0, 0.0, 1.0),
        ];
        for p in [
            Vec3F64::new(0.0, 0.0, 2.0),
            Vec3F64::new(0.4, -0.3, 1.5),
            Vec3F64::new(-1.0, 0.3, 1.0),
            Vec3F64::new(3.0, -2.0, 0.6),
        ] {
            let (pixel, rows) = camera.project_with_jacobian(&p).unwrap();
            assert_eq!(pixel, camera.project(&p).unwrap());
            for (col, axis) in axes.into_iter().enumerate() {
                let plus = camera.project(&(p + axis * h)).unwrap();
                let minus = camera.project(&(p - axis * h)).unwrap();
                let numeric = (plus - minus) / (2.0 * h);
                let analytic = Vec2F64::new(rows[0].to_array()[col], rows[1].to_array()[col]);
                assert!(
                    (numeric - analytic).length() < 1e-5 * analytic.length().max(1.0),
                    "∂/∂p[{col}] at {p:?}: {analytic:?} vs {numeric:?}"
                );
            }
        }
    }

    /// 2000 bearings at θ ∈ [0°, 79°] round-trip within 1e-9 rad.
    #[test]
    fn unproject_inverts_project_to_79_degrees() {
        let camera = r11_cam0();
        let max_theta = 79f64.to_radians();
        for i in 0..2000 {
            let theta = max_theta * (i as f64 / 1999.0);
            let phi = i as f64 * 2.399_963_229_728_653;
            let ray = bearing(theta, phi);
            let back = camera.unproject(&camera.project(&ray).unwrap()).unwrap();
            let angle = ray.cross(back).length().atan2(ray.dot(back));
            assert!(angle < 1e-9, "θ {theta}, φ {phi}: {angle} rad");
        }
    }

    /// R11 cam0's corner sits at ρ ≈ 1.66, past the θd ≈ 1.505 the radial
    /// polynomial reaches at 90°, so it has no ray.
    #[test]
    fn image_corner_is_past_90_degrees() {
        assert!(r11_cam0().unproject(&Vec2F64::new(0.0, 0.0)).is_none());
    }

    /// Pixels of bearings between 80° and 90° unproject back past 80°.
    #[test]
    fn pixels_between_80_and_90_degrees_unproject_past_80() {
        let camera = r11_cam0();
        for (theta_deg, phi) in [(81.0, 0.3), (85.0, 2.0), (89.0, 4.4)] {
            let theta = f64::to_radians(theta_deg);
            let pixel = camera.project(&bearing(theta, phi)).unwrap();
            let ray = camera.unproject(&pixel).unwrap();
            let back = (ray.z / ray.length()).acos();
            assert!(back >= 80f64.to_radians(), "θ {theta_deg}°: {back} rad");
            assert!((back - theta).abs() < 1e-6, "θ {theta_deg}°: {back} rad");
        }
    }

    /// With `k0 = -0.3` the radial polynomial folds at θ ≈ 1.054 rad, where it
    /// peaks at θd ≈ 0.703, and falls to 0.408 at 90°; the bearing at θ = 1
    /// rad (θd = 0.7) still has a unique ray before the fold.
    #[test]
    fn unproject_brackets_at_the_first_radial_fold() {
        let camera = Fisheye624Camera {
            f: 1.0,
            cx: 0.0,
            cy: 0.0,
            k: [-0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
            p: [0.0; 2],
            s: [0.0; 4],
        };
        let ray = bearing(1.0, 0.0);
        let pixel = camera.project(&ray).unwrap();
        assert!((pixel - Vec2F64::new(0.7, 0.0)).length() < 1e-12);
        let back = camera.unproject(&pixel).unwrap();
        assert!((back - ray).length() < 1e-9, "{back:?} vs {ray:?}");
        // Past the fold's peak there is no ray.
        assert!(camera.unproject(&Vec2F64::new(0.71, 0.0)).is_none());
    }

    /// `project` depends on the direction only, not on the point's scale.
    #[test]
    fn project_is_invariant_to_point_scale() {
        let camera = r11_cam0();
        for direction in [Vec3F64::new(1.0, 0.0, 1.0), Vec3F64::new(-0.3, 0.8, 0.5)] {
            let (pixel, rows) = camera.project_with_jacobian(&direction).unwrap();
            for scale in [1e-200, 5e-13, 1e-6, 1e6, 1e200] {
                let (scaled, scaled_rows) =
                    camera.project_with_jacobian(&(direction * scale)).unwrap();
                assert!(
                    (scaled - pixel).length() < 1e-9,
                    "scale {scale}: {scaled:?} vs {pixel:?}"
                );
                // The Jacobian scales as 1/scale; compare at unit scale, since
                // squaring entries near 1e±200 in `length` under- or overflows.
                for (row, scaled_row) in rows.iter().zip(scaled_rows) {
                    let rescaled = scaled_row * scale;
                    assert!(
                        (rescaled - *row).length() < 1e-9 * row.length(),
                        "scale {scale}: {rescaled:?} vs {row:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn project_rejects_points_at_or_behind_the_camera() {
        assert!(r11_cam0().project(&Vec3F64::new(0.1, 0.0, 0.0)).is_none());
        assert!(r11_cam0().project(&Vec3F64::new(0.1, 0.0, -1.0)).is_none());
    }

    /// Central-difference (radial, tangential) magnification from raw pixels
    /// to virtual-pinhole pixels `f·tanθ` at `(theta, phi)`.
    fn numeric_magnification(camera: &Fisheye624Camera, theta: f64, phi: f64) -> (f64, f64) {
        let h = 1e-6;
        let raw = |t: f64, p: f64| camera.project(&bearing(t, p)).unwrap();
        let d_theta = (raw(theta + h, phi) - raw(theta - h, phi)).length() / (2.0 * h);
        let d_phi = (raw(theta, phi + h) - raw(theta, phi - h)).length() / (2.0 * h);
        let cos = theta.cos();
        (
            camera.f / (cos * cos) / d_theta,
            camera.f * theta.tan() / d_phi,
        )
    }

    fn assert_relative(actual: f64, expected: f64, tolerance: f64, what: &str) {
        assert!(
            ((actual - expected) / expected).abs() < tolerance,
            "{what}: {actual} vs {expected}"
        );
    }

    /// Closed form against differences of `project`, on the radial-only copy.
    #[test]
    fn pinhole_magnification_matches_radial_only_differences() {
        let camera = radial_only(r11_cam0());
        for (degrees, expected_radial) in [
            (10.0, 1.0330),
            (45.0, 1.9154),
            (70.0, 9.0586),
            (79.0, 34.913),
        ] {
            let theta = f64::to_radians(degrees);
            let (radial, tangential) = camera.pinhole_magnification(theta);
            let (numeric_radial, numeric_tangential) = numeric_magnification(&camera, theta, 0.7);
            assert_relative(radial, numeric_radial, 1e-4, "radial");
            assert_relative(tangential, numeric_tangential, 1e-4, "tangential");
            assert_relative(radial, expected_radial, 1e-4, "expected radial");
        }
        assert_eq!(camera.pinhole_magnification(0.0), (1.0, 1.0));
    }

    /// Radial-only closed form against the full model, 16 azimuths.
    #[test]
    fn pinhole_magnification_matches_full_model_within_5e_3() {
        let camera = r11_cam0();
        for degrees in [10.0, 45.0, 70.0, 79.0] {
            let theta = f64::to_radians(degrees);
            let (radial, tangential) = camera.pinhole_magnification(theta);
            for i in 0..16 {
                let phi = i as f64 * std::f64::consts::TAU / 16.0;
                let (numeric_radial, numeric_tangential) =
                    numeric_magnification(&camera, theta, phi);
                assert_relative(radial, numeric_radial, 5e-3, "radial");
                assert_relative(tangential, numeric_tangential, 5e-3, "tangential");
            }
        }
    }
}
