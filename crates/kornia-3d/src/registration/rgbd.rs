//! Depth-image (RGBD) geometry: vertex maps, normal maps and depth pyramids.
//!
//! Conventions used throughout this module (and the projective ICP built on it):
//!
//! * Camera model: OpenCV pinhole, +Z forward, +X right, +Y down.
//! * Depth images: `u16` in millimetres, row-major, `0` = invalid (no measurement).
//! * Vertex maps: `[f32; 3]` camera-frame points in metres, `[0, 0, 0]` = invalid
//!   (a valid vertex always has `z > 0`).
//! * Normal maps: unit `[f32; 3]` oriented towards the camera (`n · v < 0`),
//!   `[0, 0, 0]` = invalid.

/// Errors produced by the RGBD depth-geometry and projective ICP routines.
#[derive(Debug, thiserror::Error)]
pub enum RgbdIcpError {
    /// The depth buffer length does not match the intrinsics dimensions.
    #[error("depth buffer has {got} pixels, expected {width}x{height}={expected}")]
    InvalidDepthSize {
        /// Image width in pixels.
        width: usize,
        /// Image height in pixels.
        height: usize,
        /// Expected number of pixels (`width * height`).
        expected: usize,
        /// Actual buffer length.
        got: usize,
    },
    /// The vertex map length does not match the given dimensions.
    #[error("vertex map has {got} entries, expected {width}x{height}={expected}")]
    InvalidVertexMapSize {
        /// Image width in pixels.
        width: usize,
        /// Image height in pixels.
        height: usize,
        /// Expected number of entries (`width * height`).
        expected: usize,
        /// Actual buffer length.
        got: usize,
    },
    /// A pyramid was requested with zero levels.
    #[error("pyramid needs at least 1 level, got {0}")]
    InvalidNumLevels(usize),
    /// The image becomes too small to process at the requested pyramid level.
    #[error("image is {width}x{height} at pyramid level {level}; need at least 3x3")]
    ImageTooSmall {
        /// Pyramid level at which the image became too small (0 = finest).
        level: usize,
        /// Image width at that level.
        width: usize,
        /// Image height at that level.
        height: usize,
    },
    /// The point-to-plane normal equations are (near-)singular: the observed
    /// geometry does not constrain all six pose DoF (e.g. a single plane /
    /// blank wall).
    #[error("point-to-plane normal equations are singular (degenerate geometry)")]
    SingularNormalEquations,
    /// Too few correspondences survived projective association and gating to
    /// constrain a pose — low overlap or a bad initial guess, as opposed to
    /// degenerate geometry ([`SingularNormalEquations`](Self::SingularNormalEquations)).
    /// Recovery differs (re-initialize / widen gates vs hold pose), so the two
    /// causes are distinct variants.
    #[error("only {0} correspondences survived gating; need at least 6")]
    TooFewCorrespondences(usize),
}

/// Pinhole intrinsics of a depth image (OpenCV convention, +Z forward).
///
/// Pixel centers sit at integer coordinates, i.e. pixel `(u, v)` covers the
/// continuous range `[u - 0.5, u + 0.5) x [v - 0.5, v + 0.5)`.
#[derive(Debug, Clone, PartialEq)]
pub struct DepthIntrinsics {
    /// Focal length in pixels along x.
    pub fx: f64,
    /// Focal length in pixels along y.
    pub fy: f64,
    /// Principal point x in pixels.
    pub cx: f64,
    /// Principal point y in pixels.
    pub cy: f64,
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
}

impl DepthIntrinsics {
    /// Back-project pixel `(u, v)` at depth `z` metres into a camera-frame point.
    pub fn unproject(&self, u: f64, v: f64, z: f64) -> [f64; 3] {
        [(u - self.cx) * z / self.fx, (v - self.cy) * z / self.fy, z]
    }

    /// Project a camera-frame point to continuous pixel coordinates `[u, v]`.
    ///
    /// Returns `None` for points at or behind the camera (`z <= 0`).
    pub fn project(&self, point: &[f64; 3]) -> Option<[f64; 2]> {
        if point[2] <= 0.0 {
            return None;
        }
        Some([
            self.fx * point[0] / point[2] + self.cx,
            self.fy * point[1] / point[2] + self.cy,
        ])
    }

    /// Intrinsics of the half-resolution image produced by [`downsample_depth_half`].
    ///
    /// With pixel centers at integer coordinates, the continuous coordinates of
    /// the two grids relate as `u_full = 2 * u_half + 0.5` (a half-res pixel
    /// covers a 2x2 full-res block whose center lies at `2u + 0.5`). Substituting
    /// into the projection equation gives `fx' = fx / 2` and
    /// `cx' = (cx + 0.5) / 2 - 0.5` (likewise for y). Dimensions floor-halve.
    pub fn half(&self) -> DepthIntrinsics {
        DepthIntrinsics {
            fx: self.fx / 2.0,
            fy: self.fy / 2.0,
            cx: (self.cx + 0.5) / 2.0 - 0.5,
            cy: (self.cy + 0.5) / 2.0 - 0.5,
            width: self.width / 2,
            height: self.height / 2,
        }
    }
}

/// Returns true if a vertex-map entry is a valid measurement (`z > 0`).
#[inline]
pub fn is_valid_vertex(v: &[f32; 3]) -> bool {
    v[2] > 0.0
}

/// Returns true if a normal-map entry is valid (non-zero).
#[inline]
pub fn is_valid_normal(n: &[f32; 3]) -> bool {
    n[0] != 0.0 || n[1] != 0.0 || n[2] != 0.0
}

/// Back-project a `u16` millimetre depth image into a camera-frame vertex map in metres.
///
/// Invalid depth (`0`) maps to `[0, 0, 0]`. `out` is resized to `width * height`.
pub fn depth_to_vertex_map(
    depth_mm: &[u16],
    intr: &DepthIntrinsics,
    out: &mut Vec<[f32; 3]>,
) -> Result<(), RgbdIcpError> {
    let expected = intr.width * intr.height;
    if depth_mm.len() != expected {
        return Err(RgbdIcpError::InvalidDepthSize {
            width: intr.width,
            height: intr.height,
            expected,
            got: depth_mm.len(),
        });
    }

    out.clear();
    out.resize(expected, [0.0; 3]);

    for v in 0..intr.height {
        let row = &depth_mm[v * intr.width..(v + 1) * intr.width];
        let out_row = &mut out[v * intr.width..(v + 1) * intr.width];
        for (u, (&d, out_px)) in row.iter().zip(out_row.iter_mut()).enumerate() {
            if d == 0 {
                continue;
            }
            let z = d as f64 * 1e-3;
            let p = intr.unproject(u as f64, v as f64, z);
            *out_px = [p[0] as f32, p[1] as f32, p[2] as f32];
        }
    }

    Ok(())
}

/// Compute a normal map from a vertex map via the cross product of central differences.
///
/// For each interior pixel with a valid center and four valid neighbors, the
/// normal is `normalize((v[u+1] - v[u-1]) x (v[v+1] - v[v-1]))`, flipped to face
/// the camera (`n · vertex < 0`). Border pixels and pixels with any invalid
/// neighbor get `[0, 0, 0]`. `out` is resized to `width * height`.
pub fn vertex_map_to_normal_map(
    vertices: &[[f32; 3]],
    width: usize,
    height: usize,
    out: &mut Vec<[f32; 3]>,
) -> Result<(), RgbdIcpError> {
    let expected = width * height;
    if vertices.len() != expected {
        return Err(RgbdIcpError::InvalidVertexMapSize {
            width,
            height,
            expected,
            got: vertices.len(),
        });
    }

    out.clear();
    out.resize(expected, [0.0; 3]);

    if width < 3 || height < 3 {
        return Ok(());
    }

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let idx = y * width + x;
            let center = &vertices[idx];
            let left = &vertices[idx - 1];
            let right = &vertices[idx + 1];
            let up = &vertices[idx - width];
            let down = &vertices[idx + width];
            if !is_valid_vertex(center)
                || !is_valid_vertex(left)
                || !is_valid_vertex(right)
                || !is_valid_vertex(up)
                || !is_valid_vertex(down)
            {
                continue;
            }

            let du = [right[0] - left[0], right[1] - left[1], right[2] - left[2]];
            let dv = [down[0] - up[0], down[1] - up[1], down[2] - up[2]];
            let mut n = [
                du[1] * dv[2] - du[2] * dv[1],
                du[2] * dv[0] - du[0] * dv[2],
                du[0] * dv[1] - du[1] * dv[0],
            ];

            let norm = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if norm < 1e-12 {
                continue;
            }

            // orient towards the camera: the view ray is the vertex itself
            let flip = if n[0] * center[0] + n[1] * center[1] + n[2] * center[2] > 0.0 {
                -1.0
            } else {
                1.0
            };
            n = [flip * n[0] / norm, flip * n[1] / norm, flip * n[2] / norm];
            out[idx] = n;
        }
    }

    Ok(())
}

/// Maximum in-block depth spread that is still averaged together when
/// downsampling; values further than this from the block median are treated as
/// belonging to a different surface (depth discontinuity) and excluded.
const DOWNSAMPLE_JUMP_MM: i32 = 50;

/// Downsample a `u16` millimetre depth image by 2x, validity- and discontinuity-aware.
///
/// Each output pixel covers a 2x2 input block: invalid (`0`) samples are
/// ignored, the lower median of the valid samples is taken as pivot, and only
/// samples within [`DOWNSAMPLE_JUMP_MM`] of the pivot are averaged. This never
/// averages across depth discontinuities and never fills holes: a fully
/// invalid block stays `0`. `out` is resized to `(width / 2) * (height / 2)`
/// (floor; a trailing odd row/column is dropped).
pub fn downsample_depth_half(
    depth_mm: &[u16],
    width: usize,
    height: usize,
    out: &mut Vec<u16>,
) -> Result<(), RgbdIcpError> {
    let expected = width * height;
    if depth_mm.len() != expected {
        return Err(RgbdIcpError::InvalidDepthSize {
            width,
            height,
            expected,
            got: depth_mm.len(),
        });
    }

    let half_w = width / 2;
    let half_h = height / 2;
    out.clear();
    out.resize(half_w * half_h, 0);

    for y in 0..half_h {
        let row0 = &depth_mm[2 * y * width..];
        let row1 = &depth_mm[(2 * y + 1) * width..];
        for x in 0..half_w {
            let block = [row0[2 * x], row0[2 * x + 1], row1[2 * x], row1[2 * x + 1]];
            let mut valid = [0u16; 4];
            let mut n = 0;
            for &d in &block {
                if d != 0 {
                    valid[n] = d;
                    n += 1;
                }
            }
            if n == 0 {
                continue;
            }
            valid[..n].sort_unstable();
            // lower median: always an actual sample, so it cannot itself
            // straddle a discontinuity
            let pivot = valid[(n - 1) / 2] as i32;
            let mut sum = 0u32;
            let mut count = 0u32;
            for &d in &valid[..n] {
                if (d as i32 - pivot).abs() <= DOWNSAMPLE_JUMP_MM {
                    sum += d as u32;
                    count += 1;
                }
            }
            out[y * half_w + x] = ((sum + count / 2) / count) as u16;
        }
    }

    Ok(())
}

/// One resolution level of an [`RgbdPyramid`]: intrinsics plus vertex and normal maps.
#[derive(Debug, Clone)]
pub struct RgbdLevel {
    /// Intrinsics at this level's resolution.
    pub intrinsics: DepthIntrinsics,
    /// Camera-frame vertex map in metres, `[0, 0, 0]` = invalid.
    pub vertices: Vec<[f32; 3]>,
    /// Unit normal map oriented towards the camera, `[0, 0, 0]` = invalid.
    pub normals: Vec<[f32; 3]>,
}

/// Coarse-to-fine pyramid of vertex/normal maps built from a single depth image.
///
/// `levels[0]` is the finest (input) resolution; each subsequent level halves
/// the resolution via [`downsample_depth_half`] and [`DepthIntrinsics::half`].
#[derive(Debug, Clone)]
pub struct RgbdPyramid {
    /// Pyramid levels, finest first.
    pub levels: Vec<RgbdLevel>,
}

impl RgbdPyramid {
    /// Build a pyramid with `num_levels` levels from a `u16` millimetre depth image.
    pub fn from_depth_mm(
        depth_mm: &[u16],
        intr: &DepthIntrinsics,
        num_levels: usize,
    ) -> Result<Self, RgbdIcpError> {
        if num_levels == 0 {
            return Err(RgbdIcpError::InvalidNumLevels(num_levels));
        }

        let mut levels = Vec::with_capacity(num_levels);
        // level 0 reads `depth_mm` directly (no copy — this sits on per-frame hot paths);
        // `owned` holds the downsampled buffer from level 1 on.
        let mut owned: Vec<u16> = Vec::new();
        let mut level_intr = intr.clone();

        for level in 0..num_levels {
            if level > 0 {
                let mut half = Vec::new();
                let cur: &[u16] = if level == 1 { depth_mm } else { &owned };
                downsample_depth_half(cur, level_intr.width, level_intr.height, &mut half)?;
                owned = half;
                level_intr = level_intr.half();
            }
            if level_intr.width < 3 || level_intr.height < 3 {
                return Err(RgbdIcpError::ImageTooSmall {
                    level,
                    width: level_intr.width,
                    height: level_intr.height,
                });
            }
            let depth: &[u16] = if level == 0 { depth_mm } else { &owned };
            let mut vertices = Vec::new();
            depth_to_vertex_map(depth, &level_intr, &mut vertices)?;
            let mut normals = Vec::new();
            vertex_map_to_normal_map(&vertices, level_intr.width, level_intr.height, &mut normals)?;
            levels.push(RgbdLevel {
                intrinsics: level_intr.clone(),
                vertices,
                normals,
            });
        }

        Ok(Self { levels })
    }
}

#[cfg(test)]
mod tests {
    use super::super::synth::{render_depth_mm, render_vertex_map, Scene};
    use super::*;

    const IDENTITY_ROT: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    fn test_intrinsics() -> DepthIntrinsics {
        DepthIntrinsics {
            fx: 200.0,
            fy: 200.0,
            cx: 159.5,
            cy: 89.5,
            width: 320,
            height: 180,
        }
    }

    #[test]
    fn test_vertex_map_round_trip() -> Result<(), Box<dyn std::error::Error>> {
        let intr = test_intrinsics();
        let depth = render_depth_mm(&Scene::corner_and_sphere(), &intr, &IDENTITY_ROT, &[0.0; 3]);

        let mut verts = Vec::new();
        depth_to_vertex_map(&depth, &intr, &mut verts)?;

        let mut checked = 0;
        for v in 0..intr.height {
            for u in 0..intr.width {
                let idx = v * intr.width + u;
                let d = depth[idx];
                let p = verts[idx];
                if d == 0 {
                    assert_eq!(p, [0.0; 3], "invalid depth must give [0,0,0] at ({u},{v})");
                    continue;
                }
                assert!(is_valid_vertex(&p));
                // z is the depth in metres
                assert!((p[2] as f64 - d as f64 * 1e-3).abs() < 1e-6);
                // project(unproject(u, v, z)) == (u, v)
                let uv = intr
                    .project(&[p[0] as f64, p[1] as f64, p[2] as f64])
                    .ok_or("valid vertex must project")?;
                assert!(
                    (uv[0] - u as f64).abs() < 1e-3 && (uv[1] - v as f64).abs() < 1e-3,
                    "round trip failed at ({u},{v}): got ({}, {})",
                    uv[0],
                    uv[1]
                );
                checked += 1;
            }
        }
        assert!(
            checked > 10_000,
            "scene should cover most pixels: {checked}"
        );

        // pure f64 unproject/project round trip on a hand-picked point
        let p = intr.unproject(37.25, 101.75, 1.234);
        let uv = intr.project(&p).ok_or("must project")?;
        assert!((uv[0] - 37.25).abs() < 1e-12 && (uv[1] - 101.75).abs() < 1e-12);

        Ok(())
    }

    #[test]
    fn test_tilted_plane_normals_within_one_degree() -> Result<(), Box<dyn std::error::Error>> {
        let intr = test_intrinsics();
        // plane tilted 20 degrees about the x-axis, ~1.5 m in front of the camera
        let (s, c) = (20.0f64).to_radians().sin_cos();
        let scene = Scene {
            planes: vec![super::super::synth::Plane {
                normal: [0.0, s, c],
                d: 1.5,
            }],
            spheres: vec![],
        };
        // camera-facing ground-truth normal (n_z < 0)
        let n_gt = [0.0, -s, -c];

        // exact (float) vertex map, so the test isolates the cross-product math
        // from u16 millimetre quantization
        let verts = render_vertex_map(&scene, &intr, &IDENTITY_ROT, &[0.0; 3]);
        let mut normals = Vec::new();
        vertex_map_to_normal_map(&verts, intr.width, intr.height, &mut normals)?;

        let max_angle_rad = 1.0f64.to_radians();
        let mut checked = 0;
        for (idx, n) in normals.iter().enumerate() {
            if !is_valid_normal(n) {
                continue;
            }
            let dot = (n[0] as f64 * n_gt[0] + n[1] as f64 * n_gt[1] + n[2] as f64 * n_gt[2])
                .clamp(-1.0, 1.0);
            let angle = dot.acos();
            assert!(
                angle < max_angle_rad,
                "normal at pixel {idx} off by {} deg",
                angle.to_degrees()
            );
            checked += 1;
        }
        // all interior pixels see the plane
        assert_eq!(checked, (intr.width - 2) * (intr.height - 2));

        Ok(())
    }

    #[test]
    fn test_downsample_holes_and_discontinuities() -> Result<(), Box<dyn std::error::Error>> {
        // 8x8: left half at 1000 mm, right half at 2000 mm, one hole block,
        // one block with a single valid sample
        let (w, h) = (8usize, 8usize);
        let mut depth = vec![0u16; w * h];
        for y in 0..h {
            for x in 0..w {
                depth[y * w + x] = if x < 4 { 1000 } else { 2000 };
            }
        }
        // fully invalid 2x2 block at output (0, 0)
        depth[0] = 0;
        depth[1] = 0;
        depth[w] = 0;
        depth[w + 1] = 0;
        // single valid sample in block at output (1, 1): keep only (2, 2)
        depth[2 * w + 3] = 0;
        depth[3 * w + 2] = 0;
        depth[3 * w + 3] = 0;
        // 3-valid block at output (0, 2) (covers input x 4..6, y 0..2)
        // straddling the discontinuity: one foreground sample, two background
        depth[w + 4] = 0;
        depth[4] = 1000; // foreground intruding into the background side
        depth[5] = 2000;
        depth[w + 5] = 2000;

        let mut half = Vec::new();
        downsample_depth_half(&depth, w, h, &mut half)?;
        let hw = w / 2;

        // hole block stays invalid — holes never get filled
        assert_eq!(half[0], 0);
        // single valid sample passes through unchanged
        assert_eq!(half[hw + 1], 1000);
        // uniform blocks average exactly
        assert_eq!(half[2 * hw], 1000);
        assert_eq!(half[2 * hw + 3], 2000);
        // mixed 1000/2000 block must side with the median surface, never smear
        let mixed = half[2];
        assert_eq!(
            mixed, 2000,
            "block with samples [1000, 2000, 2000] must resolve to the median surface"
        );
        // no output pixel anywhere may be an average across the 1 m jump
        for &d in &half {
            assert!(
                d == 0 || (d as i32 - 1500).abs() > 400,
                "smeared depth {d} across discontinuity"
            );
        }

        Ok(())
    }

    #[test]
    fn test_pyramid_levels_and_intrinsics() -> Result<(), Box<dyn std::error::Error>> {
        let intr = test_intrinsics();
        let mut depth =
            render_depth_mm(&Scene::corner_and_sphere(), &intr, &IDENTITY_ROT, &[0.0; 3]);
        // punch a 4x4 hole so coarser levels must keep it invalid
        for y in 60..64 {
            for x in 100..104 {
                depth[y * intr.width + x] = 0;
            }
        }

        let pyr = RgbdPyramid::from_depth_mm(&depth, &intr, 3)?;
        assert_eq!(pyr.levels.len(), 3);

        let mut expected = intr.clone();
        for (i, level) in pyr.levels.iter().enumerate() {
            if i > 0 {
                expected = expected.half();
            }
            assert_eq!(level.intrinsics, expected, "intrinsics at level {i}");
            let n = level.intrinsics.width * level.intrinsics.height;
            assert_eq!(level.vertices.len(), n);
            assert_eq!(level.normals.len(), n);
            assert!(
                level.vertices.iter().filter(|v| is_valid_vertex(v)).count() > n / 2,
                "level {i} should be mostly valid"
            );
        }
        assert_eq!(pyr.levels[1].intrinsics.width, 160);
        assert_eq!(pyr.levels[1].intrinsics.height, 90);
        assert_eq!(pyr.levels[2].intrinsics.width, 80);
        assert_eq!(pyr.levels[2].intrinsics.height, 45);
        // cx convention: cx' = (cx + 0.5) / 2 - 0.5
        assert!((pyr.levels[1].intrinsics.cx - ((159.5 + 0.5) / 2.0 - 0.5)).abs() < 1e-12);
        assert!((pyr.levels[1].intrinsics.fx - 100.0).abs() < 1e-12);

        // the hole survives downsampling: level-1 block covering the hole center
        // must be invalid (holes propagate, they are never filled)
        let l1 = &pyr.levels[1];
        assert_eq!(l1.vertices[31 * l1.intrinsics.width + 51][2], 0.0);

        // errors
        assert!(matches!(
            RgbdPyramid::from_depth_mm(&depth, &intr, 0),
            Err(RgbdIcpError::InvalidNumLevels(0))
        ));
        assert!(matches!(
            RgbdPyramid::from_depth_mm(&depth, &intr, 8),
            Err(RgbdIcpError::ImageTooSmall { .. })
        ));

        Ok(())
    }
}
