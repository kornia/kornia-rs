//! PLY binary point-cloud writer (XYZ + RGB + normals).
//!
//! Emits files that `kornia_3d::io::ply::read_ply_binary` can read back
//! (`PlyType::XYZRgbNormals`): an ASCII header declaring the vertex properties,
//! followed by little-endian binary data of 27 bytes per vertex
//! (`x, y, z` as `f32`, `red, green, blue` as `u8`, `nx, ny, nz` as `f32`).
//!
//! Colours are sampled (before reconstruction) from each track's raw
//! observations; each point then takes the colour of whichever observation
//! survived into the solve. Normals are estimated from the k nearest neighbours
//! via PCA and oriented toward the point's own observing cameras.

use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::num::NonZeroUsize;
use std::path::Path;

use kiddo::immutable::float::kdtree::ImmutableKdTree;
use kornia_3d::pose::Pose3d;
use kornia_algebra::Vec2F64;
use kornia_calib::{FeatureTrack, Reconstruction};
use kornia_image::Image;

/// Number of neighbours used for PCA normal estimation.
const NORMAL_K: usize = 20;

/// A single point-cloud vertex.
pub struct Vertex {
    /// World-frame position.
    pub position: [f64; 3],
    /// RGB colour (0-255).
    pub color: [u8; 3],
    /// Surface normal (unit length).
    pub normal: [f64; 3],
}

/// Assemble coloured, normal-estimated vertices from an SfM reconstruction.
///
/// # Arguments
///
/// * `reconstruction` - Output of `kornia_calib::reconstruct`.
/// * `observation_colors` - Per-track raw-observation colours, from
///   [`sample_observation_colors`]. Sampled before reconstruction so the source
///   frames can be dropped; each point then picks the colour of a *surviving*
///   observation (not necessarily the first raw one).
pub fn build_vertices(
    reconstruction: &Reconstruction,
    observation_colors: &[Vec<(usize, [u8; 3])>],
) -> Vec<Vertex> {
    let points: Vec<[f64; 3]> = reconstruction
        .points
        .iter()
        .map(|p| [p.position.x, p.position.y, p.position.z])
        .collect();
    let axes = pca_normals_axis(&points);

    // Group surviving observations by the point they belong to.
    let mut obs_by_point: Vec<Vec<usize>> = vec![Vec::new(); reconstruction.points.len()];
    for (oi, obs) in reconstruction.observations.iter().enumerate() {
        if let Some(list) = obs_by_point.get_mut(obs.point) {
            list.push(oi);
        }
    }

    reconstruction
        .points
        .iter()
        .enumerate()
        .map(|(pi, pt)| {
            let p = points[pi];
            let axis = axes[pi];
            let track_id = pt.track_id;

            // Observers that actually saw this point (surviving observations).
            let mut observers: Vec<([f64; 3], Option<[u8; 3]>)> = obs_by_point[pi]
                .iter()
                .filter_map(|&oi| {
                    let view = reconstruction.observations[oi].view;
                    let center = view_centre(&reconstruction.views, view)?;
                    Some((center, colour_for_view(observation_colors, track_id, view)))
                })
                .collect();
            // Fallback: the track's raw observations (a filtered point may have
            // no surviving observation).
            if observers.is_empty() {
                if let Some(raw) = observation_colors.get(track_id) {
                    observers = raw
                        .iter()
                        .filter_map(|&(view, color)| {
                            view_centre(&reconstruction.views, view).map(|c| (c, Some(color)))
                        })
                        .collect();
                }
            }

            let (normal, color) = orient_normal_and_pick_color(p, axis, &observers);
            Vertex {
                position: p,
                color,
                normal,
            }
        })
        .collect()
}

/// Write vertices to a binary PLY file.
///
/// # Arguments
///
/// * `path` - Destination file path.
/// * `vertices` - Vertices to write.
///
/// # Errors
///
/// Returns an error if the file cannot be created or written.
pub fn write_ply(path: &Path, vertices: &[Vertex]) -> Result<(), Box<dyn Error>> {
    let mut writer = BufWriter::new(File::create(path)?);

    writeln!(writer, "ply")?;
    writeln!(writer, "format binary_little_endian 1.0")?;
    writeln!(writer, "element vertex {}", vertices.len())?;
    writeln!(writer, "property float x")?;
    writeln!(writer, "property float y")?;
    writeln!(writer, "property float z")?;
    writeln!(writer, "property uchar red")?;
    writeln!(writer, "property uchar green")?;
    writeln!(writer, "property uchar blue")?;
    writeln!(writer, "property float nx")?;
    writeln!(writer, "property float ny")?;
    writeln!(writer, "property float nz")?;
    writeln!(writer, "end_header")?;

    for v in vertices {
        writer.write_all(&(v.position[0] as f32).to_le_bytes())?;
        writer.write_all(&(v.position[1] as f32).to_le_bytes())?;
        writer.write_all(&(v.position[2] as f32).to_le_bytes())?;
        writer.write_all(&v.color)?;
        writer.write_all(&(v.normal[0] as f32).to_le_bytes())?;
        writer.write_all(&(v.normal[1] as f32).to_le_bytes())?;
        writer.write_all(&(v.normal[2] as f32).to_le_bytes())?;
    }
    writer.flush()?;

    Ok(())
}

/// Sample the RGB colour at a (sub-pixel) observation in a video frame.
fn sample_color(rgb_frames: &[Image<u8, 3>], cam_idx: usize, pixel: Vec2F64) -> [u8; 3] {
    let Some(frame) = rgb_frames.get(cam_idx) else {
        return [0; 3];
    };
    let (w, h) = (frame.width(), frame.height());
    if w == 0 || h == 0 {
        return [0; 3];
    }
    let col = (pixel.x.round() as isize).clamp(0, w as isize - 1) as usize;
    let row = (pixel.y.round() as isize).clamp(0, h as isize - 1) as usize;
    let s = &frame.as_slice()[(row * w + col) * 3..][..3];
    [s[0], s[1], s[2]]
}

/// Sample every raw observation's colour from the source RGB frames, indexed by
/// track. Computed before reconstruction so the RGB frames can be dropped; a
/// point then looks up the colour for whichever of its observations survived.
pub fn sample_observation_colors(
    tracks: &[FeatureTrack],
    rgb_frames: &[Image<u8, 3>],
) -> Vec<Vec<(usize, [u8; 3])>> {
    tracks
        .iter()
        .map(|track| {
            track
                .obs
                .iter()
                .map(|&(cam_idx, pixel)| (cam_idx, sample_color(rgb_frames, cam_idx, pixel)))
                .collect()
        })
        .collect()
}

/// World-frame centre of `view`, or `None` if it is unregistered.
///
/// `views` are camera→world (`T_world_cam`), so a camera's centre is simply its
/// translation — no inversion (inverting would give the world→camera term
/// `-Rᵀ·C`, not the centre).
fn view_centre(views: &[Option<Pose3d>], view: usize) -> Option<[f64; 3]> {
    let pose = views.get(view)?.as_ref()?;
    Some([pose.translation.x, pose.translation.y, pose.translation.z])
}

/// Colour recorded for `track_id` at `view`, if that observation exists.
fn colour_for_view(
    observation_colors: &[Vec<(usize, [u8; 3])>],
    track_id: usize,
    view: usize,
) -> Option<[u8; 3]> {
    observation_colors
        .get(track_id)?
        .iter()
        .find(|(v, _)| *v == view)
        .map(|(_, c)| *c)
}

/// Orient a PCA normal axis toward the observer that sees the point most
/// head-on, and return that observer's colour.
///
/// `axis` is a unit normal with an arbitrary sign. Observers are
/// `(camera_centre, colour)`. Choosing the observer whose viewing direction is
/// most aligned with the axis (`max |dot|`) rejects grazing views, and using a
/// point's own observers (not a global mean) keeps surround captures correct.
fn orient_normal_and_pick_color(
    p: [f64; 3],
    axis: [f64; 3],
    observers: &[([f64; 3], Option<[u8; 3]>)],
) -> ([f64; 3], [u8; 3]) {
    let first_color = observers.iter().find_map(|(_, c)| *c).unwrap_or([0, 0, 0]);
    if observers.is_empty() {
        return (axis, first_color);
    }
    let mut best_idx = 0usize;
    let mut best_score = f64::NEG_INFINITY;
    let mut best_dir = [0.0, 0.0, 1.0];
    for (i, (c, _)) in observers.iter().enumerate() {
        let dir = normalize([c[0] - p[0], c[1] - p[1], c[2] - p[2]]);
        let score = dot(axis, dir).abs();
        if score > best_score {
            best_score = score;
            best_idx = i;
            best_dir = dir;
        }
    }
    let normal = if dot(axis, best_dir) < 0.0 {
        [-axis[0], -axis[1], -axis[2]]
    } else {
        axis
    };
    let color = observers[best_idx].1.unwrap_or(first_color);
    (normal, color)
}

/// Smallest-eigenvector (PCA) normal axis for every point; the sign is
/// arbitrary and is fixed later by [`orient_normal_and_pick_color`].
fn pca_normals_axis(points: &[[f64; 3]]) -> Vec<[f64; 3]> {
    if points.is_empty() {
        return Vec::new();
    }
    let kdtree: ImmutableKdTree<f64, u32, 3, 32> = ImmutableKdTree::new_from_slice(points);
    let k = NonZeroUsize::new(NORMAL_K.min(points.len()).max(2)).unwrap();

    points
        .iter()
        .map(|p| {
            let nn = kdtree.nearest_n::<kiddo::SquaredEuclidean>(p, k);
            let neighbours: Vec<[f64; 3]> = nn.iter().map(|nb| points[nb.item as usize]).collect();
            pca_normal(&neighbours)
        })
        .collect()
}

/// Smallest-eigenvector normal of the neighbour covariance via PCA.
fn pca_normal(neighbours: &[[f64; 3]]) -> [f64; 3] {
    let m = mean(neighbours);
    let mut cov = [[0.0; 3]; 3];
    for p in neighbours {
        let d = [p[0] - m[0], p[1] - m[1], p[2] - m[2]];
        for i in 0..3 {
            for j in 0..3 {
                cov[i][j] += d[i] * d[j];
            }
        }
    }

    let (evecs, evals) = jacobi_symmetric(&cov);
    let mut min_i = 0;
    if evals[1] < evals[min_i] {
        min_i = 1;
    }
    if evals[2] < evals[min_i] {
        min_i = 2;
    }
    let v = [evecs[0][min_i], evecs[1][min_i], evecs[2][min_i]];
    normalize(v)
}

/// Jacobi eigen-decomposition of a symmetric 3x3 matrix.
///
/// Returns `(eigenvectors_as_columns, eigenvalues)`.
#[allow(clippy::needless_range_loop)] // fixed-size 3x3 index arithmetic is clearest as ranges
fn jacobi_symmetric(a: &[[f64; 3]; 3]) -> ([[f64; 3]; 3], [f64; 3]) {
    let mut a = *a;
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    for _ in 0..100 {
        let (mut p, mut q, mut max) = (0, 1, a[0][1].abs());
        for i in 0..3 {
            for j in (i + 1)..3 {
                if a[i][j].abs() > max {
                    max = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if a[p][q].abs() < 1e-15 {
            break;
        }

        let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
        let t = if theta >= 0.0 {
            1.0 / (theta + (1.0 + theta * theta).sqrt())
        } else {
            -1.0 / (-theta + (1.0 + theta * theta).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        let (app, aqq, apq) = (a[p][p], a[q][q], a[p][q]);
        a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;

        for i in 0..3 {
            if i != p && i != q {
                let (aip, aiq) = (a[i][p], a[i][q]);
                a[i][p] = c * aip - s * aiq;
                a[p][i] = a[i][p];
                a[i][q] = s * aip + c * aiq;
                a[q][i] = a[i][q];
            }
        }
        for i in 0..3 {
            let (vip, viq) = (v[i][p], v[i][q]);
            v[i][p] = c * vip - s * viq;
            v[i][q] = s * vip + c * viq;
        }
    }

    (v, [a[0][0], a[1][1], a[2][2]])
}

fn mean(pts: &[[f64; 3]]) -> [f64; 3] {
    let n = pts.len() as f64;
    if n == 0.0 {
        return [0.0; 3];
    }
    let mut s = [0.0; 3];
    for p in pts {
        s[0] += p[0];
        s[1] += p[1];
        s[2] += p[2];
    }
    [s[0] / n, s[1] / n, s[2] / n]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalize(v: [f64; 3]) -> [f64; 3] {
    let len = dot(v, v).sqrt();
    if len < 1e-12 {
        [0.0, 0.0, 1.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reconstruction::run_sfm;
    use crate::test_util;
    use kornia_3d::io::ply::{read_ply_binary, PlyType};
    use kornia_image::ImageSize;
    use std::io::BufRead;
    use tempfile::tempdir;

    #[allow(clippy::too_many_arguments)] // test helper mirroring the Vertex field order
    fn vertex(x: f64, y: f64, z: f64, r: u8, g: u8, b: u8, nx: f64, ny: f64, nz: f64) -> Vertex {
        Vertex {
            position: [x, y, z],
            color: [r, g, b],
            normal: [nx, ny, nz],
        }
    }

    fn assert_close(a: &[f64; 3], b: &[f64; 3], tol: f64) {
        for i in 0..3 {
            assert!((a[i] - b[i]).abs() < tol, "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn sample_color_reads_pixel() {
        let mut frame = Image::<u8, 3>::from_size_val(
            ImageSize {
                width: 2,
                height: 1,
            },
            0,
        )
        .unwrap();
        frame.as_slice_mut().copy_from_slice(&[1, 2, 3, 4, 5, 6]);

        assert_eq!(
            sample_color(&[frame.clone()], 0, Vec2F64::new(0.0, 0.0)),
            [1, 2, 3]
        );
        assert_eq!(
            sample_color(&[frame.clone()], 0, Vec2F64::new(1.0, 0.0)),
            [4, 5, 6]
        );
        // Out-of-image pixels clamp to the border.
        assert_eq!(
            sample_color(&[frame.clone()], 0, Vec2F64::new(100.0, 100.0)),
            [4, 5, 6]
        );
        // Unknown camera index falls back to black.
        assert_eq!(sample_color(&[frame], 7, Vec2F64::new(0.0, 0.0)), [0, 0, 0]);
    }

    #[test]
    fn pca_normals_axis_on_plane() {
        let mut points = Vec::new();
        for x in [-1.0, -0.5, 0.0, 0.5, 1.0] {
            for y in [-1.0, -0.5, 0.0, 0.5, 1.0] {
                points.push([x, y, 0.0]);
            }
        }
        let normals = pca_normals_axis(&points);
        assert_eq!(normals.len(), points.len());
        for n in &normals {
            // Axis is ±z (sign is fixed later by the orientation step).
            assert!(n[2].abs() > 0.99, "plane axis should be ±z, got {n:?}");
            let len = dot(*n, *n).sqrt();
            assert!((len - 1.0).abs() < 1e-6, "normal should be unit length");
        }
    }

    #[test]
    fn write_ply_round_trips_with_kornia_reader() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("out.ply");
        let vertices = vec![
            vertex(1.0, 2.0, 3.0, 10, 20, 30, 0.0, 0.0, 1.0),
            vertex(-1.0, -2.0, -3.0, 200, 100, 50, 0.0, 1.0, 0.0),
        ];

        write_ply(&path, &vertices).unwrap();

        let pc = read_ply_binary(&path, PlyType::XYZRgbNormals).unwrap();
        assert_eq!(pc.len(), vertices.len());
        for (i, v) in vertices.iter().enumerate() {
            assert_close(&pc.points()[i], &v.position, 1e-4);
            assert_eq!(pc.colors().unwrap()[i], v.color);
            assert_close(&pc.normals().unwrap()[i], &v.normal, 1e-4);
        }
    }

    #[test]
    fn write_ply_creates_valid_header() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("out.ply");
        let vertices = vec![vertex(0.0, 0.0, 0.0, 0, 0, 0, 0.0, 0.0, 1.0)];

        write_ply(&path, &vertices).unwrap();

        // Read only the ASCII header (up to `end_header`); the data section is binary.
        let file = std::fs::File::open(&path).unwrap();
        let mut lines = std::io::BufReader::new(file).lines();
        let mut header = String::new();
        for line in lines.by_ref() {
            let line = line.unwrap();
            header.push_str(&line);
            header.push('\n');
            if line == "end_header" {
                break;
            }
        }

        assert!(header.starts_with("ply\nformat binary_little_endian 1.0\n"));
        assert!(header.contains("element vertex 1\n"));
        for prop in [
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "property float nx",
            "property float ny",
            "property float nz",
        ] {
            assert!(
                header.lines().any(|l| l == prop),
                "missing header line: {prop}"
            );
        }
        assert!(header.ends_with("end_header\n"));
    }

    #[test]
    fn build_vertices_matches_reconstruction_point_count() {
        let tracks = test_util::synthetic_tracks();
        let recon = run_sfm(
            &tracks,
            test_util::FX,
            test_util::FY,
            test_util::CX,
            test_util::CY,
            test_util::N_FRAMES,
            None,
            crate::reconstruction::ReconstructionOverrides::default(),
        )
        .expect("synthetic scene must reconstruct");

        // Three neutral RGB frames, same size as the synthetic scene's image.
        let mut frames = Vec::new();
        for _ in 0..test_util::N_FRAMES {
            frames.push(
                Image::<u8, 3>::from_size_val(
                    ImageSize {
                        width: 640,
                        height: 480,
                    },
                    128,
                )
                .unwrap(),
            );
        }

        let observation_colors = sample_observation_colors(&tracks, &frames);
        let verts = build_vertices(&recon, &observation_colors);
        assert_eq!(verts.len(), recon.points.len());
        assert!(verts.iter().all(|v| v.normal.iter().all(|c| c.is_finite())));
    }

    #[test]
    fn view_centre_is_pose_translation() {
        use kornia_algebra::{Mat3F64, Vec3F64};
        // Views are camera→world (`T_world_cam`), so the centre IS the
        // translation; inverting first (the old bug) returns -Rᵀ·C.
        let rot = Mat3F64::from_cols(
            Vec3F64::new(0.0, 0.0, -1.0),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(1.0, 0.0, 0.0),
        );
        let pose = Pose3d::new(rot, Vec3F64::new(1.0, 2.0, 3.0));
        let views = [Some(pose), None];
        assert_eq!(view_centre(&views, 0), Some([1.0, 2.0, 3.0]));
        assert_eq!(
            view_centre(&views, 1),
            None,
            "unregistered view has no centre"
        );
        assert_eq!(
            view_centre(&views, 9),
            None,
            "out-of-range view has no centre"
        );
    }

    #[test]
    fn orient_normal_picks_head_on_observer_and_color() {
        let p = [0.0, 0.0, 0.0];
        let axis = [0.0, 0.0, 1.0];
        // A grazing observer to the side, a head-on observer on +z. The
        // head-on one must win even though it is second in the list, and its
        // colour is the one returned.
        let grazing = ([1.0, 0.0, 0.1], Some([9, 9, 9]));
        let head_on = ([0.0, 0.0, 5.0], Some([1, 2, 3]));
        let (normal, color) = orient_normal_and_pick_color(p, axis, &[grazing, head_on]);
        assert!(
            normal[2] > 0.99,
            "normal should point at the head-on camera"
        );
        assert_eq!(color, [1, 2, 3]);

        // An observer on −z flips the axis.
        let (flipped, _) = orient_normal_and_pick_color(p, axis, &[([0.0, 0.0, -5.0], None)]);
        assert!(flipped[2] < -0.99, "normal must flip toward a −z observer");

        // No observers: axis unchanged, colour defaults to black.
        let (unchanged, color) = orient_normal_and_pick_color(p, axis, &[]);
        assert_eq!(unchanged, axis);
        assert_eq!(color, [0, 0, 0]);
    }

    #[test]
    fn sample_observation_colors_reads_each_observation() {
        // Two frames of distinct colours; a track observed in both must carry a
        // colour per view (not just the first).
        let mut f0 = Image::<u8, 3>::from_size_val(
            ImageSize {
                width: 1,
                height: 1,
            },
            0,
        )
        .unwrap();
        f0.as_slice_mut().copy_from_slice(&[10, 20, 30]);
        let mut f1 = Image::<u8, 3>::from_size_val(
            ImageSize {
                width: 1,
                height: 1,
            },
            0,
        )
        .unwrap();
        f1.as_slice_mut().copy_from_slice(&[40, 50, 60]);
        let tracks = vec![FeatureTrack {
            obs: vec![(0, Vec2F64::new(0.0, 0.0)), (1, Vec2F64::new(0.0, 0.0))],
        }];
        let colors = sample_observation_colors(&tracks, &[f0, f1]);
        assert_eq!(colors.len(), 1);
        assert_eq!(colors[0], vec![(0, [10, 20, 30]), (1, [40, 50, 60])]);
    }
}
