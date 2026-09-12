//! PLY binary point-cloud writer (XYZ + RGB + normals).
//!
//! Emits files that `kornia_3d::io::ply::read_ply_binary` can read back
//! (`PlyType::XYZRgbNormals`): an ASCII header declaring the vertex properties,
//! followed by little-endian binary data of 27 bytes per vertex
//! (`x, y, z` as `f32`, `red, green, blue` as `u8`, `nx, ny, nz` as `f32`).
//!
//! Colors are sampled from the source RGB frames at each reconstructed point's
//! first track observation. Normals are estimated from the k nearest neighbours
//! via PCA and oriented toward the cameras.

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
/// * `tracks` - The feature tracks the reconstruction was built from (used to
///   look up each point's source pixels for colour sampling).
/// * `rgb_frames` - The original RGB video frames, indexed by camera/frame.
pub fn build_vertices(
    reconstruction: &Reconstruction,
    tracks: &[FeatureTrack],
    rgb_frames: &[Image<u8, 3>],
) -> Vec<Vertex> {
    let points: Vec<[f64; 3]> = reconstruction
        .points
        .iter()
        .map(|p| [p.position.x, p.position.y, p.position.z])
        .collect();
    let colors = extract_colors(reconstruction, tracks, rgb_frames);
    let cameras = camera_centers(&reconstruction.views);
    let normals = estimate_normals(&points, &cameras);

    points
        .into_iter()
        .zip(colors)
        .zip(normals)
        .map(|((position, color), normal)| Vertex {
            position,
            color,
            normal,
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

/// Extract one RGB colour per reconstructed point, sampled from the first
/// camera observation of each point's track.
fn extract_colors(
    reconstruction: &Reconstruction,
    tracks: &[FeatureTrack],
    rgb_frames: &[Image<u8, 3>],
) -> Vec<[u8; 3]> {
    reconstruction
        .points
        .iter()
        .map(|pt| {
            let Some(track) = tracks.get(pt.track_id) else {
                return [0; 3];
            };
            let Some(&(cam_idx, pixel)) = track.obs.first() else {
                return [0; 3];
            };
            sample_color(rgb_frames, cam_idx, pixel)
        })
        .collect()
}

/// World-frame positions of the registered cameras.
fn camera_centers(views: &[Option<Pose3d>]) -> Vec<[f64; 3]> {
    views
        .iter()
        .filter_map(|v| v.as_ref())
        .map(|pose| {
            let inv = pose.inverse();
            [inv.translation.x, inv.translation.y, inv.translation.z]
        })
        .collect()
}

/// Estimate surface normals from the k nearest neighbours via PCA, oriented
/// toward the average camera position.
fn estimate_normals(points: &[[f64; 3]], cameras: &[[f64; 3]]) -> Vec<[f64; 3]> {
    if points.is_empty() {
        return Vec::new();
    }
    let kdtree: ImmutableKdTree<f64, u32, 3, 32> = ImmutableKdTree::new_from_slice(points);
    let k = NonZeroUsize::new(NORMAL_K.min(points.len()).max(2)).unwrap();

    let cam_mean = mean(cameras);
    points
        .iter()
        .map(|p| {
            let nn = kdtree.nearest_n::<kiddo::SquaredEuclidean>(p, k);
            let neighbours: Vec<[f64; 3]> = nn.iter().map(|nb| points[nb.item as usize]).collect();
            let mut normal = pca_normal(&neighbours);
            if !cameras.is_empty() {
                // Flip so the normal points toward the cameras.
                let to_cam = [cam_mean[0] - p[0], cam_mean[1] - p[1], cam_mean[2] - p[2]];
                if dot(normal, to_cam) < 0.0 {
                    normal = [-normal[0], -normal[1], -normal[2]];
                }
            }
            normal
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
    fn estimate_normals_on_plane_point_up() {
        let mut points = Vec::new();
        for x in [-1.0, -0.5, 0.0, 0.5, 1.0] {
            for y in [-1.0, -0.5, 0.0, 0.5, 1.0] {
                points.push([x, y, 0.0]);
            }
        }
        let cameras = [[0.0, 0.0, 5.0]];
        let normals = estimate_normals(&points, &cameras);

        assert_eq!(normals.len(), points.len());
        for n in &normals {
            assert!(n[2] > 0.99, "plane normal should point up, got {n:?}");
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

        let verts = build_vertices(&recon, &tracks, &frames);
        assert_eq!(verts.len(), recon.points.len());
        assert!(verts.iter().all(|v| v.normal.iter().all(|c| c.is_finite())));
    }
}
