use std::io::{BufWriter, Error, ErrorKind, Write};
use std::path::Path;

use super::PlyError;
use crate::mesh::TriangleMesh;

/// Writes a triangle mesh as a binary little-endian PLY file.
///
/// Vertices carry `float x y z`, plus `float nx ny nz` when the mesh has
/// normals and `uchar red green blue` when it has colours; faces are
/// `list uchar uint vertex_indices`. The bytes depend only on the mesh.
///
/// # Arguments
///
/// * `path` - Output file.
/// * `mesh` - The mesh; `normals` and `colors` must be empty or one per
///   vertex, and every index must address a vertex.
///
/// # Returns
///
/// `Ok(())` once the file is written.
///
/// # Errors
///
/// [`PlyError::Io`] on a write failure, or with
/// [`std::io::ErrorKind::InvalidInput`] for an inconsistent mesh.
///
/// # Example
///
/// ```no_run
/// use kornia_3d::io::ply::write_ply_mesh;
/// use kornia_3d::mesh::TriangleMesh;
///
/// let mesh = TriangleMesh {
///     vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
///     triangles: vec![[0, 1, 2]],
///     ..TriangleMesh::default()
/// };
/// write_ply_mesh("triangle.ply", &mesh)?;
/// # Ok::<(), kornia_3d::io::ply::PlyError>(())
/// ```
pub fn write_ply_mesh(path: impl AsRef<Path>, mesh: &TriangleMesh) -> Result<(), PlyError> {
    let n = mesh.vertices.len();
    let invalid = |msg: &str| PlyError::Io(Error::new(ErrorKind::InvalidInput, msg));
    let normals = !mesh.normals.is_empty();
    let colors = !mesh.colors.is_empty();
    if normals && mesh.normals.len() != n {
        return Err(invalid("normals must be empty or one per vertex"));
    }
    if colors && mesh.colors.len() != n {
        return Err(invalid("colors must be empty or one per vertex"));
    }
    if mesh.triangles.iter().flatten().any(|&i| i as usize >= n) {
        return Err(invalid("triangle index out of range"));
    }

    let mut w = BufWriter::new(std::fs::File::create(path)?);
    let mut header = format!(
        "ply\nformat binary_little_endian 1.0\nelement vertex {n}\n\
         property float x\nproperty float y\nproperty float z\n"
    );
    if normals {
        header += "property float nx\nproperty float ny\nproperty float nz\n";
    }
    if colors {
        header += "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    }
    header += &format!(
        "element face {}\nproperty list uchar uint vertex_indices\nend_header\n",
        mesh.triangles.len()
    );
    w.write_all(header.as_bytes())?;
    for i in 0..n {
        for v in mesh.vertices[i] {
            w.write_all(&v.to_le_bytes())?;
        }
        if normals {
            for v in mesh.normals[i] {
                w.write_all(&v.to_le_bytes())?;
            }
        }
        if colors {
            w.write_all(&mesh.colors[i])?;
        }
    }
    for t in &mesh.triangles {
        w.write_all(&[3u8])?;
        for i in t {
            w.write_all(&i.to_le_bytes())?;
        }
    }
    w.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal parser of the files `write_ply_mesh` produces.
    fn read_back(bytes: &[u8]) -> TriangleMesh {
        let end = b"end_header\n";
        let pos = bytes.windows(end.len()).position(|w| w == end).unwrap() + end.len();
        let header = std::str::from_utf8(&bytes[..pos]).unwrap();
        let count = |name: &str| -> usize {
            header
                .lines()
                .find_map(|l| l.strip_prefix(&format!("element {name} ")))
                .unwrap()
                .parse()
                .unwrap()
        };
        let (nv, nf) = (count("vertex"), count("face"));
        let normals = header.contains("property float nx");
        let colors = header.contains("property uchar red");
        let mut b = &bytes[pos..];
        let mut f32_ = || {
            let v = f32::from_le_bytes(b[..4].try_into().unwrap());
            b = &b[4..];
            v
        };
        assert!(!colors, "colour layout is checked byte-wise");
        let mut mesh = TriangleMesh::new();
        for _ in 0..nv {
            mesh.vertices.push([f32_(), f32_(), f32_()]);
            if normals {
                mesh.normals.push([f32_(), f32_(), f32_()]);
            }
        }
        for _ in 0..nf {
            assert_eq!(b[0], 3);
            let i = |k: usize| u32::from_le_bytes(b[1 + 4 * k..5 + 4 * k].try_into().unwrap());
            mesh.triangles.push([i(0), i(1), i(2)]);
            b = &b[13..];
        }
        assert!(b.is_empty());
        mesh
    }

    fn sample_mesh() -> TriangleMesh {
        TriangleMesh {
            vertices: vec![[0.0, 0.0, 0.0], [1.5, -0.0, 2.0], [0.25, 1.0, -3.0]],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            colors: vec![],
            triangles: vec![[0, 1, 2], [2, 1, 0]],
        }
    }

    #[test]
    fn ply_mesh_round_trip() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("mesh.ply");
        let mesh = sample_mesh();
        write_ply_mesh(&path, &mesh)?;
        let back = read_back(&std::fs::read(&path)?);
        assert_eq!(back, mesh);
        // bit-exact floats, including the sign of zero
        assert_eq!(back.vertices[1][1].to_bits(), (-0.0f32).to_bits());
        Ok(())
    }

    #[test]
    fn ply_mesh_colors_layout() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("mesh.ply");
        let mut mesh = sample_mesh();
        mesh.normals.clear();
        mesh.colors = vec![[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        write_ply_mesh(&path, &mesh)?;
        let bytes = std::fs::read(&path)?;
        let end = b"end_header\n";
        let pos = bytes.windows(end.len()).position(|w| w == end).unwrap() + end.len();
        // 3 vertices x (12 B position + 3 B colour), 2 faces x 13 B
        assert_eq!(bytes.len() - pos, 3 * 15 + 2 * 13);
        assert_eq!(&bytes[pos + 12..pos + 15], &[1, 2, 3]);
        assert_eq!(&bytes[pos + 27..pos + 30], &[4, 5, 6]);
        Ok(())
    }

    #[test]
    fn ply_mesh_golden_bytes() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("mesh.ply");
        let mesh = TriangleMesh {
            vertices: vec![[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [-1.0, 0.5, 0.25]],
            normals: vec![[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
            colors: vec![[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            triangles: vec![[0, 1, 2]],
        };
        write_ply_mesh(&path, &mesh)?;
        let mut want = b"ply\nformat binary_little_endian 1.0\nelement vertex 3\n\
            property float x\nproperty float y\nproperty float z\n\
            property float nx\nproperty float ny\nproperty float nz\n\
            property uchar red\nproperty uchar green\nproperty uchar blue\n\
            element face 1\nproperty list uchar uint vertex_indices\nend_header\n"
            .to_vec();
        for i in 0..3 {
            for v in mesh.vertices[i].iter().chain(&mesh.normals[i]) {
                want.extend(v.to_le_bytes());
            }
            want.extend(mesh.colors[i]);
        }
        want.push(3);
        for i in [0u32, 1, 2] {
            want.extend(i.to_le_bytes());
        }
        assert_eq!(std::fs::read(path)?, want);
        Ok(())
    }

    #[test]
    fn ply_mesh_bytes_deterministic() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let (a, b) = (dir.path().join("a.ply"), dir.path().join("b.ply"));
        write_ply_mesh(&a, &sample_mesh())?;
        write_ply_mesh(&b, &sample_mesh())?;
        assert_eq!(std::fs::read(a)?, std::fs::read(b)?);
        Ok(())
    }

    #[test]
    fn ply_mesh_rejects_bad_index() {
        let mut mesh = sample_mesh();
        mesh.triangles.push([0, 1, 3]);
        let dir = tempfile::tempdir().unwrap();
        let r = write_ply_mesh(dir.path().join("x.ply"), &mesh);
        assert!(matches!(r, Err(PlyError::Io(e)) if e.kind() == ErrorKind::InvalidInput));
    }
}
