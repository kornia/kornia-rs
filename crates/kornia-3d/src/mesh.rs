//! Indexed triangle meshes.

/// An indexed triangle mesh with optional per-vertex normals and colours.
///
/// `normals` and `colors` are either empty or one entry per vertex;
/// `triangles` index `vertices`. Producers write into a caller-owned mesh
/// (e.g. [`crate::tsdf::TsdfVolume::extract_mesh_into`]) so its buffers are
/// reused across calls.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TriangleMesh {
    /// Vertex positions.
    pub vertices: Vec<[f32; 3]>,
    /// Unit vertex normals, empty when absent.
    pub normals: Vec<[f32; 3]>,
    /// Vertex colours (RGB), empty when absent.
    pub colors: Vec<[u8; 3]>,
    /// Vertex index triples.
    pub triangles: Vec<[u32; 3]>,
}

impl TriangleMesh {
    /// An empty mesh.
    ///
    /// # Returns
    ///
    /// A mesh with no vertices or triangles.
    pub fn new() -> Self {
        Self::default()
    }

    /// Empties the mesh, keeping its capacity.
    pub fn clear(&mut self) {
        self.vertices.clear();
        self.normals.clear();
        self.colors.clear();
        self.triangles.clear();
    }
}
