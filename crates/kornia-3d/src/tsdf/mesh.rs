//! Marching cubes per block, full and incremental (dirty-block) extraction,
//! and the zero-crossing point cloud.
//!
//! Block `K`'s mesh is a pure function of its `3x3x3` block neighbourhood:
//! cubes with their min corner in `K`, one vertex per crossed edge they
//! use (face vertices are repeated in the neighbouring block, as nvblox
//! does), triangles in cube order. Full extraction concatenates blocks in
//! ascending key order, so it equals an incremental cache rebuilt from
//! empty, byte for byte.

use kornia_oxide_kernels::tsdf_core as tc;

use super::volume::VolumeRef;
use super::workspace::TsdfWorkspace;
use super::{check_fp_env, BlockDim, StdOps, SupportedBlockDim, TsdfError, TsdfVolume};
use crate::mesh::TriangleMesh;

/// Meshing parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshParams {
    /// Smallest weight of a valid voxel, `>= 1` (a cube is meshed only when
    /// its 8 corners are valid).
    pub weight_threshold: u16,
    /// Output per-vertex normals (the interpolated TSDF gradient, pointing
    /// into free space).
    pub normals: bool,
    /// Output per-vertex colours (needs colour storage).
    pub colors: bool,
}

impl Default for MeshParams {
    fn default() -> Self {
        Self {
            weight_threshold: 1,
            normals: true,
            colors: false,
        }
    }
}

impl MeshParams {
    pub(crate) fn validate(&self, has_color: bool) -> Result<(), TsdfError> {
        if self.weight_threshold == 0 {
            return Err(TsdfError::InvalidParameter {
                field: "weight_threshold",
                value: 0.0,
            });
        }
        if self.colors && !has_color {
            return Err(TsdfError::InvalidParameter {
                field: "colors",
                value: 1.0,
            });
        }
        Ok(())
    }
}

/// Counters of [`TsdfVolume::update_mesh_cache`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MeshUpdateStats {
    /// Blocks re-meshed by this update.
    pub remeshed_blocks: u32,
    /// Vertices in the cache after the update.
    pub vertices: u64,
    /// Triangles in the cache after the update.
    pub triangles: u64,
}

impl MeshUpdateStats {
    pub(crate) fn accumulate(&mut self, o: &MeshUpdateStats) {
        self.remeshed_blocks = self.remeshed_blocks.saturating_add(o.remeshed_blocks);
        self.vertices += o.vertices;
        self.triangles += o.triangles;
    }
}

/// Block offsets whose meshes read a block: `{-1, 0, 1}^3` with normals
/// (gradients read `±1`), `{-1, 0}^3` without.
fn affected_offsets(normals: bool) -> &'static [[i32; 3]] {
    const NB27: [[i32; 3]; 27] = {
        let mut o = [[0i32; 3]; 27];
        let mut i = 0;
        while i < 27 {
            o[i] = [
                (i % 3) as i32 - 1,
                ((i / 3) % 3) as i32 - 1,
                (i / 9) as i32 - 1,
            ];
            i += 1;
        }
        o
    };
    const NB8: [[i32; 3]; 8] = {
        let mut o = [[0i32; 3]; 8];
        let mut i = 0;
        while i < 8 {
            o[i] = [
                -((i & 1) as i32),
                -(((i >> 1) & 1) as i32),
                -((i >> 2) as i32),
            ];
            i += 1;
        }
        o
    };
    if normals {
        &NB27
    } else {
        &NB8
    }
}

/// Loads the voxels `[-1, B + 1]^3` around block `bc` into the workspace
/// grid (value, validity, colour).
fn load_neighbourhood(
    view: &VolumeRef<'_>,
    bc: [i32; 3],
    threshold: u16,
    colors: bool,
    ws: &mut TsdfWorkspace,
) {
    let l = view.grid.log2;
    let b = 1i32 << l;
    let n = (b + 3) as usize;
    let mut bases = [None; 27];
    for (i, base) in bases.iter_mut().enumerate() {
        let d = [
            (i % 3) as i32 - 1,
            ((i / 3) % 3) as i32 - 1,
            (i / 9) as i32 - 1,
        ];
        *base = tc::neighbour_key(bc, d).and_then(|k| view.block_base(k));
    }
    let len = n * n * n;
    if ws.grid_f.len() < len {
        ws.grid_f.resize(len, 0.0);
        ws.grid_ok.resize(len, false);
        ws.grid_rgb.resize(len, [0; 3]);
    }
    let mask = b - 1;
    let mut gi = 0;
    for z in -1..=b + 1 {
        for y in -1..=b + 1 {
            for x in -1..=b + 1 {
                let ni = ((x >> l) + 1) + 3 * (((y >> l) + 1) + 3 * ((z >> l) + 1));
                let slot = bases[ni as usize]
                    .map(|base| base + tc::local_index([x & mask, y & mask, z & mask], l));
                let valid = match slot {
                    Some(i) if view.pools.weight[i] >= threshold => {
                        ws.grid_f[gi] = view.pools.tsdf[i];
                        if colors {
                            ws.grid_rgb[gi] = tc::unpack_rgb(view.pools.rgbw[i]);
                        }
                        true
                    }
                    _ => false,
                };
                ws.grid_ok[gi] = valid;
                gi += 1;
            }
        }
    }
}

/// Index into the loaded grid of local voxel `v` in `[-1, B + 1]^3`.
#[inline(always)]
fn gidx(v: [i32; 3], n: i32) -> usize {
    ((v[0] + 1) + n * ((v[1] + 1) + n * (v[2] + 1))) as usize
}

fn axis(k: u32) -> [i32; 3] {
    let mut e = [0; 3];
    e[k as usize] = 1;
    e
}

fn add(a: [i32; 3], b: [i32; 3]) -> [i32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// The loaded grid of one block as the shared core's mesher reads it.
struct LoadedGrid<'a> {
    ws: &'a TsdfWorkspace,
    n: i32,
}

impl tc::MeshVoxels for LoadedGrid<'_> {
    #[inline(always)]
    fn value(&self, v: [i32; 3]) -> Option<f32> {
        let i = gidx(v, self.n);
        self.ws.grid_ok[i].then(|| self.ws.grid_f[i])
    }

    #[inline(always)]
    fn rgb(&self, v: [i32; 3]) -> [u8; 3] {
        self.ws.grid_rgb[gidx(v, self.n)]
    }
}

/// Edge vertex (position, colour, normal) from owner `a` along `k`.
fn edge_point(
    view: &VolumeRef<'_>,
    ws: &TsdfWorkspace,
    g0: [i32; 3],
    a: [i32; 3],
    k: u32,
    normals: bool,
    colors: bool,
) -> ([f32; 3], [u8; 3], [f32; 3]) {
    let grid = LoadedGrid {
        ws,
        n: (1i32 << view.grid.log2) + 3,
    };
    let g = &view.grid;
    tc::mesh_edge_point::<StdOps, _>(
        &grid,
        g0,
        a,
        k as usize,
        g.voxel_size,
        g.half_voxel,
        normals,
        colors,
    )
}

/// Meshes block `key` into the workspace's block buffers (block-local
/// vertex indices).
pub(crate) fn mesh_block(
    view: &VolumeRef<'_>,
    key: u64,
    params: &MeshParams,
    ws: &mut TsdfWorkspace,
) {
    let l = view.grid.log2;
    let b = 1i32 << l;
    let n = b + 3;
    let bc = tc::unpack_block_key(key);
    load_neighbourhood(view, bc, params.weight_threshold, params.colors, ws);

    let edges = (3 * (b + 1) * (b + 1) * (b + 1)) as usize;
    if ws.edge_mark.len() < edges {
        ws.edge_mark.resize(edges, 0);
        ws.edge_vid.resize(edges, 0);
    }
    if ws.edge_epoch == u32::MAX {
        ws.edge_mark.fill(0);
        ws.edge_epoch = 0;
    }
    ws.edge_epoch += 1;
    let epoch = ws.edge_epoch;
    ws.reserve_block_mesh(b as usize, params.normals, params.colors);

    // cubes in ascending lx + B * (ly + B * lz); mark the edges they use
    ws.cubes.clear();
    let mut cube = 0u32;
    for lz in 0..b {
        for ly in 0..b {
            for lx in 0..b {
                let l0 = [lx, ly, lz];
                let mut f = [0f32; 8];
                let mut all = true;
                for (c, off) in tc::CORNER.iter().enumerate() {
                    let i = gidx(add(l0, *off), n);
                    if !ws.grid_ok[i] {
                        all = false;
                        break;
                    }
                    f[c] = ws.grid_f[i];
                }
                if all {
                    let ci = tc::cube_index(&f);
                    let et = tc::EDGE_TABLE[ci];
                    if et != 0 {
                        for k in 0..12 {
                            if et & (1 << k) != 0 {
                                let o = add(l0, tc::EDGE_OWNER[k]);
                                ws.edge_mark[tc::edge_key(o, tc::EDGE_AXIS[k], b)] = epoch;
                            }
                        }
                        ws.cubes.push((cube, ci as u8));
                    }
                }
                cube += 1;
            }
        }
    }

    // one vertex per used edge, ascending edge index
    ws.block_vertices.clear();
    ws.block_normals.clear();
    ws.block_colors.clear();
    ws.block_triangles.clear();
    let g0 = [bc[0] * b, bc[1] * b, bc[2] * b];
    for oz in 0..=b {
        for oy in 0..=b {
            for ox in 0..=b {
                for k in 0..3u32 {
                    let o = [ox, oy, oz];
                    let e = tc::edge_key(o, k, b);
                    if ws.edge_mark[e] != epoch {
                        continue;
                    }
                    let (p, c, nrm) = edge_point(view, ws, g0, o, k, params.normals, params.colors);
                    ws.edge_vid[e] = ws.block_vertices.len() as u32;
                    ws.block_vertices.push(p);
                    if params.colors {
                        ws.block_colors.push(c);
                    }
                    if params.normals {
                        ws.block_normals.push(nrm);
                    }
                }
            }
        }
    }

    // triangles in cube order; Bourke's triples are wound clockwise seen
    // from free space, so emit (i0, i2, i1)
    for ci in 0..ws.cubes.len() {
        let (lin, conf) = ws.cubes[ci];
        let lin = lin as i32;
        let l0 = [lin & (b - 1), (lin >> l) & (b - 1), lin >> (2 * l)];
        let row = &tc::TRI_TABLE[conf as usize];
        let mut j = 0;
        while j + 2 < 16 && row[j] >= 0 {
            let vid = |ek: i8| {
                let ek = ek as usize;
                let o = add(l0, tc::EDGE_OWNER[ek]);
                ws.edge_vid[tc::edge_key(o, tc::EDGE_AXIS[ek], b)]
            };
            let tri = [vid(row[j]), vid(row[j + 2]), vid(row[j + 1])];
            ws.block_triangles.push(tri);
            j += 3;
        }
    }
}

/// Appends the workspace's block mesh to `mesh`, mapped by `xf`
/// (`v = R v + t`, `n = R n`) when given.
pub(crate) fn append_block(
    mesh: &mut TriangleMesh,
    ws: &TsdfWorkspace,
    xf: Option<&tc::Rigid>,
) -> Result<(), TsdfError> {
    append_parts(
        mesh,
        &ws.block_vertices,
        &ws.block_normals,
        &ws.block_colors,
        &ws.block_triangles,
        xf,
    )
}

fn append_parts(
    mesh: &mut TriangleMesh,
    vertices: &[[f32; 3]],
    normals: &[[f32; 3]],
    colors: &[[u8; 3]],
    triangles: &[[u32; 3]],
    xf: Option<&tc::Rigid>,
) -> Result<(), TsdfError> {
    let base = mesh.vertices.len();
    let total = base + vertices.len();
    if total > u32::MAX as usize {
        return Err(TsdfError::MeshTooLarge { vertices: total });
    }
    match xf {
        None => {
            mesh.vertices.extend_from_slice(vertices);
            mesh.normals.extend_from_slice(normals);
        }
        Some(x) => {
            mesh.vertices
                .extend(vertices.iter().map(|&v| tc::xform::<StdOps>(&x.r, &x.t, v)));
            mesh.normals
                .extend(normals.iter().map(|&v| tc::rot::<StdOps>(&x.r, v)));
        }
    }
    mesh.colors.extend_from_slice(colors);
    let b = base as u32;
    mesh.triangles
        .extend(triangles.iter().map(|t| [t[0] + b, t[1] + b, t[2] + b]));
    Ok(())
}

/// Full extraction of one volume, blocks in ascending key order.
pub(crate) fn extract_view(
    view: &VolumeRef<'_>,
    params: &MeshParams,
    ws: &mut TsdfWorkspace,
    mesh: &mut TriangleMesh,
    xf: Option<&tc::Rigid>,
) -> Result<(), TsdfError> {
    for &(key, _) in &view.index.sorted {
        mesh_block(view, key, params, ws);
        append_block(mesh, ws, xf)?;
    }
    Ok(())
}

/// Zero-crossing points of the edges each block owns (`o` in `[0, B)^3`),
/// blocks in ascending key, then ascending edge index.
pub(crate) fn point_cloud_view(
    view: &VolumeRef<'_>,
    params: &MeshParams,
    ws: &mut TsdfWorkspace,
    points: &mut Vec<[f32; 3]>,
    mut colors: Option<&mut Vec<[u8; 3]>>,
    mut normals: Option<&mut Vec<[f32; 3]>>,
) {
    let l = view.grid.log2;
    let b = 1i32 << l;
    let n = b + 3;
    for &(key, _) in &view.index.sorted {
        let bc = tc::unpack_block_key(key);
        let g0 = [bc[0] * b, bc[1] * b, bc[2] * b];
        load_neighbourhood(view, bc, params.weight_threshold, params.colors, ws);
        for oz in 0..b {
            for oy in 0..b {
                for ox in 0..b {
                    for k in 0..3u32 {
                        let a = [ox, oy, oz];
                        let (ia, ib) = (gidx(a, n), gidx(add(a, axis(k)), n));
                        if !(ws.grid_ok[ia] && ws.grid_ok[ib]) {
                            continue;
                        }
                        if (ws.grid_f[ia] < 0.0) == (ws.grid_f[ib] < 0.0) {
                            continue;
                        }
                        let want_n = normals.is_some();
                        let want_c = colors.is_some();
                        let (p, c, nrm) = edge_point(view, ws, g0, a, k, want_n, want_c);
                        points.push(p);
                        if let Some(cs) = colors.as_deref_mut() {
                            cs.push(c);
                        }
                        if let Some(ns) = normals.as_deref_mut() {
                            ns.push(nrm);
                        }
                    }
                }
            }
        }
    }
}

/// Offsets of one block's mesh in the cache slab.
#[derive(Debug, Clone, Copy, Default)]
struct Slot {
    v_off: usize,
    v_len: usize,
    v_cap: usize,
    t_off: usize,
    t_len: usize,
    t_cap: usize,
}

/// Slack below which the slab is never compacted.
const COMPACT_SLACK: usize = 1 << 16;

/// Per-block mesh cache for incremental meshing: each block's mesh lives
/// at an `(offset, len)` in shared arrays; a re-meshed block that still
/// fits stays in place, otherwise it is appended (compaction is a cold
/// operation). Use one cache per volume.
#[derive(Debug, Clone, Default)]
pub struct BlockMeshCache {
    vertices: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    colors: Vec<[u8; 3]>,
    triangles: Vec<[u32; 3]>,
    slots: Vec<Slot>,
    order: Vec<(u64, u32)>,
    epoch: u64,
    params: Option<MeshParams>,
    live_vertices: usize,
    live_triangles: usize,
}

impl BlockMeshCache {
    /// An empty cache.
    ///
    /// # Returns
    ///
    /// The cache; buffers grow on first use.
    pub fn new() -> Self {
        Self::default()
    }

    /// An empty cache with reserved slab space.
    ///
    /// # Arguments
    ///
    /// * `vertices` - Vertex capacity.
    /// * `triangles` - Triangle capacity.
    /// * `blocks` - Block capacity.
    ///
    /// # Returns
    ///
    /// The cache.
    pub fn with_capacity(vertices: usize, triangles: usize, blocks: usize) -> Self {
        Self {
            vertices: Vec::with_capacity(vertices),
            normals: Vec::with_capacity(vertices),
            colors: Vec::with_capacity(vertices),
            triangles: Vec::with_capacity(triangles),
            slots: Vec::with_capacity(blocks),
            order: Vec::with_capacity(blocks),
            ..Self::default()
        }
    }

    /// Forgets the content (the next update rebuilds it), keeping the slab.
    pub(crate) fn invalidate(&mut self) {
        self.params = None;
    }

    /// Starts an update from a volume of `len` blocks at step `epoch`:
    /// resets the cache on new params or an older volume, then sizes the
    /// slots. Returns the cache's epoch and block count before the update
    /// (blocks modified after that epoch, or at or past that count, are
    /// dirty).
    pub(crate) fn begin_update(
        &mut self,
        params: &MeshParams,
        epoch: u64,
        len: usize,
    ) -> (u64, usize) {
        if self.params != Some(*params) || self.epoch > epoch || self.slots.len() > len {
            self.reset(*params);
        }
        let before = (self.epoch, self.slots.len());
        self.slots.resize(len, Slot::default());
        before
    }

    /// Ends an update: block order from the volume's ascending `(key,
    /// pool)` list, compaction if the slab is mostly garbage, the epoch.
    pub(crate) fn finish_update(
        &mut self,
        sorted: &[(u64, u32)],
        epoch: u64,
        remeshed: u32,
    ) -> MeshUpdateStats {
        self.order.clear();
        self.order.extend_from_slice(sorted);
        let garbage =
            self.vertices.len() - self.live_vertices + self.triangles.len() - self.live_triangles;
        if garbage > COMPACT_SLACK && garbage > self.live_vertices + self.live_triangles {
            self.compact();
        }
        self.epoch = epoch;
        MeshUpdateStats {
            remeshed_blocks: remeshed,
            vertices: self.live_vertices as u64,
            triangles: self.live_triangles as u64,
        }
    }

    fn reset(&mut self, params: MeshParams) {
        self.vertices.clear();
        self.normals.clear();
        self.colors.clear();
        self.triangles.clear();
        self.slots.clear();
        self.order.clear();
        self.epoch = 0;
        self.params = Some(params);
        self.live_vertices = 0;
        self.live_triangles = 0;
    }

    /// Stores the workspace's block mesh as block `p`'s.
    fn store(&mut self, p: usize, ws: &TsdfWorkspace, params: &MeshParams) {
        self.store_parts(
            p,
            &ws.block_vertices,
            &ws.block_normals,
            &ws.block_colors,
            &ws.block_triangles,
            params,
        );
    }

    /// Stores one block's mesh (block-local triangle indices) as block
    /// `p`'s; `normals`/`colors` are empty unless `params` asks for them.
    pub(crate) fn store_parts(
        &mut self,
        p: usize,
        vertices: &[[f32; 3]],
        normals: &[[f32; 3]],
        colors: &[[u8; 3]],
        triangles: &[[u32; 3]],
        params: &MeshParams,
    ) {
        let s = &mut self.slots[p];
        let nv = vertices.len();
        let nt = triangles.len();
        if nv > s.v_cap {
            s.v_off = self.vertices.len();
            s.v_cap = nv;
            let end = s.v_off + nv;
            self.vertices.resize(end, [0.0; 3]);
            if params.normals {
                self.normals.resize(end, [0.0; 3]);
            }
            if params.colors {
                self.colors.resize(end, [0; 3]);
            }
        }
        if nt > s.t_cap {
            s.t_off = self.triangles.len();
            s.t_cap = nt;
            self.triangles.resize(s.t_off + nt, [0; 3]);
        }
        let vr = s.v_off..s.v_off + nv;
        self.vertices[vr.clone()].copy_from_slice(vertices);
        if params.normals {
            self.normals[vr.clone()].copy_from_slice(normals);
        }
        if params.colors {
            self.colors[vr].copy_from_slice(colors);
        }
        self.triangles[s.t_off..s.t_off + nt].copy_from_slice(triangles);
        self.live_vertices = self.live_vertices + nv - s.v_len;
        self.live_triangles = self.live_triangles + nt - s.t_len;
        s.v_len = nv;
        s.t_len = nt;
    }

    /// Repacks the live meshes densely (allocates; cold). The new buffers
    /// keep the old capacity, so appends after a compaction stay in the slab.
    fn compact(&mut self) {
        let mut v = Vec::with_capacity(self.vertices.capacity());
        let mut nrm = Vec::with_capacity(self.normals.capacity());
        let mut col = Vec::with_capacity(self.colors.capacity());
        let mut t = Vec::with_capacity(self.triangles.capacity());
        for &(_, p) in &self.order {
            let s = &mut self.slots[p as usize];
            let off = v.len();
            v.extend_from_slice(&self.vertices[s.v_off..s.v_off + s.v_len]);
            if !self.normals.is_empty() {
                nrm.extend_from_slice(&self.normals[s.v_off..s.v_off + s.v_len]);
            }
            if !self.colors.is_empty() {
                col.extend_from_slice(&self.colors[s.v_off..s.v_off + s.v_len]);
            }
            let toff = t.len();
            t.extend_from_slice(&self.triangles[s.t_off..s.t_off + s.t_len]);
            *s = Slot {
                v_off: off,
                v_len: s.v_len,
                v_cap: s.v_len,
                t_off: toff,
                t_len: s.t_len,
                t_cap: s.t_len,
            };
        }
        self.vertices = v;
        self.normals = nrm;
        self.colors = col;
        self.triangles = t;
    }

    /// Writes the cached mesh, blocks in ascending key order.
    ///
    /// # Arguments
    ///
    /// * `mesh` - Output mesh (cleared first; its capacity is reused).
    ///
    /// # Returns
    ///
    /// `Ok(())` once `mesh` holds the cached mesh.
    ///
    /// # Errors
    ///
    /// [`TsdfError::MeshTooLarge`] beyond `u32` vertex indices.
    pub fn write_into(&self, mesh: &mut TriangleMesh) -> Result<(), TsdfError> {
        mesh.clear();
        self.append_to(mesh, None)
    }

    pub(crate) fn append_to(
        &self,
        mesh: &mut TriangleMesh,
        xf: Option<&tc::Rigid>,
    ) -> Result<(), TsdfError> {
        for &(_, p) in &self.order {
            let s = &self.slots[p as usize];
            let vr = s.v_off..s.v_off + s.v_len;
            let normals = if self.normals.is_empty() {
                &[][..]
            } else {
                &self.normals[vr.clone()]
            };
            let colors = if self.colors.is_empty() {
                &[][..]
            } else {
                &self.colors[vr.clone()]
            };
            append_parts(
                mesh,
                &self.vertices[vr],
                normals,
                colors,
                &self.triangles[s.t_off..s.t_off + s.t_len],
                xf,
            )?;
        }
        Ok(())
    }
}

/// Re-meshes the blocks of `view` modified since `cache`'s last update and
/// their affected neighbours.
pub(crate) fn update_cache(
    view: &VolumeRef<'_>,
    ws: &mut TsdfWorkspace,
    params: &MeshParams,
    cache: &mut BlockMeshCache,
) -> MeshUpdateStats {
    let index = view.index;
    let (cache_epoch, cache_len) = cache.begin_update(params, index.epoch, index.len());
    ws.dirty.clear();
    if ws.dirty_mark.len() < index.len() {
        ws.dirty_mark.resize(index.len(), 0);
    }
    if ws.dirty_epoch == u32::MAX {
        ws.dirty_mark.fill(0);
        ws.dirty_epoch = 0;
    }
    ws.dirty_epoch += 1;
    let mark = ws.dirty_epoch;
    let offsets = affected_offsets(params.normals);
    for &(key, p) in &index.sorted {
        if index.modified[p as usize] <= cache_epoch && (p as usize) < cache_len {
            continue;
        }
        let bc = tc::unpack_block_key(key);
        for d in offsets {
            let Some(q) = tc::neighbour_key(bc, *d).and_then(|k| index.lookup(k)) else {
                continue;
            };
            let m = &mut ws.dirty_mark[q as usize];
            if *m != mark {
                *m = mark;
                ws.dirty.push(q);
            }
        }
    }
    for i in 0..ws.dirty.len() {
        let q = ws.dirty[i] as usize;
        mesh_block(view, index.pool_keys[q], params, ws);
        cache.store(q, ws, params);
    }
    cache.finish_update(&index.sorted, index.epoch, ws.dirty.len() as u32)
}

impl<const B: usize> TsdfVolume<B>
where
    BlockDim<B>: SupportedBlockDim,
{
    /// Extracts the full triangle mesh (marching cubes per block, blocks in
    /// ascending key order, counter-clockwise seen from free space).
    ///
    /// # Arguments
    ///
    /// * `ws` - Reused scratch.
    /// * `params` - Weight threshold, normals, colours.
    /// * `mesh` - Output mesh (cleared first; its capacity is reused).
    ///
    /// # Returns
    ///
    /// `Ok(())` once `mesh` holds the surface.
    ///
    /// # Errors
    ///
    /// [`TsdfError::InvalidParameter`] (threshold 0, colours without colour
    /// storage), [`TsdfError::MeshTooLarge`], [`TsdfError::FpEnvironment`].
    pub fn extract_mesh_into(
        &self,
        ws: &mut TsdfWorkspace,
        params: &MeshParams,
        mesh: &mut TriangleMesh,
    ) -> Result<(), TsdfError> {
        check_fp_env()?;
        params.validate(self.pools.color)?;
        mesh.clear();
        extract_view(&self.view(), params, ws, mesh, None)
    }

    /// Re-meshes only the blocks whose mesh may have changed since the
    /// cache's last update (blocks modified since, and their neighbours);
    /// [`BlockMeshCache::write_into`] then gives the same bytes as
    /// [`TsdfVolume::extract_mesh_into`]. Changing `params` rebuilds the
    /// cache.
    ///
    /// # Arguments
    ///
    /// * `ws` - Reused scratch.
    /// * `params` - Weight threshold, normals, colours.
    /// * `cache` - This volume's cache.
    ///
    /// # Returns
    ///
    /// The update's counters.
    ///
    /// # Errors
    ///
    /// As [`TsdfVolume::extract_mesh_into`].
    pub fn update_mesh_cache(
        &self,
        ws: &mut TsdfWorkspace,
        params: &MeshParams,
        cache: &mut BlockMeshCache,
    ) -> Result<MeshUpdateStats, TsdfError> {
        check_fp_env()?;
        params.validate(self.pools.color)?;
        Ok(update_cache(&self.view(), ws, params, cache))
    }

    /// Extracts the zero-crossing points: one per sign-changing edge
    /// between valid voxels, each edge counted once (no duplicates),
    /// positions as the mesh vertices.
    ///
    /// # Arguments
    ///
    /// * `ws` - Reused scratch.
    /// * `params` - Weight threshold (`normals`/`colors` are taken from the
    ///   output options).
    /// * `points` - Output positions (cleared first).
    /// * `colors` - Optional output colours (needs colour storage).
    /// * `normals` - Optional output normals.
    ///
    /// # Returns
    ///
    /// `Ok(())` once the outputs hold the points.
    ///
    /// # Errors
    ///
    /// [`TsdfError::InvalidParameter`], [`TsdfError::FpEnvironment`].
    pub fn extract_point_cloud_into(
        &self,
        ws: &mut TsdfWorkspace,
        params: &MeshParams,
        points: &mut Vec<[f32; 3]>,
        mut colors: Option<&mut Vec<[u8; 3]>>,
        mut normals: Option<&mut Vec<[f32; 3]>>,
    ) -> Result<(), TsdfError> {
        check_fp_env()?;
        let p = MeshParams {
            weight_threshold: params.weight_threshold,
            normals: normals.is_some(),
            colors: colors.is_some(),
        };
        p.validate(self.pools.color)?;
        points.clear();
        if let Some(c) = colors.as_deref_mut() {
            c.clear();
        }
        if let Some(n) = normals.as_deref_mut() {
            n.clear();
        }
        point_cloud_view(&self.view(), &p, ws, points, colors, normals);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tsdf_compact_keeps_the_slab_capacity() {
        let mut c = BlockMeshCache::with_capacity(64, 64, 2);
        // a block meshed at offset 10 (rows 0..10 are garbage), 3 vertices and 1 triangle at 5
        c.vertices = (0..13).map(|i| [i as f32; 3]).collect();
        c.normals = c.vertices.clone();
        c.triangles = (0..6).map(|i| [i; 3]).collect();
        c.slots = vec![Slot {
            v_off: 10,
            v_len: 3,
            v_cap: 3,
            t_off: 5,
            t_len: 1,
            t_cap: 1,
        }];
        c.order = vec![(0, 0)];
        (c.live_vertices, c.live_triangles) = (3, 1);
        let (cv, ct) = (c.vertices.capacity(), c.triangles.capacity());
        c.compact();
        assert_eq!(c.vertices, [[10.0; 3], [11.0; 3], [12.0; 3]]);
        assert_eq!(c.normals, c.vertices);
        assert_eq!(c.triangles, [[5; 3]]);
        assert_eq!((c.slots[0].v_off, c.slots[0].t_off), (0, 0));
        assert!(c.vertices.capacity() >= cv && c.normals.capacity() >= cv);
        assert!(c.triangles.capacity() >= ct);
    }
}
