//! Reusable scratch for integration, meshing and raycasting.

use kornia_oxide_kernels::tsdf_core as tc;

use super::TsdfKernel;

/// Entries of the direct-mapped filter that drops repeated misses of one
/// observation before the sort (a power of two).
pub(crate) const MISS_FILTER: usize = 1024;

/// Per-observation slices of the step buffers.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ObsRange {
    /// `visible[visible.0..visible.1]`: hit pools.
    pub(crate) visible: (usize, usize),
    /// `obs_misses[misses.0..misses.1]`: keys not in the hash.
    pub(crate) misses: (usize, usize),
    /// `visible_new[new.0..new.1]`: pools allocated for the misses.
    pub(crate) new: (usize, usize),
    /// Observation mark used for de-duplication.
    pub(crate) mark: u32,
}

/// Scratch buffers reused across calls. Sized on the first (cold) calls
/// and grown geometrically; a warmed call of the same shape allocates
/// nothing.
#[derive(Debug, Default)]
pub struct TsdfWorkspace {
    kernel: TsdfKernel,
    // integration
    pub(crate) visible: Vec<u32>,
    pub(crate) visible_new: Vec<u32>,
    pub(crate) obs_misses: Vec<u64>,
    pub(crate) misses: Vec<u64>,
    pub(crate) ranges: Vec<ObsRange>,
    pub(crate) miss_filter: Vec<u64>,
    // submap steps and batches
    pub(crate) frame_order: Vec<u32>,
    pub(crate) batch_order: Vec<u32>,
    // meshing
    pub(crate) grid_f: Vec<f32>,
    pub(crate) grid_ok: Vec<bool>,
    pub(crate) grid_rgb: Vec<[u8; 3]>,
    pub(crate) edge_mark: Vec<u32>,
    pub(crate) edge_vid: Vec<u32>,
    pub(crate) edge_epoch: u32,
    pub(crate) cubes: Vec<(u32, u8)>,
    pub(crate) block_vertices: Vec<[f32; 3]>,
    pub(crate) block_normals: Vec<[f32; 3]>,
    pub(crate) block_colors: Vec<[u8; 3]>,
    pub(crate) block_triangles: Vec<[u32; 3]>,
    pub(crate) dirty: Vec<u32>,
    pub(crate) dirty_mark: Vec<u32>,
    pub(crate) dirty_epoch: u32,
    // raycast over submaps
    pub(crate) ray_params: Vec<tc::RaycastParams>,
    pub(crate) ray_slots: Vec<u32>,
}

/// Expected sizes for [`TsdfWorkspace::reserve`].
#[derive(Debug, Clone, Copy, Default)]
pub struct WorkspaceHint {
    /// Depth pixels per step (all observations).
    pub pixels_per_step: usize,
    /// Observations per step.
    pub observations_per_step: usize,
    /// Blocks touched per step.
    pub blocks: usize,
    /// Steps per [`super::SubmapTsdf::integrate_batch`] call.
    pub steps_per_batch: usize,
}

impl TsdfWorkspace {
    /// An empty workspace.
    ///
    /// # Arguments
    ///
    /// * `kernel` - CPU kernel choice; every choice gives identical bits.
    ///
    /// # Returns
    ///
    /// The workspace; buffers grow on first use.
    pub fn new(kernel: TsdfKernel) -> Self {
        Self {
            kernel,
            ..Self::default()
        }
    }

    /// The CPU kernel choice.
    pub fn kernel(&self) -> TsdfKernel {
        self.kernel
    }

    /// Pre-sizes the buffers so that the first calls of this shape do not
    /// allocate either.
    ///
    /// # Arguments
    ///
    /// * `hint` - Expected sizes.
    pub fn reserve(&mut self, hint: &WorkspaceHint) {
        let blocks = hint.blocks;
        let obs = hint.observations_per_step.max(1);
        self.visible.reserve(blocks);
        self.visible_new.reserve(blocks);
        // a cold frame misses up to every pixel's box
        self.obs_misses.reserve(blocks.max(hint.pixels_per_step));
        self.misses.reserve(blocks);
        self.ranges.reserve(obs);
        self.miss_filter.reserve(MISS_FILTER);
        self.frame_order.reserve(obs);
        self.batch_order.reserve(hint.steps_per_batch);
        self.dirty.reserve(blocks);
        self.dirty_mark.reserve(blocks);
    }

    /// Empties the block mesh buffers and sizes them for the worst block of
    /// edge `b`, so that no later block, whatever its surface, grows them.
    pub(crate) fn reserve_block_mesh(&mut self, b: usize, normals: bool, colors: bool) {
        self.cubes.clear();
        self.block_vertices.clear();
        self.block_normals.clear();
        self.block_colors.clear();
        self.block_triangles.clear();
        // at most one vertex per edge, five triangles per cube
        let vertices = 3 * (b + 1) * (b + 1) * (b + 1);
        self.cubes.reserve(b * b * b);
        self.block_vertices.reserve(vertices);
        if normals {
            self.block_normals.reserve(vertices);
        }
        if colors {
            self.block_colors.reserve(vertices);
        }
        self.block_triangles.reserve(5 * b * b * b);
    }

    /// Clears the step buffers.
    pub(crate) fn begin_step(&mut self) {
        self.visible.clear();
        self.visible_new.clear();
        self.obs_misses.clear();
        self.misses.clear();
        self.ranges.clear();
        if self.miss_filter.len() != MISS_FILTER {
            self.miss_filter.clear();
            self.miss_filter.resize(MISS_FILTER, tc::EMPTY_KEY);
        }
    }
}
