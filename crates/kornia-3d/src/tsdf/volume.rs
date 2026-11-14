//! Block storage: the per-volume block index (hash, sorted keys, page
//! table, epochs) and the SoA voxel pools, plus [`TsdfVolume`].

use std::marker::PhantomData;

use kornia_oxide_kernels::tsdf_core as tc;

use super::{
    max_blocks, BlockDim, BlockKey, PoolGrowth, SupportedBlockDim, TsdfError, TsdfLayout,
    TsdfParams,
};

/// Fewest hash slots (`hash_slot` needs `log2_cap >= 1`).
const MIN_HASH_SLOTS: usize = 2;

/// Hash slots for `capacity` blocks: a power of two `>= 2 * capacity`.
fn hash_slots(capacity: usize) -> usize {
    (2 * capacity).next_power_of_two().max(MIN_HASH_SLOTS)
}

/// The block index of one volume (or submap): which keys exist and where
/// their voxels live. Local pool indices are dense, append-only and
/// assigned in ascending key order per step.
#[derive(Debug, Clone)]
pub(crate) struct BlockIndex {
    /// Hash keys, `EMPTY_KEY` = free slot.
    pub(crate) keys: Vec<u64>,
    /// Hash values: local pool index.
    pub(crate) vals: Vec<u32>,
    /// `log2(keys.len())`.
    pub(crate) log2_cap: u32,
    /// `(key, local pool)`, ascending key.
    pub(crate) sorted: Vec<(u64, u32)>,
    /// Local pool -> key.
    pub(crate) pool_keys: Vec<u64>,
    /// Local pool -> pool page.
    pub(crate) pages: Vec<u32>,
    /// Local pool -> step epoch of its last touch (0 = never).
    pub(crate) modified: Vec<u64>,
    /// Local pool -> observation mark (scratch for de-duplication).
    pub(crate) mark: Vec<u32>,
    /// Current mark value.
    pub(crate) mark_epoch: u32,
    /// Local pool capacity.
    pub(crate) capacity: usize,
    /// Steps applied (the epoch of the latest one).
    pub(crate) epoch: u64,
}

impl BlockIndex {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        let slots = hash_slots(capacity);
        Self {
            keys: vec![tc::EMPTY_KEY; slots],
            vals: vec![0; slots],
            log2_cap: slots.trailing_zeros(),
            sorted: Vec::with_capacity(capacity),
            pool_keys: Vec::with_capacity(capacity),
            pages: Vec::with_capacity(capacity),
            modified: Vec::with_capacity(capacity),
            mark: Vec::with_capacity(capacity),
            mark_epoch: 0,
            capacity,
            epoch: 0,
        }
    }

    /// Empties the index, keeping its buffers.
    pub(crate) fn clear(&mut self) {
        self.keys.fill(tc::EMPTY_KEY);
        self.sorted.clear();
        self.pool_keys.clear();
        self.pages.clear();
        self.modified.clear();
        self.mark.clear();
        self.mark_epoch = 0;
        self.epoch = 0;
    }

    /// Allocated blocks.
    pub(crate) fn len(&self) -> usize {
        self.pool_keys.len()
    }

    #[inline(always)]
    pub(crate) fn lookup(&self, key: u64) -> Option<u32> {
        tc::hash_lookup(&self.keys, &self.vals, self.log2_cap, key)
    }

    /// A fresh observation mark (resets the marks on wrap).
    pub(crate) fn next_mark(&mut self) -> u32 {
        if self.mark_epoch == u32::MAX {
            self.mark.fill(0);
            self.mark_epoch = 0;
        }
        self.mark_epoch += 1;
        self.mark_epoch
    }

    /// Raises the capacity to `capacity` and rebuilds the hash (cold).
    pub(crate) fn grow(&mut self, capacity: usize) {
        if capacity <= self.capacity {
            return;
        }
        let extra = capacity - self.len();
        self.sorted.reserve_exact(extra);
        self.pool_keys.reserve_exact(extra);
        self.pages.reserve_exact(extra);
        self.modified.reserve_exact(extra);
        self.mark.reserve_exact(extra);
        let slots = hash_slots(capacity);
        if slots > self.keys.len() {
            self.keys = vec![tc::EMPTY_KEY; slots];
            self.vals = vec![0; slots];
            self.log2_cap = slots.trailing_zeros();
            for (p, &k) in self.pool_keys.iter().enumerate() {
                tc::hash_insert(&mut self.keys, &mut self.vals, self.log2_cap, k, p as u32);
            }
        }
        self.capacity = capacity;
    }

    /// Host bytes held.
    pub(crate) fn bytes(&self) -> usize {
        self.keys.capacity() * 8
            + self.vals.capacity() * 4
            + self.sorted.capacity() * std::mem::size_of::<(u64, u32)>()
            + self.pool_keys.capacity() * 8
            + self.pages.capacity() * 4
            + self.modified.capacity() * 8
            + self.mark.capacity() * 4
    }
}

/// SoA voxel pools, block-major: voxel `i` of page `p` is at `p * B³ + i`.
#[derive(Debug, Clone)]
pub(crate) struct Pools {
    pub(crate) tsdf: Vec<f32>,
    pub(crate) weight: Vec<u16>,
    /// `r | g << 8 | b << 16 | color_weight << 24`; empty without colour.
    pub(crate) rgbw: Vec<u32>,
    /// `B³`.
    pub(crate) voxels: usize,
    pub(crate) color: bool,
    /// Page capacity.
    pub(crate) capacity: usize,
}

impl Pools {
    pub(crate) fn with_capacity(capacity: usize, voxels: usize, color: bool) -> Self {
        let n = capacity * voxels;
        Self {
            tsdf: Vec::with_capacity(n),
            weight: Vec::with_capacity(n),
            rgbw: Vec::with_capacity(if color { n } else { 0 }),
            voxels,
            color,
            capacity,
        }
    }

    pub(crate) fn pages(&self) -> usize {
        self.weight.len() / self.voxels
    }

    /// Appends `m` zeroed pages (within capacity); returns the first.
    pub(crate) fn alloc(&mut self, m: usize) -> u32 {
        let first = self.pages();
        let n = (first + m) * self.voxels;
        self.tsdf.resize(n, 0.0);
        self.weight.resize(n, 0);
        if self.color {
            self.rgbw.resize(n, 0);
        }
        first as u32
    }

    /// Raises the page capacity (cold).
    pub(crate) fn grow(&mut self, capacity: usize) {
        if capacity <= self.capacity {
            return;
        }
        let extra = (capacity - self.pages()) * self.voxels;
        self.tsdf.reserve_exact(extra);
        self.weight.reserve_exact(extra);
        if self.color {
            self.rgbw.reserve_exact(extra);
        }
        self.capacity = capacity;
    }

    /// Bytes of `pages` pages.
    pub(crate) fn page_bytes(&self, pages: usize) -> usize {
        pages * self.voxels * if self.color { 10 } else { 6 }
    }

    pub(crate) fn bytes(&self) -> usize {
        self.tsdf.capacity() * 4 + self.weight.capacity() * 2 + self.rgbw.capacity() * 4
    }
}

/// Grows `index` and `pools` for `m` more blocks, or fails without
/// mutating anything observable.
pub(crate) fn ensure_capacity(
    index: &mut BlockIndex,
    pools: &mut Pools,
    growth: PoolGrowth,
    m: usize,
) -> Result<(), TsdfError> {
    let limit = max_blocks(pools.voxels);
    let need_index = index.len() + m;
    let need_pages = pools.pages() + m;
    let fits = need_index <= index.capacity && need_pages <= pools.capacity;
    if fits {
        return Ok(());
    }
    if growth == PoolGrowth::Fixed || need_pages > limit {
        let (capacity, needed) = if need_index > index.capacity {
            (index.capacity, need_index)
        } else {
            (pools.capacity, need_pages)
        };
        return Err(TsdfError::PoolFull { capacity, needed });
    }
    if need_index > index.capacity {
        index.grow((2 * index.capacity).max(need_index).min(limit));
    }
    if need_pages > pools.capacity {
        pools.grow((2 * pools.capacity).max(need_pages).min(limit));
    }
    Ok(())
}

/// Appends the sorted, unique, absent `misses` to `index` (pool indices in
/// ascending key order) with zeroed pages; capacity must be ensured.
pub(crate) fn insert_blocks(index: &mut BlockIndex, pools: &mut Pools, misses: &[u64]) {
    let n = index.len();
    let m = misses.len();
    if m == 0 {
        return;
    }
    let first_page = pools.alloc(m);
    for (j, &key) in misses.iter().enumerate() {
        let pool = (n + j) as u32;
        index.pool_keys.push(key);
        index.pages.push(first_page + j as u32);
        index.modified.push(0);
        index.mark.push(0);
        tc::hash_insert(&mut index.keys, &mut index.vals, index.log2_cap, key, pool);
    }
    // merge into `sorted` from the back, in place
    let old = index.sorted.len();
    index.sorted.resize(old + m, (0, 0));
    let s = &mut index.sorted;
    let (mut i, mut j, mut k) = (old, m, old + m);
    while j > 0 {
        if i > 0 && s[i - 1].0 > misses[j - 1] {
            s[k - 1] = s[i - 1];
            i -= 1;
        } else {
            s[k - 1] = (misses[j - 1], (n + j - 1) as u32);
            j -= 1;
        }
        k -= 1;
    }
}

/// Read access to one volume's voxels (a [`TsdfVolume`] or a submap).
#[derive(Clone, Copy)]
pub(crate) struct VolumeRef<'a> {
    pub(crate) index: &'a BlockIndex,
    pub(crate) pools: &'a Pools,
    pub(crate) grid: tc::Grid,
}

impl VolumeRef<'_> {
    /// Pool offset of the first voxel of block `key`.
    #[inline(always)]
    pub(crate) fn block_base(&self, key: u64) -> Option<usize> {
        let p = self.index.lookup(key)?;
        Some(self.index.pages[p as usize] as usize * self.pools.voxels)
    }

    /// Pool offset of voxel `g`, `None` when its block is not allocated.
    #[inline(always)]
    pub(crate) fn voxel_slot(&self, g: [i32; 3]) -> Option<usize> {
        let l = self.grid.log2;
        let key = tc::pack_block_key([g[0] >> l, g[1] >> l, g[2] >> l]);
        let m = (1 << l) - 1;
        let base = self.block_base(key)?;
        Some(base + tc::local_index([g[0] & m, g[1] & m, g[2] & m], l))
    }
}

impl tc::VoxelSampler for VolumeRef<'_> {
    #[inline(always)]
    fn sample(&self, g: [i32; 3]) -> Option<(f32, u16)> {
        let i = self.voxel_slot(g)?;
        Some((self.pools.tsdf[i], self.pools.weight[i]))
    }
}

/// FNV-1a 64 of the blocks in ascending key order: key, then per voxel
/// tsdf bits, weight and (with colour) rgbw, all little endian.
pub(crate) fn content_hash(index: &BlockIndex, pools: &Pools) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0100_0000_01b3;
    let mut h = OFFSET;
    let mut eat = |bytes: &[u8]| {
        for &b in bytes {
            h = (h ^ u64::from(b)).wrapping_mul(PRIME);
        }
    };
    let v = pools.voxels;
    for &(key, p) in &index.sorted {
        eat(&key.to_le_bytes());
        let base = index.pages[p as usize] as usize * v;
        for i in base..base + v {
            eat(&pools.tsdf[i].to_bits().to_le_bytes());
            eat(&pools.weight[i].to_le_bytes());
            if pools.color {
                eat(&pools.rgbw[i].to_le_bytes());
            }
        }
    }
    h
}

/// One stored voxel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Voxel {
    /// Normalised signed distance in `[-1, 1]` (`+0.0` when unobserved).
    pub tsdf: f32,
    /// Observation count (capped).
    pub weight: u16,
    /// Mean colour (zero without colour).
    pub rgb: [u8; 3],
    /// Colour observation count (capped).
    pub color_weight: u8,
}

/// Borrowed voxels of one block, in-block order `x + B * (y + B * z)`.
#[derive(Debug, Clone, Copy)]
pub struct BlockView<'a> {
    /// Block key.
    pub key: BlockKey,
    /// `B³` normalised distances.
    pub tsdf: &'a [f32],
    /// `B³` weights.
    pub weight: &'a [u16],
    /// `B³` packed colours, `None` without colour.
    pub rgbw: Option<&'a [u32]>,
}

/// A sparse block-hashed TSDF volume with `B³` voxel blocks.
///
/// Storage is structure-of-arrays per block: `tsdf: f32`, `weight: u16`
/// and, with colour, a packed `rgbw: u32` (6 or 10 bytes per voxel).
/// Integration ([`TsdfVolume::integrate`]) allocates every block within
/// the truncation band of the step's measurements, then updates each
/// observation's blocks in slice order.
pub struct TsdfVolume<const B: usize = 8>
where
    BlockDim<B>: SupportedBlockDim,
{
    pub(crate) layout: TsdfLayout,
    pub(crate) params: TsdfParams,
    pub(crate) index: BlockIndex,
    pub(crate) pools: Pools,
    _block: PhantomData<BlockDim<B>>,
}

impl<const B: usize> TsdfVolume<B>
where
    BlockDim<B>: SupportedBlockDim,
{
    pub(crate) const LOG2: u32 = <BlockDim<B> as SupportedBlockDim>::LOG2;
    pub(crate) const VOXELS: usize = <BlockDim<B> as SupportedBlockDim>::VOXELS;

    /// Creates an empty volume and reserves `params.block_capacity` blocks.
    ///
    /// # Arguments
    ///
    /// * `layout` - Voxel size and truncation.
    /// * `params` - Weight caps, colour, capacity and growth.
    ///
    /// # Returns
    ///
    /// The volume.
    ///
    /// # Errors
    ///
    /// [`TsdfError::InvalidParameter`] for a zero cap or a capacity whose
    /// voxel indices exceed `u32`.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_3d::tsdf::{TsdfLayout, TsdfParams, TsdfVolume};
    /// let params = TsdfParams { block_capacity: 1024, ..TsdfParams::default() };
    /// let volume = TsdfVolume::<8>::new(TsdfLayout::new(0.02, 4.0)?, params)?;
    /// assert_eq!(volume.num_blocks(), 0);
    /// # Ok::<(), kornia_3d::tsdf::TsdfError>(())
    /// ```
    pub fn new(layout: TsdfLayout, params: TsdfParams) -> Result<Self, TsdfError> {
        params.validate(Self::VOXELS)?;
        Ok(Self {
            layout,
            index: BlockIndex::with_capacity(params.block_capacity),
            pools: Pools::with_capacity(params.block_capacity, Self::VOXELS, params.color),
            params,
            _block: PhantomData,
        })
    }

    /// Grid geometry.
    pub fn layout(&self) -> &TsdfLayout {
        &self.layout
    }

    /// Volume parameters.
    pub fn params(&self) -> &TsdfParams {
        &self.params
    }

    pub(crate) fn view(&self) -> VolumeRef<'_> {
        VolumeRef {
            index: &self.index,
            pools: &self.pools,
            grid: self.layout.grid(Self::LOG2),
        }
    }

    /// Reserves room for `additional_blocks` more blocks (an explicit,
    /// allocating growth, in either growth mode).
    ///
    /// # Arguments
    ///
    /// * `additional_blocks` - Blocks to add to the current count.
    ///
    /// # Returns
    ///
    /// `Ok(())` once the capacity holds them.
    ///
    /// # Errors
    ///
    /// [`TsdfError::PoolFull`] if voxel indices would exceed `u32`.
    pub fn reserve(&mut self, additional_blocks: usize) -> Result<(), TsdfError> {
        let needed = self.index.len().saturating_add(additional_blocks);
        let limit = max_blocks(Self::VOXELS);
        if needed > limit {
            return Err(TsdfError::PoolFull {
                capacity: limit,
                needed,
            });
        }
        self.index.grow(needed);
        self.pools.grow(needed);
        Ok(())
    }

    /// Allocated blocks.
    pub fn num_blocks(&self) -> usize {
        self.index.len()
    }

    /// Allocated block keys, ascending.
    pub fn block_keys(&self) -> impl ExactSizeIterator<Item = BlockKey> + '_ {
        self.index
            .sorted
            .iter()
            .map(|&(k, _)| BlockKey::from_bits(k))
    }

    /// Blocks the last integration step touched, ascending key.
    pub fn last_step_blocks(&self) -> impl Iterator<Item = BlockKey> + '_ {
        let epoch = self.index.epoch;
        self.index
            .sorted
            .iter()
            .filter(move |&&(_, p)| epoch > 0 && self.index.modified[p as usize] == epoch)
            .map(|&(k, _)| BlockKey::from_bits(k))
    }

    /// The voxels of one block.
    ///
    /// # Arguments
    ///
    /// * `key` - Block key.
    ///
    /// # Returns
    ///
    /// The block, `None` if it is not allocated.
    pub fn block(&self, key: BlockKey) -> Option<BlockView<'_>> {
        let base = self.view().block_base(key.bits())?;
        let r = base..base + Self::VOXELS;
        Some(BlockView {
            key,
            tsdf: &self.pools.tsdf[r.clone()],
            weight: &self.pools.weight[r.clone()],
            rgbw: self.pools.color.then(|| &self.pools.rgbw[r]),
        })
    }

    /// One voxel.
    ///
    /// # Arguments
    ///
    /// * `voxel` - Voxel coordinates.
    ///
    /// # Returns
    ///
    /// The voxel if its block is allocated (its weight may be 0).
    pub fn voxel(&self, voxel: [i32; 3]) -> Option<Voxel> {
        if voxel
            .iter()
            .any(|g| !(-tc::VOXEL_LIMIT..tc::VOXEL_LIMIT).contains(g))
        {
            return None;
        }
        let i = self.view().voxel_slot(voxel)?;
        let s = if self.pools.color {
            self.pools.rgbw[i]
        } else {
            0
        };
        Some(Voxel {
            tsdf: self.pools.tsdf[i],
            weight: self.pools.weight[i],
            rgb: tc::unpack_rgb(s),
            color_weight: (s >> 24) as u8,
        })
    }

    /// Host bytes held by the volume (capacities, not just live blocks).
    pub fn memory_bytes(&self) -> usize {
        self.index.bytes() + self.pools.bytes()
    }

    /// FNV-1a 64 over the blocks in ascending key: key (`u64` LE), then per
    /// voxel tsdf bits (`u32` LE), weight (`u16` LE) and, with colour,
    /// rgbw (`u32` LE). Equal on every backend for the same inputs.
    pub fn content_hash(&self) -> u64 {
        content_hash(&self.index, &self.pools)
    }

    /// Test hook: allocates `keys` and stores `tsd(centre)` (weight 1) in
    /// every voxel where it returns `Some`.
    #[cfg(test)]
    pub(crate) fn fill_for_test(
        &mut self,
        keys: &mut Vec<u64>,
        tsd: impl Fn([f32; 3]) -> Option<f32>,
    ) -> Result<(), TsdfError> {
        keys.sort_unstable();
        keys.dedup();
        keys.retain(|&k| self.index.lookup(k).is_none());
        ensure_capacity(
            &mut self.index,
            &mut self.pools,
            self.params.growth,
            keys.len(),
        )?;
        insert_blocks(&mut self.index, &mut self.pools, keys);
        self.index.epoch += 1;
        let grid = self.layout.grid(Self::LOG2);
        let b = B as i32;
        for &(key, p) in &self.index.sorted {
            self.index.modified[p as usize] = self.index.epoch;
            let bc = tc::unpack_block_key(key);
            let base = self.index.pages[p as usize] as usize * Self::VOXELS;
            for lz in 0..b {
                for ly in 0..b {
                    for lx in 0..b {
                        let g = [bc[0] * b + lx, bc[1] * b + ly, bc[2] * b + lz];
                        let c =
                            tc::voxel_centre::<super::StdOps>(g, grid.voxel_size, grid.half_voxel);
                        if let Some(v) = tsd(c) {
                            let i = base + tc::local_index([lx, ly, lz], Self::LOG2);
                            self.pools.tsdf[i] = v;
                            self.pools.weight[i] = 1;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
