use rustc_hash::FxHashMap;

use super::{VoxelConnectivity, VoxelKey, VoxelLayout};

/// A sparse voxel grid: one value of type `T` per occupied voxel, stored in
/// a hash map keyed by [`VoxelKey`] (FxHash: fast and seed-free).
///
/// [`iter`](Self::iter) visits the voxels in hash order, which is not part
/// of the API; [`sorted_keys`](Self::sorted_keys) gives a deterministic
/// order.
///
/// # Example
///
/// ```
/// use kornia_3d::voxel::{VoxelConnectivity, VoxelGrid, VoxelLayout};
///
/// let mut grid = VoxelGrid::<u32>::new(VoxelLayout::new(0.1, [0.0; 3])?);
/// for p in [[0.05, 0.05, 0.05], [0.07, 0.01, 0.02], [0.15, 0.05, 0.05]] {
///     if let Some(key) = grid.key_from_point(&p) {
///         *grid.get_or_insert_with(key, || 0) += 1;
///     }
/// }
/// assert_eq!(grid.get(&[0, 0, 0]), Some(&2));
/// assert_eq!(grid.neighbours([0, 0, 0], VoxelConnectivity::Six).count(), 1);
/// # Ok::<(), kornia_3d::voxel::VoxelError>(())
/// ```
#[derive(Debug, Clone)]
pub struct VoxelGrid<T> {
    layout: VoxelLayout,
    cells: FxHashMap<VoxelKey, T>,
}

impl<T> VoxelGrid<T> {
    /// Creates an empty grid.
    ///
    /// # Arguments
    ///
    /// * `layout` - Voxel size and origin.
    ///
    /// # Returns
    ///
    /// A grid with no occupied voxel.
    pub fn new(layout: VoxelLayout) -> Self {
        Self {
            layout,
            cells: FxHashMap::default(),
        }
    }

    /// Creates an empty grid with room for `capacity` voxels.
    ///
    /// # Arguments
    ///
    /// * `layout` - Voxel size and origin.
    /// * `capacity` - Voxels to reserve.
    ///
    /// # Returns
    ///
    /// A grid with no occupied voxel.
    pub fn with_capacity(layout: VoxelLayout, capacity: usize) -> Self {
        Self {
            layout,
            cells: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
        }
    }

    /// The grid's voxel size and origin.
    ///
    /// # Returns
    ///
    /// The layout given at construction.
    pub fn layout(&self) -> &VoxelLayout {
        &self.layout
    }

    /// Occupied voxels.
    ///
    /// # Returns
    ///
    /// The number of stored voxels.
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    /// Whether no voxel is stored.
    ///
    /// # Returns
    ///
    /// `true` for an empty grid.
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// Removes every voxel, keeping the allocation.
    pub fn clear(&mut self) {
        self.cells.clear();
    }

    /// The key of the voxel containing `point`; see
    /// [`VoxelLayout::key_from_point`].
    ///
    /// # Arguments
    ///
    /// * `point` - Point in the units of the layout.
    ///
    /// # Returns
    ///
    /// The key, or `None` if it does not fit `i32` or `point` is not finite.
    #[inline]
    pub fn key_from_point(&self, point: &[f64; 3]) -> Option<VoxelKey> {
        self.layout.key_from_point(point)
    }

    /// The centre of voxel `key`; see [`VoxelLayout::centre_from_key`].
    ///
    /// # Arguments
    ///
    /// * `key` - Voxel key.
    ///
    /// # Returns
    ///
    /// The voxel centre.
    #[inline]
    pub fn centre_from_key(&self, key: &VoxelKey) -> [f64; 3] {
        self.layout.centre_from_key(key)
    }

    /// Stores `value` at `key`.
    ///
    /// # Arguments
    ///
    /// * `key` - Voxel key.
    /// * `value` - Value to store.
    ///
    /// # Returns
    ///
    /// The previous value at `key`, if any.
    pub fn insert(&mut self, key: VoxelKey, value: T) -> Option<T> {
        self.cells.insert(key, value)
    }

    /// The value at `key`.
    ///
    /// # Arguments
    ///
    /// * `key` - Voxel key.
    ///
    /// # Returns
    ///
    /// A reference to the value, or `None` for an empty voxel.
    pub fn get(&self, key: &VoxelKey) -> Option<&T> {
        self.cells.get(key)
    }

    /// The value at `key`, mutably.
    ///
    /// # Arguments
    ///
    /// * `key` - Voxel key.
    ///
    /// # Returns
    ///
    /// A mutable reference to the value, or `None` for an empty voxel.
    pub fn get_mut(&mut self, key: &VoxelKey) -> Option<&mut T> {
        self.cells.get_mut(key)
    }

    /// The value at `key`, inserting `default()` first if the voxel is empty.
    ///
    /// # Arguments
    ///
    /// * `key` - Voxel key.
    /// * `default` - Builds the value of a new voxel.
    ///
    /// # Returns
    ///
    /// A mutable reference to the stored value.
    pub fn get_or_insert_with(&mut self, key: VoxelKey, default: impl FnOnce() -> T) -> &mut T {
        self.cells.entry(key).or_insert_with(default)
    }

    /// Removes the voxel at `key`.
    ///
    /// # Arguments
    ///
    /// * `key` - Voxel key.
    ///
    /// # Returns
    ///
    /// The removed value, or `None` for an empty voxel.
    pub fn remove(&mut self, key: &VoxelKey) -> Option<T> {
        self.cells.remove(key)
    }

    /// Whether voxel `key` is occupied.
    ///
    /// # Arguments
    ///
    /// * `key` - Voxel key.
    ///
    /// # Returns
    ///
    /// `true` if a value is stored at `key`.
    pub fn contains_key(&self, key: &VoxelKey) -> bool {
        self.cells.contains_key(key)
    }

    /// The occupied voxels, in unspecified (hash) order.
    ///
    /// # Returns
    ///
    /// An iterator over `(key, value)`.
    pub fn iter(&self) -> impl Iterator<Item = (&VoxelKey, &T)> + '_ {
        self.cells.iter()
    }

    /// The occupied voxels, mutably, in unspecified (hash) order.
    ///
    /// # Returns
    ///
    /// An iterator over `(key, value)`.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&VoxelKey, &mut T)> + '_ {
        self.cells.iter_mut()
    }

    /// The occupied keys in ascending lexicographic `(x, y, z)` order.
    ///
    /// # Returns
    ///
    /// A new vector of keys.
    pub fn sorted_keys(&self) -> Vec<VoxelKey> {
        let mut keys: Vec<VoxelKey> = self.cells.keys().copied().collect();
        keys.sort_unstable();
        keys
    }

    /// The occupied neighbours of voxel `key`.
    ///
    /// Neighbours whose key would overflow `i32` do not exist. `key` itself
    /// need not be occupied.
    ///
    /// # Arguments
    ///
    /// * `key` - Voxel key.
    /// * `connectivity` - Which of the 26 surrounding voxels to visit.
    ///
    /// # Returns
    ///
    /// An iterator over `(neighbour key, value)` in the lexicographic order
    /// of [`VoxelConnectivity::offsets`].
    pub fn neighbours(
        &self,
        key: VoxelKey,
        connectivity: VoxelConnectivity,
    ) -> impl Iterator<Item = (VoxelKey, &T)> + '_ {
        connectivity.offsets().iter().filter_map(move |d| {
            let n = [
                key[0].checked_add(d[0])?,
                key[1].checked_add(d[1])?,
                key[2].checked_add(d[2])?,
            ];
            self.cells.get(&n).map(|v| (n, v))
        })
    }
}
