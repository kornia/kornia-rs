use rayon::prelude::*;

/// Raw-pointer wrapper that is `Send + Sync` by construction.
///
/// Used by `process_strips_parallel` to share a pointer to disjoint sub-ranges
/// across Rayon threads.  Only the owning function may access the pointer; the
/// type exists solely to satisfy Rayon's `Sync` bound on the closure.
struct UfRawPtr32(*mut u32);
unsafe impl Send for UfRawPtr32 {}
unsafe impl Sync for UfRawPtr32 {}
impl UfRawPtr32 {
    /// Returns `self.0.add(n)` — a method call avoids Rust-2021 field-projection
    /// capture (`self.0`) which would regress the closure to `!Sync`.
    #[inline(always)]
    fn add(&self, n: usize) -> *mut u32 {
        // SAFETY: caller guarantees `n` is within bounds.
        unsafe { self.0.add(n) }
    }
}

/// A disjoint-set (union-find) data structure.
///
/// Uses `u32` internally to halve memory footprint (parent + size arrays) and
/// improve cache utilisation.  Pixel indices up to ~4 billion fit in u32.
#[derive(Clone)]
pub struct UnionFind {
    pub(crate) parent: Vec<u32>,
    size: Vec<u32>,
}

impl UnionFind {
    /// Creates a new UnionFind structure with length `len`.
    pub fn new(len: usize) -> Self {
        Self {
            parent: vec![u32::MAX; len],
            size: vec![1u32; len],
        }
    }

    /// Returns the representative (root) of the set containing `id`, with path halving.
    pub fn get_representative(&mut self, mut id: usize) -> usize {
        if self.parent[id] == u32::MAX {
            self.parent[id] = id as u32;
            return id;
        }

        while self.parent[id] as usize != id {
            let pp = self.parent[self.parent[id] as usize];
            self.parent[id] = pp;
            id = pp as usize;
        }
        id
    }

    /// Returns the representative of the set containing `id` without path compression.
    ///
    /// Safe to call from multiple threads simultaneously since it only reads `parent`.
    pub fn get_representative_ref(&self, mut id: usize) -> usize {
        if self.parent[id] == u32::MAX {
            return id;
        }
        while self.parent[id] as usize != id {
            id = self.parent[id] as usize;
        }
        id
    }

    /// Returns the size of the set containing `id`.
    pub fn get_set_size(&mut self, id: usize) -> usize {
        let repid = self.get_representative(id);
        self.size[repid] as usize
    }

    /// Returns the size of the set containing `id` without path compression.
    ///
    /// Safe to call from multiple threads simultaneously.
    pub fn get_set_size_ref(&self, id: usize) -> usize {
        let repid = self.get_representative_ref(id);
        self.size[repid] as usize
    }

    /// Unites the sets containing `aid` and `bid`, returning the representative of the resulting set.
    pub fn connect(&mut self, aid: usize, bid: usize) -> usize {
        let aroot = self.get_representative(aid);
        let broot = self.get_representative(bid);

        if aroot == broot {
            return aroot;
        }

        let asize = self.size[aroot];
        let bsize = self.size[broot];

        if asize > bsize {
            self.parent[broot] = aroot as u32;
            self.size[aroot] += bsize;
            aroot
        } else {
            self.parent[aroot] = broot as u32;
            self.size[broot] += asize;
            broot
        }
    }

    /// Resets the UnionFind structure to its initial state.
    pub fn reset(&mut self) {
        self.parent.fill(u32::MAX);
        self.size.fill(1);
    }

    /// Returns the component size when `rep` is a known root (O(1), no parent traversal).
    #[inline(always)]
    pub fn size_at(&self, rep: usize) -> usize {
        self.size[rep] as usize
    }

    /// Full path compression: writes `parent[i] = root` for every non-isolated element.
    pub fn compress_all(&mut self) {
        for i in 0..self.parent.len() {
            let p = self.parent[i];
            if p != u32::MAX {
                let root = self.get_representative(i);
                self.parent[i] = root as u32;
            }
        }
    }

    /// Fills `cache` with one u32 entry per pixel after `compress_all` has been called.
    pub fn fill_rep_cache_filtered(&self, cache: &mut Vec<u32>, min_size: usize) {
        debug_assert!(cache.len() >= self.parent.len());
        for (dst, &p) in cache.iter_mut().zip(self.parent.iter()) {
            *dst = if p == u32::MAX || (self.size[p as usize] as usize) < min_size {
                u32::MAX
            } else {
                p
            };
        }
    }

    /// Fuses path-compression + `rep_cache` fill in parallel across N Rayon strips.
    pub fn compress_and_fill_rep_cache(&mut self, cache: &mut Vec<u32>, min_size: usize) {
        debug_assert!(cache.len() >= self.parent.len());
        let total = self.parent.len();
        let n_threads = rayon::current_num_threads().max(1);
        let strip = (total + n_threads - 1) / n_threads;

        let parent_ptr = UfRawPtr32(self.parent.as_mut_ptr());
        let size_ptr   = UfRawPtr32(self.size.as_mut_ptr());
        let cache_ptr  = UfRawPtr32(cache.as_mut_ptr());

        (0..n_threads).into_par_iter().for_each(|t| {
            let start = t * strip;
            if start >= total { return; }
            let end = (start + strip).min(total);

            // Cache last parent→root mapping. Run-based CC gives pixels in the
            // same horizontal run the same parent value → skip the second-hop read
            // for every pixel in a run except the first. Avg run length ~8 → 7/8
            // of second-hop reads are eliminated.
            let mut last_p = u32::MAX;
            let mut last_root = 0u32;
            let mut last_cv = u32::MAX;

            for i in start..end {
                let p = unsafe { *parent_ptr.add(i) };
                if p == u32::MAX {
                    unsafe { *cache_ptr.add(i) = u32::MAX; }
                    last_p = u32::MAX;
                    continue;
                }
                let (root, cv) = if p == last_p {
                    // Same parent as previous pixel (same run) — reuse cached result.
                    (last_root as usize, last_cv)
                } else {
                    // New parent: walk to root with path halving.
                    let root = {
                        let mut cur = i;
                        loop {
                            let cur_p = unsafe { *parent_ptr.add(cur) } as usize;
                            if cur_p == cur { break cur; }
                            let pp = unsafe { *parent_ptr.add(cur_p) };
                            if cur >= start && cur < end {
                                unsafe { *parent_ptr.add(cur) = pp; }
                            }
                            cur = pp as usize;
                        }
                    };
                    let sz = unsafe { *size_ptr.add(root) } as usize;
                    let cv = if sz >= min_size { root as u32 } else { u32::MAX };
                    last_p = p;
                    last_root = root as u32;
                    last_cv = cv;
                    (root, cv)
                };
                unsafe { *parent_ptr.add(i) = root as u32; }
                unsafe { *cache_ptr.add(i) = cv; }
            }
        });
    }

    /// Runs `f(strip_idx, par_strip)` for each horizontal strip in parallel.
    pub(crate) fn process_strips_parallel<F>(&mut self, n_strips: usize, strip_pixels: usize, f: F)
    where
        F: Fn(usize, ParStripUF<'_>) + Sync + Send,
    {
        let total = self.parent.len();
        let parent_ptr = UfRawPtr32(self.parent.as_mut_ptr());
        let size_ptr = UfRawPtr32(self.size.as_mut_ptr());

        (0..n_strips).into_par_iter().for_each(|t| {
            let start = t * strip_pixels;
            if start >= total {
                return;
            }
            let end = (start + strip_pixels).min(total);
            let len = end - start;

            let parent_sub =
                unsafe { std::slice::from_raw_parts_mut(parent_ptr.add(start), len) };
            let size_sub =
                unsafe { std::slice::from_raw_parts_mut(size_ptr.add(start), len) };

            f(
                t,
                ParStripUF {
                    parent: parent_sub,
                    size: size_sub,
                    offset: start,
                },
            );
        });
    }

    /// Returns the number of elements in the UnionFind structure.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.parent.len()
    }
}

/// A mutable view into one horizontal strip's portion of a [`UnionFind`].
pub(crate) struct ParStripUF<'a> {
    pub(crate) parent: &'a mut [u32],
    pub(crate) size: &'a mut [u32],
    pub(crate) offset: usize,
}

impl<'a> ParStripUF<'a> {
    #[inline]
    pub(crate) fn get_representative(&mut self, global_i: usize) -> usize {
        let local = global_i - self.offset;
        if self.parent[local] == u32::MAX {
            self.parent[local] = global_i as u32;
            return global_i;
        }
        let mut cur = global_i;
        loop {
            let cur_local = cur - self.offset;
            let p = self.parent[cur_local] as usize;
            if p == cur {
                return cur;
            }
            let pp = self.parent[p - self.offset];
            self.parent[cur_local] = pp;
            cur = pp as usize;
        }
    }

    #[inline]
    pub(crate) fn connect(&mut self, global_a: usize, global_b: usize) {
        let ar = self.get_representative(global_a);
        let br = self.get_representative(global_b);
        if ar == br {
            return;
        }
        let as_ = self.size[ar - self.offset];
        let bs = self.size[br - self.offset];
        if as_ > bs {
            self.parent[br - self.offset] = ar as u32;
            self.size[ar - self.offset] += bs;
        } else {
            self.parent[ar - self.offset] = br as u32;
            self.size[br - self.offset] += as_;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_representative() {
        let mut uf = UnionFind::new(10);

        assert_eq!(uf.get_representative(0), 0);
        assert_eq!(uf.get_representative(5), 5);
    }

    #[test]
    fn test_union() {
        let mut uf = UnionFind::new(10);

        uf.connect(0, 1);
        assert_eq!(uf.get_representative(0), uf.get_representative(1));

        uf.connect(1, 2);
        assert_eq!(uf.get_representative(0), uf.get_representative(2));

        uf.connect(3, 4);
        assert_eq!(uf.get_representative(3), uf.get_representative(4));

        uf.connect(0, 3);
        assert_eq!(uf.get_representative(0), uf.get_representative(4));
    }

    #[test]
    fn test_reset() {
        let mut uf = UnionFind::new(10);
        uf.connect(0, 1);
        uf.connect(2, 3);

        assert_ne!(uf.get_representative(1), u32::MAX as usize);

        uf.reset();

        assert!(uf.parent.iter().all(|&p| p == u32::MAX));
        assert!(uf.size.iter().all(|&s| s == 1));
    }
}
