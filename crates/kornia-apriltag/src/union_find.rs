/// A disjoint-set (union-find) data structure.
#[derive(Clone)]
pub struct UnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl UnionFind {
    /// Creates a new UnionFind structure with length `len`.
    pub fn new(len: usize) -> Self {
        Self {
            parent: vec![usize::MAX; len],
            size: vec![1; len],
        }
    }

    /// Returns the representative (root) of the set containing `id`, with path halving.
    pub fn get_representative(&mut self, mut id: usize) -> usize {
        if self.parent[id] == usize::MAX {
            self.parent[id] = id;
            return id;
        }

        while self.parent[id] != id {
            self.parent[id] = self.parent[self.parent[id]];
            id = self.parent[id];
        }
        id
    }

    /// Returns the representative of the set containing `id` without path compression.
    ///
    /// Safe to call from multiple threads simultaneously since it only reads `parent`.
    /// Slightly slower per call than `get_representative` but requires only `&self`.
    pub fn get_representative_ref(&self, mut id: usize) -> usize {
        if self.parent[id] == usize::MAX {
            return id;
        }
        while self.parent[id] != id {
            id = self.parent[id];
        }
        id
    }

    /// Returns the size of the set containing `id`.
    pub fn get_set_size(&mut self, id: usize) -> usize {
        let repid = self.get_representative(id);
        self.size[repid]
    }

    /// Returns the size of the set containing `id` without path compression.
    ///
    /// Safe to call from multiple threads simultaneously.
    pub fn get_set_size_ref(&self, id: usize) -> usize {
        let repid = self.get_representative_ref(id);
        self.size[repid]
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
            self.parent[broot] = aroot;
            self.size[aroot] += bsize;
            aroot
        } else {
            self.parent[aroot] = broot;
            self.size[broot] += asize;
            broot
        }
    }

    /// Resets the UnionFind structure to its initial state.
    pub fn reset(&mut self) {
        (0..self.parent.len()).for_each(|i| {
            self.parent[i] = usize::MAX;
            self.size[i] = 1;
        });
    }

    /// Returns the number of elements in the UnionFind structure.
    // We do not provide `is_empty` because a UnionFind structure is always initialized with a fixed, nonzero length.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.parent.len()
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

        assert_ne!(uf.get_representative(1), usize::MAX);

        uf.reset();

        assert!(uf.parent.iter().all(|&p| p == usize::MAX));
        assert!(uf.size.iter().all(|&s| s == 1));
    }
}
