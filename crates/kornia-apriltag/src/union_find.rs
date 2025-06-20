/// A disjoint-set (union-find) data structure.
pub struct UnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl UnionFind {
    /// Creates a new UnionFind structure with elements from 0 to `maxid`.
    pub fn new(len: usize) -> Self {
        Self {
            parent: vec![usize::MAX; len],
            size: vec![1; len],
        }
    }

    /// Returns the representative (root) of the set containing `id`, with path compression.
    pub fn get_representative(&mut self, id: usize) -> usize {
        let mut root = self.parent[id];

        if root == usize::MAX {
            self.parent[id] = id;
            return id;
        }

        // Chase down the root
        while self.parent[root] != root {
            root = self.parent[root];
        }

        // Go back and collapse the tree
        let mut id = id;
        while self.parent[id] != root {
            let tmp = self.parent[id];
            self.parent[id] = root;
            id = tmp;
        }

        root
    }

    /// Unites the sets containing `aid` and `bid`.
    pub fn union(&mut self, aid: usize, bid: usize) {
        let aroot = self.get_representative(aid);
        let broot = self.get_representative(bid);

        if aroot == broot {
            return;
        }

        let asize = self.size[aroot];
        let bsize = self.size[broot];

        if asize > bsize {
            self.parent[broot] = aroot;
            self.size[aroot] += bsize;
        } else {
            self.parent[aroot] = broot;
            self.size[broot] += asize;
        }
    }
}
