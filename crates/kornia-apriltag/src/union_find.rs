/// A disjoint-set (union-find) data structure.
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

    /// Returns the representative (root) of the set containing `id`, with path compression.
    pub fn get_representative(&mut self, mut id: usize) -> usize {
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
        while self.parent[id] != root {
            let tmp = self.parent[id];
            self.parent[id] = root;
            id = tmp;
        }

        root
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
            return aroot;
        } else {
            self.parent[aroot] = broot;
            self.size[broot] += asize;
            return broot;
        }
    }

    /// Resets the UnionFind structure to its initial state.
    pub fn reset(&mut self) {
        (0..self.parent.len()).into_iter().for_each(|i| {
            self.parent[i] = usize::MAX;
            self.size[i] = 1;
        });
    }

    /// Returns the number of elements in the UnionFind structure.
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
