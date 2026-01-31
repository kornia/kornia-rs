use serde::{Deserialize, Serialize};

/// Sparse Bag of Words: Sorted List of (WordID, Weight).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoW(pub Vec<(u32, f32)>);

impl BoW {
    /// Normalizes the BoW vector using L1 norm (sum of absolute values = 1.0).
    pub fn normalize_l1(&mut self) {
        let sum: f32 = self.0.iter().map(|&(_, v)| v.abs()).sum();
        if sum > 0.0 {
            for (_, v) in self.0.iter_mut() {
                *v /= sum;
            }
        }
    }

    /// Normalizes the BoW vector using L2 norm (sum of squares = 1.0).
    pub fn normalize_l2(&mut self) {
        let sum_sq: f32 = self.0.iter().map(|&(_, v)| v * v).sum();
        if sum_sq > 0.0 {
            let norm = sum_sq.sqrt();
            for (_, v) in self.0.iter_mut() {
                *v /= norm;
            }
        }
    }

    /// Computes L1 similarity score.
    pub fn l1_similarity(&self, other: &Self) -> f32 {
        let mut it1 = self.0.iter().peekable();
        let mut it2 = other.0.iter().peekable();
        let mut dist = 0.0;

        loop {
            match (it1.peek(), it2.peek()) {
                (Some(&(id1, val1)), Some(&(id2, val2))) => {
                    if id1 == id2 {
                        dist += (val1 - val2).abs();
                        it1.next();
                        it2.next();
                    } else if id1 < id2 {
                        dist += val1.abs();
                        it1.next();
                    } else {
                        dist += val2.abs();
                        it2.next();
                    }
                }
                (Some(&(_id1, val1)), None) => {
                    dist += val1.abs();
                    it1.next();
                }
                (None, Some(&(_id2, val2))) => {
                    dist += val2.abs();
                    it2.next();
                }
                (None, None) => break,
            }
        }

        1.0 - 0.5 * dist
    }

    /// Computes L2 similarity score.
    pub fn l2(&self, other: &Self) -> f32 {
        let dot = self.dot_product(other);
        if dot >= 1.0 {
            1.0
        } else {
            1.0 - (1.0 - dot).sqrt()
        }
    }

    /// Computes Chi-Square similarity score.
    pub fn chi_square(&self, other: &Self) -> f32 {
        let mut it1 = self.0.iter().peekable();
        let mut it2 = other.0.iter().peekable();
        let mut score = 0.0;

        while let (Some(&(id1, val1)), Some(&(id2, val2))) = (it1.peek(), it2.peek()) {
            if id1 == id2 {
                let sum = val1 + val2;
                if sum != 0.0 {
                    score += (val1 * val2) / sum;
                }
                it1.next();
                it2.next();
            } else if id1 < id2 {
                it1.next();
            } else {
                it2.next();
            }
        }

        2.0 * score
    }

    /// Computes KL Divergence score.
    pub fn kl_divergence(&self, other: &Self) -> f32 {
        let mut it1 = self.0.iter().peekable();
        let mut it2 = other.0.iter().peekable();
        let mut score = 0.0;
        let log_eps = f32::EPSILON.ln();

        loop {
            match (it1.peek(), it2.peek()) {
                (Some(&(id1, val1)), Some(&(id2, val2))) => {
                    if id1 == id2 {
                        if *val1 != 0.0 && *val2 != 0.0 {
                            score += val1 * (val1 / val2).ln();
                        }
                        it1.next();
                        it2.next();
                    } else if id1 < id2 {
                        score += val1 * (val1.ln() - log_eps);
                        it1.next();
                    } else {
                        it2.next();
                    }
                }
                (Some(&(_id1, val1)), None) => {
                    score += val1 * (val1.ln() - log_eps);
                    it1.next();
                }
                (None, Some(_)) => {
                    it2.next();
                }
                (None, None) => break,
            }
        }

        score
    }

    /// Computes Bhattacharyya similarity score.
    pub fn bhattacharyya(&self, other: &Self) -> f32 {
        let mut it1 = self.0.iter().peekable();
        let mut it2 = other.0.iter().peekable();
        let mut score = 0.0;

        while let (Some(&(id1, val1)), Some(&(id2, val2))) = (it1.peek(), it2.peek()) {
            if id1 == id2 {
                score += (val1 * val2).sqrt();
                it1.next();
                it2.next();
            } else if id1 < id2 {
                it1.next();
            } else {
                it2.next();
            }
        }

        score
    }

    /// Computes Dot Product similarity score.
    pub fn dot_product(&self, other: &Self) -> f32 {
        let mut it1 = self.0.iter().peekable();
        let mut it2 = other.0.iter().peekable();
        let mut score = 0.0;

        while let (Some(&(id1, val1)), Some(&(id2, val2))) = (it1.peek(), it2.peek()) {
            if id1 == id2 {
                score += val1 * val2;
                it1.next();
                it2.next();
            } else if id1 < id2 {
                it1.next();
            } else {
                it2.next();
            }
        }

        score
    }
}

/// DirectIndex: Sorted List of `(NodeID, Vec<u32>)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DirectIndex(pub Vec<(u32, Vec<u32>)>);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bow_l1_similarity() {
        let bow1 = BoW(vec![(1, 0.4), (2, 0.6)]);
        let bow2 = BoW(vec![(1, 0.5), (3, 0.5)]);

        let sim = bow1.l1_similarity(&bow2);
        assert!((sim - 0.4).abs() < 1e-6);

        assert!((bow1.l1_similarity(&bow1) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_similarity() {
        let bow1 = BoW(vec![(1, 0.4), (2, 0.6)]);
        let bow2 = BoW(vec![(1, 0.5), (3, 0.5)]);

        let dot = bow1.dot_product(&bow2);
        assert!((dot - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_l2_similarity() {
        let bow1 = BoW(vec![(1, 0.4), (2, 0.6)]);
        let bow2 = BoW(vec![(1, 0.5), (3, 0.5)]);

        let l2 = bow1.l2(&bow2);
        assert!((l2 - 0.10557281).abs() < 1e-5);
    }

    #[test]
    fn test_bhattacharyya_similarity() {
        let bow1 = BoW(vec![(1, 0.4), (2, 0.6)]);
        let bow2 = BoW(vec![(1, 0.5), (3, 0.5)]);

        let bhat = bow1.bhattacharyya(&bow2);
        assert!((bhat - 0.4472136).abs() < 1e-5);
    }

    #[test]
    fn test_chi_square_similarity() {
        let bow1 = BoW(vec![(1, 0.4), (2, 0.6)]);
        let bow2 = BoW(vec![(1, 0.5), (3, 0.5)]);

        let chi = bow1.chi_square(&bow2);
        assert!((chi - 0.444444).abs() < 1e-5);
    }

    #[test]
    fn test_kl_divergence_similarity() {
        let bow1 = BoW(vec![(1, 0.4), (2, 0.6)]);
        let bow2 = BoW(vec![(1, 0.5), (3, 0.5)]);

        let kl = bow1.kl_divergence(&bow2);
        assert!(kl > 9.0 && kl < 9.3);
    }
}
