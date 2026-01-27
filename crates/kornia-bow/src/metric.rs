use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;
use std::fmt::Debug;
use std::ops::{Deref, DerefMut};

/// A wrapper for fixed-size arrays to support Serde with const generics > 32.
#[derive(Clone, Copy, Debug, PartialEq, Encode, Decode)]
#[repr(transparent)]
pub struct Feature<T, const N: usize>(pub [T; N]);

impl<T: Default + Copy, const N: usize> Default for Feature<T, N> {
    #[inline(always)]
    fn default() -> Self {
        Feature([T::default(); N])
    }
}

impl<T, const N: usize> Deref for Feature<T, N> {
    type Target = [T; N];
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T, const N: usize> DerefMut for Feature<T, N> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T: Serialize + Copy, const N: usize> Serialize for Feature<T, N> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        BigArray::serialize(&self.0, serializer)
    }
}

impl<'de, T: Deserialize<'de> + Copy + Default, const N: usize> Deserialize<'de> for Feature<T, N> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(Feature(BigArray::deserialize(deserializer)?))
    }
}

/// Distance metric abstraction for vocabulary tree operations.
pub trait DistanceMetric:
    Clone
    + Copy
    + Debug
    + Send
    + Sync
    + 'static
    + Serialize
    + for<'de> Deserialize<'de>
    + Encode
    + Decode<()>
{
    /// The underlying descriptor data type.
    type Data: Copy
        + Debug
        + Default
        + Send
        + Sync
        + Serialize
        + for<'de> Deserialize<'de>
        + Encode
        + Decode<()>
        + PartialEq;

    /// The resulting distance type.
    type Distance: Copy + PartialOrd + Default + Send + Sync + Debug;

    /// Calculates the distance between two descriptors.
    fn distance(a: &Self::Data, b: &Self::Data) -> Self::Distance;

    /// Returns the maximum possible distance value.
    fn max_distance() -> Self::Distance;

    /// Converts a distance value to a 32-bit float.
    fn to_f32(dist: Self::Distance) -> f32;

    /// Returns the identifier for the metric type.
    fn metric_type() -> MetricType;

    /// Returns a value used to pad unused slots in a block.
    fn padding() -> Self::Data;
}

/// Supported distance metric types.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Encode, Decode)]
pub enum MetricType {
    Hamming,
    L2,
}

/// Hamming distance for binary descriptors.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, Encode, Decode)]
pub struct Hamming<const D: usize>;

impl<const D: usize> DistanceMetric for Hamming<D> {
    type Data = Feature<u64, D>;
    type Distance = u32;

    #[inline(always)]
    fn distance(a: &Self::Data, b: &Self::Data) -> Self::Distance {
        if D == 4 {
            let d0 = (a[0] ^ b[0]).count_ones();
            let d1 = (a[1] ^ b[1]).count_ones();
            let d2 = (a[2] ^ b[2]).count_ones();
            let d3 = (a[3] ^ b[3]).count_ones();
            d0 + d1 + d2 + d3
        } else {
            let mut dist = 0;
            for i in 0..D {
                dist += (a[i] ^ b[i]).count_ones();
            }
            dist
        }
    }

    #[inline(always)]
    fn max_distance() -> Self::Distance {
        u32::MAX
    }

    #[inline(always)]
    fn to_f32(dist: Self::Distance) -> f32 {
        dist as f32
    }

    #[inline(always)]
    fn metric_type() -> MetricType {
        MetricType::Hamming
    }

    #[inline(always)]
    fn padding() -> Self::Data {
        Feature([u64::MAX; D])
    }
}

/// Squared L2 (Euclidean) distance for floating-point descriptors.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, Encode, Decode)]
pub struct L2<const D: usize>;

impl<const D: usize> DistanceMetric for L2<D> {
    type Data = Feature<f32, D>;
    type Distance = f32;

    #[inline(always)]
    fn distance(a: &Self::Data, b: &Self::Data) -> Self::Distance {
        let mut dist_sq = 0.0;
        for i in 0..D {
            let diff = a[i] - b[i];
            dist_sq += diff * diff;
        }
        dist_sq
    }

    #[inline(always)]
    fn max_distance() -> Self::Distance {
        f32::INFINITY
    }

    #[inline(always)]
    fn to_f32(dist: Self::Distance) -> f32 {
        dist
    }

    #[inline(always)]
    fn metric_type() -> MetricType {
        MetricType::L2
    }

    #[inline(always)]
    fn padding() -> Self::Data {
        Feature([f32::MAX; D])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamming_distance() {
        let a = Feature([0b1010, 0, 0, 0]);
        let b = Feature([0b1100, 0, 0, 0]);
        // 1010 ^ 1100 = 0110 (2 bits different)
        assert_eq!(Hamming::<4>::distance(&a, &b), 2);

        let c = Feature([u64::MAX, u64::MAX, u64::MAX, u64::MAX]);
        let d = Feature([0, 0, 0, 0]);
        assert_eq!(Hamming::<4>::distance(&c, &d), 256);
    }

    #[test]
    fn test_l2_distance() {
        let a = Feature([1.0, 2.0, 3.0]);
        let b = Feature([4.0, 6.0, 3.0]);
        // (4-1)^2 + (6-2)^2 + (3-3)^2 = 3^2 + 4^2 + 0^2 = 9 + 16 = 25
        assert_eq!(L2::<3>::distance(&a, &b), 25.0);
    }
}
