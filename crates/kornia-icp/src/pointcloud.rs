use kornia_core::{allocator::CpuAllocator, tensor::Tensor};

#[derive(thiserror::Error, Debug)]
pub enum PointCloudError {
    #[error("Invalid pointcloud shape")]
    InvalidShape(#[from] kornia_core::TensorError),

    #[error("Pointcloud data is empty")]
    EmptyData,
}

pub struct PointCloud(Tensor<f32, 2, CpuAllocator>);

impl PointCloud {
    pub fn from_vec(points: Vec<[f32; 3]>) -> Result<Self, PointCloudError> {
        if points.is_empty() {
            return Err(PointCloudError::EmptyData);
        }
        let num_points = points.len();
        let vec_points = points.into_iter().flatten().collect::<Vec<_>>();

        Ok(Self(Tensor::from_shape_vec(
            [num_points, 3],
            vec_points,
            CpuAllocator,
        )?))
    }

    pub fn num_points(&self) -> usize {
        self.0.numel() / 3
    }

    pub fn is_empty(&self) -> bool {
        self.0.numel() == 0
    }
}

impl std::ops::Deref for PointCloud {
    type Target = Tensor<f32, 2, CpuAllocator>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pointcloud_smoke() -> Result<(), PointCloudError> {
        let data = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let pc = PointCloud::from_vec(data)?;
        assert_eq!(pc.num_points(), 3);
        Ok(())
    }

    #[test]
    fn test_pointcloud_empty() -> Result<(), PointCloudError> {
        let data = vec![];
        let pc = PointCloud::from_vec(data);
        assert!(matches!(pc, Err(PointCloudError::EmptyData)));
        Ok(())
    }
}
