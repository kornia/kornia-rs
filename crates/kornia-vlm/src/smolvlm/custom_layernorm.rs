use candle_core::{Result, Tensor};
use candle_nn::Module;

#[derive(Debug, Clone)]
pub struct CustomLayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl CustomLayerNorm {
    pub fn new(weight: Tensor, bias: Tensor, eps: f64) -> Self {
        Self { weight, bias, eps }
    }

    /// PyTorch-compatible LayerNorm implementation with better numerical stability
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let original_dtype = x.dtype();

        // Convert to F32 for higher precision computation
        let x_f32 = x; //.to_dtype(DType::F32)?;

        // Get the last dimension for normalization
        let shape = x_f32.shape();
        let last_dim = shape.dims().len() - 1;
        let normalized_shape = shape.dims()[last_dim];

        // Calculate mean along the last dimension
        let mean = x_f32.mean_keepdim(last_dim)?;

        // Center the input
        let centered = x_f32.broadcast_sub(&mean)?;

        // Calculate variance: E[(x - mean)^2]
        let variance = centered.sqr()?.mean_keepdim(last_dim)?;

        // Add epsilon to variance for numerical stability
        let variance_eps = (variance + self.eps)?;

        // Calculate standard deviation
        let std = variance_eps.sqrt()?;

        // Normalize: (x - mean) / std
        let normalized = centered.broadcast_div(&std)?;

        // Apply learned parameters
        let result = normalized.broadcast_mul(&self.weight)?;
        let result = result.broadcast_add(&self.bias)?;

        // Convert back to original dtype
        result.to_dtype(original_dtype)
    }
}

impl Module for CustomLayerNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward(xs)
    }
}
