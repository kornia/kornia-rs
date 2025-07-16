use candle_core::{Result, Tensor, D};
use candle_nn::Module;

/// Custom RmsNorm implementation to match PyTorch exactly
#[derive(Clone, Debug)]
pub struct CustomRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl CustomRmsNorm {
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }
}

impl Module for CustomRmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Store original dtype to match PyTorch behavior
        let input_dtype = x.dtype();

        // Convert to F32 for calculations exactly like PyTorch LlamaRMSNorm
        let x_f32 = x.to_dtype(candle_core::DType::F32)?;

        // Calculate variance: mean of squares (PyTorch does pow(2).mean(-1, keepdim=True))
        let variance = x_f32.sqr()?.mean_keepdim(D::Minus1)?;

        // Normalize using rsqrt (reciprocal square root) like PyTorch
        let rsqrt_var = (1.0 / (variance + self.eps)?.sqrt()?)?;
        let x_normed = x_f32.broadcast_mul(&rsqrt_var)?;

        // Multiply by weight and return in original dtype
        // let weight_f32 = self.weight.to_dtype(candle_core::DType::F32)?;
        // x_normed.broadcast_mul(&weight_f32)?.to_dtype(input_dtype)
        self.weight.broadcast_mul(&x_normed.to_dtype(input_dtype)?)
    }
}
