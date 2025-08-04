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
        let input_dtype = x.dtype();

        let x_f32 = x.to_dtype(candle_core::DType::F32)?;

        let variance = x_f32.sqr()?.mean_keepdim(D::Minus1)?;

        let rsqrt_var = (1.0 / (variance + self.eps)?.sqrt()?)?;
        let x_normed = x_f32.broadcast_mul(&rsqrt_var)?;

        self.weight.broadcast_mul(&x_normed.to_dtype(input_dtype)?)
    }
}
