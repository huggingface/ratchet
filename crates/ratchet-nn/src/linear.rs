use ratchet::Tensor;

use crate::Module;

#[derive(Debug)] // Modified to remove derive_new for custom initialization logic
pub struct Linear {
    w: Tensor,
    w_t: Tensor, // Transposed weights
    b: Option<Tensor>,
}

impl Linear {
    // Custom constructor to initialize and transpose weights
    pub fn new(w: Tensor, b: Option<Tensor>) -> anyhow::Result<Self> {
        let w_t = w.permute(&[1, 0])?; // Transpose once during initialization
        Ok(Self { w, w_t, b })
    }
}

impl Module for Linear {
    type Input = Tensor;
    fn forward(&self, input: &Self::Input) -> anyhow::Result<Tensor> {
        let y = input.matmul(&self.w_t)?; // Use pre-transposed weights
        if let Some(b) = &self.b {
            y.add(b)
        } else {
            Ok(y)
        }
    }
}
