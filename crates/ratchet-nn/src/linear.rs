use ratchet::Tensor;

use crate::Module;

#[derive(Debug)]
pub struct Linear {
    w: Tensor,
    w_t: Tensor,
    b: Option<Tensor>,
}

impl Linear {
    pub fn new(w: Tensor, b: Option<Tensor>) -> anyhow::Result<Self> {
        let w_t = w.permute(&[1, 0])?;
        Ok(Self { w, w_t, b })
    }
}

impl Module for Linear {
    type Input = Tensor;
    fn forward(&self, input: &Self::Input) -> anyhow::Result<Tensor> {
        let y = input.matmul(&self.w_t)?;
        if let Some(b) = &self.b {
            y.add(b)
        } else {
            Ok(y)
        }
    }
}
