use ratchet::Tensor;

use crate::Module;

#[derive(Debug)]
pub struct Linear {
    w: Tensor,
    b: Option<Tensor>,
}

impl Linear {
    pub fn new(w: Tensor, b: Option<Tensor>) -> anyhow::Result<Self> {
        Ok(Self { w, b })
    }

    pub fn w_t(&self) -> anyhow::Result<Tensor> {
        self.w.permute(&[1, 0])
    }
}

impl Module for Linear {
    type Input = Tensor;
    fn forward(&self, input: &Self::Input) -> anyhow::Result<Tensor> {
        let w_t = self.w_t()?;
        let y = input.matmul(&w_t)?;
        if let Some(b) = &self.b {
            y.add(b)
        } else {
            Ok(y)
        }
    }
}
