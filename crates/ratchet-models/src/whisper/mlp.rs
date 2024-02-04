use ratchet::Tensor;
use ratchet_nn::{Linear, Module};

#[derive(Debug, derive_new::new)]
pub struct MLP {
    l1: Linear,
    l2: Linear,
}

impl Module for MLP {
    fn forward(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let x = self.l1.forward(x)?;
        self.l2.forward(&x.gelu()?)
    }
}
