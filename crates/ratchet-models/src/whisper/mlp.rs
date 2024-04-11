use ratchet::Tensor;
use ratchet_nn::{Module, RLinear};

#[derive(Debug, derive_new::new)]
pub struct MLP {
    l1: RLinear,
    l2: RLinear,
}

impl Module for MLP {
    type Input = Tensor;
    fn forward(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        self.l2.forward(self.l1.forward(input)?.gelu()?)
    }
}
