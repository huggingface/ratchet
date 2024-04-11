use ratchet::Tensor;
use ratchet_nn::{Module, RLinear};

#[derive(Debug, derive_new::new)]
pub struct MLP {
    l1: RLinear,
    l2: RLinear,
}

impl Module for MLP {
    type Input = Tensor;
    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        self.l2.schedule(self.l1.schedule(input)?.gelu()?)
    }
}
