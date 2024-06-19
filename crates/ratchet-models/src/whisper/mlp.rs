use ratchet::Tensor;
use ratchet_nn::{Linear, Module};

#[derive(Debug, derive_new::new)]
pub struct MLP {
    l1: Linear,
    l2: Linear,
}

impl MLP {
    pub fn activation_dt(&self) -> ratchet::DType {
        self.l1.w.dt().activation_dt()
    }
}

impl Module for MLP {
    type Input = Tensor;
    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let input_dt = input.dt();
        self.l2
            .schedule(self.l1.schedule(input)?.full()?.gelu()?.cast(input_dt)?)
    }
}
