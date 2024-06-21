use ratchet::Tensor;
use ratchet_nn::{Linear, Module};

#[derive(Debug, derive_new::new)]
pub struct MLP {
    pub fc1: Linear,
    pub fc2: Linear,
}

impl Module for MLP {
    type Input = Tensor;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let input_dt = input.dt();
        self.fc2
            .schedule(self.fc1.schedule(input)?.full()?.gelu()?.cast(input_dt)?)
    }
}
