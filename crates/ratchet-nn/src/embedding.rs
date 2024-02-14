use ratchet::{shape, Tensor};

use crate::Module;

#[derive(Debug, derive_new::new)]
pub struct Embedding {
    pub weight: Tensor,
}

impl Module for Embedding {
    type Input = Tensor;

    fn forward(&self, input: &Self::Input) -> anyhow::Result<Tensor> {
        let mut output_shape = input.shape().clone();
        let weight_rank = self.weight.rank();
        output_shape.push(self.weight.shape()[weight_rank - 1]);

        let flat = input.view(shape![input.shape().numel()])?;
        let indexed = self.weight.index_select(&flat, 0)?;
        let x = indexed.view(output_shape)?;
        Ok(x)
    }
}
