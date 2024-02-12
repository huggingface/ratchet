use ratchet::Tensor;

use crate::Module;

#[derive(Debug, derive_new::new)]
pub struct Embedding {
    pub weight: Tensor,
}

impl Module for Embedding {
    type Input = Tensor;

    fn forward(&self, _input: &Self::Input) -> anyhow::Result<Tensor> {
        //self.weight.index_select(0, input)
        todo!()
    }
}
