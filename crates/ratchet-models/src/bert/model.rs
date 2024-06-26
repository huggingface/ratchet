use std::io::{BufRead, Seek};

use ratchet::{Device, Tensor};
use ratchet_nn::Module;

pub struct EncoderLayer {}

impl EncoderLayer {
    pub fn load<R: BufRead + Seek>(device: &Device) -> anyhow::Result<Self> {
        Ok(Self {})
    }
}

pub struct BertInput {
    pub input_ids: Tensor,
    pub token_type_ids: Tensor,
}

pub struct BertEmbeddings {}

impl Module for BertEmbeddings {
    type Input = BertInput;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        todo!()
    }
}

pub struct BertEncoder {
    pub layers: Vec<EncoderLayer>,
}
impl Module for BertEncoder {
    type Input = Tensor;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        todo!()
    }
}

pub struct Bert {
    encoder: BertEncoder,
    embeddings: BertEmbeddings,
    pub device: Device,
}

impl Module for Bert {
    type Input = BertInput;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let embedding_output: Tensor = self.embeddings.schedule(input)?;
        let sequence_output: Tensor = self.encoder.schedule(embedding_output)?;
        Ok(sequence_output)
    }
}
