use std::io::{BufRead, Seek};

use ratchet::{Device, Tensor};
use ratchet_loader::gguf::gguf::Header;
use ratchet_nn::{Embedding, Module};
use tokenizers::Tokenizer;

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
    tokenizer: Tokenizer,
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

struct BertConfig {
    pub n_layers: u32,
    pub ctx_len: u32,
    pub embedding_len: u32,
    pub ff_len: u32,
    pub n_attn_head: u32,
    pub layer_norm_epsilon: f32,
}

impl BertConfig {
    pub fn from_header(header: &Header) -> anyhow::Result<BertConfig> {
        let n_layers = header.metadata.get("bert.block_count")?.to_u32()?;
        let ctx_len = header.metadata.get("bert.context_length")?.to_u32()?;
        let embedding_len = header.metadata.get("bert.embedding_length")?.to_u32()?;
        let ff_len = header.metadata.get("bert.feed_forward_length")?.to_u32()?;
        let n_attn_head = header.metadata.get("bert.attention.head_count")?.to_u32()?;
        let layer_norm_epsilon = header
            .metadata
            .get("bert.attention.layer_norm_epsilon")?
            .to_f32()?;
        Ok(Self {
            n_layers,
            ctx_len,
            embedding_len,
            ff_len,
            n_attn_head,
            layer_norm_epsilon,
        })
    }
}
impl Bert {
    pub fn load_gguf<R: BufRead + Seek>(
        header: Header,
        reader: &mut R,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let config: BertConfig = BertConfig::from_header(&header)?;

        let embedding = Embedding::new(header.tensor(reader, "token_embd.weight", device)?);

        anyhow::bail!("todo")
    }
}
