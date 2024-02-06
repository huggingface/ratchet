use std::{
    collections::HashMap,
    io::{BufRead, Seek},
};

use ratchet::{Device, Tensor};
use ratchet_loader::{GGMLModel, TensorHeader};
use ratchet_nn::{LayerNorm, Module};

use crate::{MultiHeadAttention, Whisper, MLP};

#[derive(Debug, derive_new::new)]
struct ConvBlock {
    w: Tensor,
    b: Tensor,
    stride: usize,
    padding: usize,
}

impl Module for ConvBlock {
    type Input = Tensor;

    fn forward(&self, input: &Self::Input) -> anyhow::Result<Tensor> {
        input
            .conv1d(&self.w, Some(&self.b), self.stride, self.padding)?
            .gelu()
    }
}

#[derive(Debug)]
pub(crate) struct EncoderStem {
    conv1: ConvBlock,
    conv2: ConvBlock,
    pos_embed: Tensor,
}

impl Module for EncoderStem {
    type Input = Tensor;

    fn forward(&self, input: &Self::Input) -> anyhow::Result<Tensor> {
        let convolved = self.conv2.forward(&self.conv1.forward(input)?)?;
        convolved.permute(&[0, 2, 1])?.add(&self.pos_embed)
    }
}

impl EncoderStem {
    pub fn load<R: BufRead + Seek>(
        disk_model: &GGMLModel<Whisper>,
        reader: &mut R,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let mut lt = |name: &str| {
            let key = format!("encoder.{}", name);
            disk_model.load_tensor(&key, reader)
        };

        Ok(Self {
            conv1: ConvBlock::new(lt("conv1.weight")?, lt("conv1.bias")?, 1, 1),
            conv2: ConvBlock::new(lt("conv2.weight")?, lt("conv2.bias")?, 2, 1),
            pos_embed: lt("positional_embedding")?,
        })
    }
}

pub struct ResidualAttentionBlock {
    attn_ln: LayerNorm,
    attn: MultiHeadAttention,
    x_attn_ln: Option<LayerNorm>,
    x_attn: Option<MultiHeadAttention>,
    mlp_ln: LayerNorm,
    mlp: MLP,
}

pub struct ResidualAttentionBlockInputs {
    x: Tensor,
    xa: Option<Tensor>,
    mask: Option<Tensor>,
}

impl Module for ResidualAttentionBlock {
    type Input = ResidualAttentionBlockInputs;
    fn forward(&self, input: &Self::Input) -> anyhow::Result<Tensor> {
        todo!()
    }
}
