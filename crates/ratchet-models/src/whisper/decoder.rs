use std::io::{BufRead, Seek};

use ratchet::prelude::*;
use ratchet_loader::GGMLModel;
use ratchet_nn::{Embedding, LayerNorm, Module};

use crate::{ResidualAttentionBlock, ResidualAttentionBlockInputs, Whisper};

#[derive(Debug)]
pub(crate) struct DecoderStem {
    pub token_embed: Embedding,
    pub pos_embed: Tensor,
}

impl DecoderStem {
    pub fn load<R: BufRead + Seek>(
        disk_model: &GGMLModel<Whisper>,
        reader: &mut R,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let mut lt = |name: &str| {
            let key = format!("decoder.{}", name);
            disk_model.load_tensor(&key, reader, device)
        };

        Ok(Self {
            token_embed: Embedding::new(lt("token_embedding.weight")?),
            pos_embed: lt("positional_embedding")?,
        })
    }
}

impl Module for DecoderStem {
    type Input = Tensor;

    fn forward(&self, input: &Self::Input) -> anyhow::Result<Tensor> {
        let num_tokens = input.shape()[input.rank() - 1];
        self.token_embed
            .forward(input)?
            .add(&self.pos_embed.slice(&[0..num_tokens])?)
    }
}

#[derive(Debug)]
pub struct WhisperDecoder {
    stem: DecoderStem,
    blocks: Vec<ResidualAttentionBlock>,
    mask: Tensor,
    ln_post: LayerNorm,
}

impl Module for WhisperDecoder {
    type Input = [Tensor; 2];

    fn forward(&self, input: &Self::Input) -> anyhow::Result<Tensor> {
        let [audio_ctx, tokens] = input;
        let mut x = self.stem.forward(tokens)?;
        for block in &self.blocks {
            let block_input = ResidualAttentionBlockInputs {
                x,
                xa: Some(audio_ctx.clone()),
                mask: Some(self.mask.clone()),
            };
            x = block.forward(&block_input)?;
        }
        x = self.ln_post.forward(&x)?;
        let logits = x.matmul(&self.stem.token_embed.weight.permute(&[1, 0])?)?;
        Ok(logits)
    }
}

impl WhisperDecoder {
    fn load_mask(n_ctx: usize, device: &Device) -> Tensor {
        let mask: Vec<_> = (0..n_ctx)
            .flat_map(|i| (0..n_ctx).map(move |j| if j > i { f32::NEG_INFINITY } else { 0f32 }))
            .collect();
        Tensor::from_data(mask, shape![n_ctx, n_ctx], device.clone())
    }

    pub fn load<R: BufRead + Seek>(
        disk_model: &GGMLModel<Whisper>,
        reader: &mut R,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let hparams = &disk_model.header.hparams;
        let stem = DecoderStem::load(disk_model, reader, device)?;
        let (n_layers, n_heads) = (hparams.n_text_layer, hparams.n_text_head);

        let blocks = (0..n_layers)
            .fold(Vec::with_capacity(n_layers as _), |mut blocks, i| {
                blocks.push(ResidualAttentionBlock::load(
                    disk_model,
                    reader,
                    i as _,
                    n_heads as _,
                    "encoder",
                    false,
                    device,
                ));
                blocks
            })
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;

        let mut lt = |name: &str| {
            let key = format!("encoder.ln_post.{}", name);
            disk_model.load_tensor(&key, reader, device)
        };

        Ok(Self {
            stem,
            blocks,
            mask: Self::load_mask(hparams.n_text_ctx as _, device),
            ln_post: LayerNorm::new(lt("weight")?, Some(lt("bias")?), 1e-5),
        })
    }
}
