use std::{
    io::{BufRead, Seek},
};

use ratchet::{Device, Tensor};
use ratchet_loader::{GGMLModel};
use ratchet_nn::{LayerNorm, Linear, Module};

use crate::{HyperParameters, MHAInputs, MultiHeadAttention, Whisper, MLP};

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
            disk_model.load_tensor(&key, reader, device)
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
        let ResidualAttentionBlockInputs { x, xa, mask } = input;

        let attn_ln = self.attn_ln.forward(x)?;
        let self_attn = self
            .attn
            .forward(&MHAInputs::new(attn_ln, None, mask.clone(), true))?;

        let mut attn = self_attn.add(x)?;

        if let Some(ref xa_blck) = self.x_attn {
            if let Some(xa_ln) = &self.x_attn_ln {
                let x_attn_ln = xa_ln.forward(&attn)?;
                let x_attn =
                    xa_blck.forward(&MHAInputs::new(x_attn_ln, xa.clone(), mask.clone(), false))?;
                attn = attn.add(&x_attn)?;
            }
        }

        let mlp_ln = self.mlp_ln.forward(&attn)?;
        let mlp = self.mlp.forward(&mlp_ln)?;
        mlp.add(&attn)
    }
}

impl ResidualAttentionBlock {
    pub fn load<R: BufRead + Seek>(
        disk_model: &GGMLModel<Whisper>,
        layer_index: usize,
        n_heads: usize,
        enable_x_attn: bool,
        reader: &mut R,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let mut lt = |name: &str| {
            let key = format!("encoder.blocks.{}.{}", layer_index, name);
            disk_model.load_tensor(&key, reader, device)
        };
        let attn_ln = LayerNorm::new(lt("attn_ln.weight")?, Some(lt("attn_ln.bias")?), 1e-5);
        let attn = MultiHeadAttention::new(
            Linear::new(lt("attn.query.weight")?, Some(lt("attn.query.bias")?)),
            Linear::new(lt("attn.key.weight")?, None),
            Linear::new(lt("attn.value.weight")?, Some(lt("attn.value.bias")?)),
            Linear::new(lt("attn.out.weight")?, Some(lt("attn.out.bias")?)),
            n_heads,
        );
        let (x_attn_ln, x_attn) = if enable_x_attn {
            let x_attn_ln = LayerNorm::new(
                lt("cross_attn_ln.weight")?,
                Some(lt("cross_attn_ln.bias")?),
                1e-5,
            );
            let x_attn = MultiHeadAttention::new(
                Linear::new(
                    lt("cross_attn.query.weight")?,
                    Some(lt("cross_attn.query.bias")?),
                ),
                Linear::new(lt("cross_attn.key.weight")?, None),
                Linear::new(
                    lt("cross_attn.value.weight")?,
                    Some(lt("cross_attn.value.bias")?),
                ),
                Linear::new(
                    lt("cross_attn.out.weight")?,
                    Some(lt("cross_attn.out.bias")?),
                ),
                n_heads,
            );
            (Some(x_attn_ln), Some(x_attn))
        } else {
            (None, None)
        };

        let mlp_ln = LayerNorm::new(lt("mlp_ln.weight")?, Some(lt("mlp_ln.bias")?), 1e-5);
        let mlp = MLP::new(
            Linear::new(lt("mlp.0.weight")?, Some(lt("mlp.0.bias")?)),
            Linear::new(lt("mlp.2.weight")?, Some(lt("mlp.2.bias")?)),
        );
        Ok(Self {
            attn_ln,
            attn,
            x_attn_ln,
            x_attn,
            mlp_ln,
            mlp,
        })
    }
}

pub struct WhisperEncoder {
    stem: EncoderStem,
    blocks: Vec<ResidualAttentionBlock>,
    ln_post: LayerNorm,
}

impl Module for WhisperEncoder {
    type Input = Tensor;

    fn forward(&self, input: &Self::Input) -> anyhow::Result<Tensor> {
        let mut x = self.stem.forward(input)?;
        for block in &self.blocks {
            x = block.forward(&ResidualAttentionBlockInputs {
                x,
                xa: None,
                mask: None,
            })?;
        }
        self.ln_post.forward(&x)
    }
}

impl WhisperEncoder {
    pub fn load<R: BufRead + Seek>(
        disk_model: &GGMLModel<Whisper>,
        reader: &mut R,
        device: &Device,
        hparams: &HyperParameters,
    ) -> anyhow::Result<Self> {
        let stem = EncoderStem::load(disk_model, reader, device)?;
        let (n_layers, n_heads) = (hparams.n_audio_layer, hparams.n_audio_head);

        let blocks = (0..n_layers)
            .fold(Vec::with_capacity(n_layers as _), |mut blocks, i| {
                blocks.push(ResidualAttentionBlock::load(
                    disk_model,
                    i as _,
                    n_heads as _,
                    false,
                    reader,
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
            ln_post: LayerNorm::new(lt("weight")?, Some(lt("bias")?), 1e-5),
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::{EncoderStem, Whisper};
    use hf_hub::api::sync::Api;
    use ratchet::Device;
    use ratchet_loader::GGMLCompatible;

    #[test]
    fn load_ggml() {
        let api = Api::new().unwrap();
        let model = api.model("ggerganov/whisper.cpp".to_string());
        let path = model.get("ggml-tiny.bin").unwrap();

        let mut reader = std::io::BufReader::new(std::fs::File::open(path).unwrap());
        let gg_disk = Whisper::load_ggml(&mut reader).unwrap();
        assert_eq!(gg_disk.tensors.len(), 167);

        let stem = EncoderStem::load(&gg_disk, &mut reader, &Device::CPU).unwrap();
        println!("{:?}", stem);
    }
}
