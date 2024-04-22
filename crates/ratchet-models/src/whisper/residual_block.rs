use super::{mha::*, mlp::MLP};
use ratchet::{Device, Tensor};
use ratchet_loader::gguf::gguf::Header;
use ratchet_nn::{KVEntry, LayerNorm, Linear, Module};
use std::io::{BufRead, Seek};

#[cfg(target_arch = "wasm32")]
use {crate::TensorMap, ratchet_loader::ratchet_from_gguf_web};

#[derive(Debug)]
pub struct ResidualAttentionBlock {
    attn_ln: LayerNorm,
    attn: MultiHeadAttention,
    x_attn_ln: Option<LayerNorm>,
    x_attn: Option<MultiHeadAttention>,
    mlp_ln: LayerNorm,
    mlp: MLP,
}

#[derive(Debug, derive_new::new)]
pub struct ResidualAttentionBlockInputs {
    pub x: Tensor,
    pub xa: Option<Tensor>,
    pub mask: Option<Tensor>,
    pub cache: Option<KVEntry>,
}

impl Module for ResidualAttentionBlock {
    type Input = ResidualAttentionBlockInputs;
    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let ResidualAttentionBlockInputs { x, xa, mask, cache } = input;

        let attn_ln = self.attn_ln.schedule(x.clone())?;
        let self_attn =
            self.attn
                .schedule(MHAInputs::new(attn_ln, None, mask.clone(), cache, true))?;

        let mut attn = x.add(self_attn)?;

        if let Some(ref xa_blck) = self.x_attn {
            if let Some(xa_ln) = &self.x_attn_ln {
                let x_attn_ln = xa_ln.schedule(attn.clone())?;
                let x_attn =
                    xa_blck.schedule(MHAInputs::new(x_attn_ln, xa.clone(), None, None, false))?;
                attn = x_attn.add(attn.clone())?;
            }
        }
        let mlp_ln = self.mlp_ln.schedule(attn.clone())?;
        let mlp = self.mlp.schedule(mlp_ln)?;
        mlp.add(attn)
    }
}

impl ResidualAttentionBlock {
    pub fn load<R: BufRead + Seek>(
        header: &Header,
        reader: &mut R,
        layer_index: usize,
        n_heads: usize,
        prefix: &str,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let lt = |name: &str| {
            let key = format!("model.{}.layers.{}.{}", prefix, layer_index, name);
            header.tensor(reader, &key, device)
        };
        Self::load_inner(lt, prefix, n_heads)
    }

    #[cfg(target_arch = "wasm32")]
    pub fn from_web(
        header: &Header,
        tensor_map: &mut TensorMap,
        layer_index: usize,
        n_heads: usize,
        prefix: &str,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let mut lt = |name: &str| {
            let key = format!("model.{}.layers.{}.{}", prefix, layer_index, name);
            let tensor = tensor_map
                .remove(&key)
                .ok_or_else(|| anyhow::anyhow!("missing tensor"))?;
            ratchet_from_gguf_web(tensor, device)
        };
        Self::load_inner(lt, prefix, n_heads)
    }

    pub fn load_inner<F>(mut lt: F, prefix: &str, n_heads: usize) -> anyhow::Result<Self>
    where
        F: FnMut(&str) -> anyhow::Result<Tensor>,
    {
        let attn_ln = LayerNorm::new(
            lt("self_attn_layer_norm.weight")?,
            Some(lt("self_attn_layer_norm.bias")?),
            1e-5,
        );
        //model.encoder.layers.0.self_attn.v_proj.weight
        let attn = MultiHeadAttention::new(
            Linear::new(
                lt("self_attn.q_proj.weight")?,
                Some(lt("self_attn.q_proj.bias")?),
            ),
            Linear::new(lt("self_attn.k_proj.weight")?, None),
            Linear::new(
                lt("self_attn.v_proj.weight")?,
                Some(lt("self_attn.v_proj.bias")?),
            ),
            Linear::new(
                lt("self_attn.out_proj.weight")?,
                Some(lt("self_attn.out_proj.bias")?),
            ),
            n_heads,
        );

        let (x_attn_ln, x_attn) = if prefix == "decoder" {
            let x_attn_ln = LayerNorm::new(
                lt("encoder_attn_layer_norm.weight")?,
                Some(lt("encoder_attn_layer_norm.bias")?),
                1e-5,
            );
            let x_attn = MultiHeadAttention::new(
                Linear::new(
                    lt("encoder_attn.q_proj.weight")?,
                    Some(lt("encoder_attn.q_proj.bias")?),
                ),
                Linear::new(lt("encoder_attn.k_proj.weight")?, None),
                Linear::new(
                    lt("encoder_attn.v_proj.weight")?,
                    Some(lt("encoder_attn.v_proj.bias")?),
                ),
                Linear::new(
                    lt("encoder_attn.out_proj.weight")?,
                    Some(lt("encoder_attn.out_proj.bias")?),
                ),
                n_heads,
            );
            (Some(x_attn_ln), Some(x_attn))
        } else {
            (None, None)
        };

        let mlp_ln = LayerNorm::new(
            lt("final_layer_norm.weight")?,
            Some(lt("final_layer_norm.bias")?),
            1e-5,
        );
        let mlp = MLP::new(
            Linear::new(lt("fc1.weight")?, Some(lt("fc1.bias")?)),
            Linear::new(lt("fc2.weight")?, Some(lt("fc2.bias")?)),
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
