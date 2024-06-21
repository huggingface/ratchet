use ratchet::{rvec, shape, Device, Tensor};
use ratchet_nn::{
    Embedding, KVCache, KVEntry, LayerNorm, Linear, Module, RotaryEmbedding, RotaryInput,
};

use super::mlp::MLP;

#[derive(Debug, derive_new::new)]
pub struct SelfAttention {
    qkv: Linear,
    o: Linear,
    rope: RotaryEmbedding,
    n_heads: u32,
    softmax_scale: Tensor,
    n_kv_heads: u32,
}

pub struct AttnInput {
    pub input: Tensor,
    pub mask: Option<Tensor>,
    pub kv_cache: Option<KVEntry>,
}

impl Module for SelfAttention {
    type Input = AttnInput;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let AttnInput {
            input,
            mask,
            kv_cache,
        } = input;
        let [batch_size, q_len, n_state]: [usize; 3] = input.shape().try_into()?;

        let hdim = n_state / self.n_heads as usize;
        let kv_x_hdim = self.n_kv_heads as usize * hdim;

        let qkv = self.qkv.schedule(input)?;
        let query_pos = self.n_heads as usize * hdim;
        let key_pos = query_pos + kv_x_hdim;
        let value_pos = key_pos + kv_x_hdim;

        let query_states = qkv
            .clone()
            .slice(&[0..batch_size, 0..q_len, 0..query_pos])?;
        let key_states = qkv
            .clone()
            .slice(&[0..batch_size, 0..q_len, query_pos..key_pos])?;
        let value_states = qkv
            .clone()
            .slice(&[0..batch_size, 0..q_len, key_pos..value_pos])?;

        let q_shape = shape![batch_size as _, q_len, self.n_heads as _, hdim];
        let kv_shape = shape![batch_size as _, q_len, self.n_kv_heads as _, hdim];

        let query_states = query_states.view(q_shape)?.permute(&[0, 2, 1, 3])?;
        let key_states = key_states.view(kv_shape.clone())?.permute(&[0, 2, 1, 3])?;
        let value_states = value_states.view(kv_shape)?.permute(&[0, 2, 1, 3])?;

        let offset = kv_cache.as_ref().map(|kv| kv.entries).unwrap_or(0);
        let q_dt = query_states.dt();
        let query_states = self
            .rope
            .schedule(RotaryInput {
                input: query_states.full()?,
                offset,
            })?
            .cast(q_dt)?;
        let key_states = self
            .rope
            .schedule(RotaryInput {
                input: key_states.full()?,
                offset,
            })?
            .cast(q_dt)?;

        let (key_states, value_states) = if let Some(kv) = kv_cache {
            let k_cache = kv.k_cache.cache(key_states, 2, offset)?;
            let v_cache = kv.v_cache.cache(value_states, 2, offset)?;
            (k_cache, v_cache)
        } else {
            (key_states, value_states)
        };

        let mut attn_weights = query_states
            .full()?
            .matmul(key_states.full()?, false, true)?
            .mul(self.softmax_scale.clone())?
            .cast(q_dt)?;

        if let Some(m) = mask {
            let attn_dt = attn_weights.dt();
            attn_weights = attn_weights.add(m.cast(attn_dt)?)?;
        }

        let w = attn_weights.full()?.softmax(3)?.cast(value_states.dt())?;
        let wv = w
            .matmul(value_states, false, false)?
            .permute(&[0, 2, 1, 3])?;
        let wv = wv.view(shape![batch_size as _, q_len, n_state])?;
        self.o.schedule(wv)
    }
}

#[derive(Debug, derive_new::new)]
pub struct DecoderLayer {
    pub ln: LayerNorm,
    pub self_attn: SelfAttention,
    pub mlp: MLP,
}

#[derive(Debug)]
pub struct DecoderLayerInput {
    pub x: Tensor,
    pub mask: Option<Tensor>,
    pub kv_cache: Option<KVEntry>,
}

impl Module for DecoderLayer {
    type Input = DecoderLayerInput;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let DecoderLayerInput { x, mask, kv_cache } = input;
        let residual = x.clone();
        let xs = self.ln.schedule(x)?;
        let attn_output = self.self_attn.schedule(AttnInput {
            input: xs.clone(),
            mask,
            kv_cache,
        })?;
        let ff_hs = self.mlp.schedule(xs)?;
        attn_output.add(ff_hs)?.add(residual)
    }
}

#[derive(Debug, derive_new::new)]
pub struct TextModel {
    pub embedding: Embedding,
    pub layers: Vec<DecoderLayer>,
    pub ln_post: LayerNorm,
    pub lm_head: Linear,
    pub kv_cache: KVCache,
    pub device: Device,
}

impl Module for TextModel {
    type Input = Tensor;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let mut x = input.clone();
        let [_, seq_len, n_state]: [usize; 3] = x.shape().try_into()?;
        let mask = if seq_len <= 1 {
            None
        } else {
            Some(Self::generate_mask(seq_len, x.device())?)
        };

        for (i, layer) in self.layers.iter().enumerate() {
            let input = DecoderLayerInput {
                x,
                mask: mask.clone(),
                kv_cache: Some(self.kv_cache[i].clone()),
            };
            x = layer.schedule(input)?;
        }
        x = self.ln_post.schedule(x)?;
        x = x.slice(&[0..1, seq_len - 1..seq_len, 0..n_state])?;
        let logits = self.lm_head.schedule(x)?;
        Ok(logits)
    }
}

impl TextModel {
    pub fn generate_mask(seq_len: usize, device: &Device) -> anyhow::Result<Tensor> {
        let mask: Vec<_> = (0..seq_len)
            .flat_map(|i| (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0f32 }))
            .collect();

        Ok(Tensor::from_data(
            mask,
            shape![seq_len, seq_len],
            device.clone(),
        ))
    }

    pub fn cache_mut(&mut self) -> &mut KVCache {
        &mut self.kv_cache
    }
    pub fn reset(&mut self) {
        self.kv_cache.reset();
    }
}
