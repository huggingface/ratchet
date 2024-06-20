use std::io::{BufRead, Seek};

use ratchet::{prelude::shape, rvec, Device, Tensor};
use ratchet_loader::gguf::gguf::Header;
use ratchet_nn::{KVEntry, Linear, Module, RotaryEmbedding, RotaryInput};

#[cfg(target_arch = "wasm32")]
use crate::{ratchet_from_gguf_web, TensorMap};

#[derive(Debug)]
pub struct PhiSelfAttention {
    qkv: Linear,
    o: Linear,
    rope: RotaryEmbedding,
    n_heads: u32,
    softmax_scale: Tensor,
    n_kv_heads: u32,
}

impl PhiSelfAttention {
    pub fn load<R: BufRead + Seek>(
        disk_model: &Header,
        reader: &mut R,
        layer_index: usize,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let lt = |name: &str| {
            let key = format!("blk.{}.{}", layer_index, name);
            disk_model.tensor(reader, &key, device)
        };
        Self::load_inner(disk_model, lt, device)
    }

    #[cfg(target_arch = "wasm32")]
    pub fn from_web(
        header: &Header,
        tensors: &mut TensorMap,
        layer_index: usize,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let lt = |name: &str| {
            let key = format!("blk.{}.{}", layer_index, name);
            let tensor = tensors
                .remove(&key)
                .ok_or_else(|| anyhow::anyhow!("missing tensor"))?;
            ratchet_from_gguf_web(tensor, device)
        };
        Self::load_inner(header, lt, device)
    }

    fn load_inner<F>(header: &Header, mut lt: F, device: &Device) -> anyhow::Result<Self>
    where
        F: FnMut(&str) -> anyhow::Result<Tensor>,
    {
        let qkv = Linear::new(lt("attn_qkv.weight")?, None);
        let o = Linear::new(lt("attn_output.weight")?, None);

        let metadata = &header.metadata;
        let n_heads = metadata.get("phi3.attention.head_count")?.to_u32()?;
        let n_kv_heads = metadata.get("phi3.attention.head_count_kv")?.to_u32()?;
        let d_model = metadata.get("phi3.embedding_length")?.to_u32()?;
        let rope_base = 10000.0f32;
        let rope_dim = metadata.get("phi3.rope.dimension_count")?.to_u32()?;

        let hdim = d_model as f32 / n_heads as f32;
        let softmax_scale = Tensor::from_data([1.0 / hdim.sqrt()], shape![1], device.clone());
        let rope = RotaryEmbedding::new(rope_dim as _, false, rope_base, 1.0);
        Ok(Self {
            qkv,
            o,
            rope,
            n_heads,
            softmax_scale,
            n_kv_heads,
        })
    }
}

pub struct PhiAttnInput {
    pub input: Tensor,
    pub mask: Option<Tensor>,
    pub cache: Option<KVEntry>,
}

impl Module for PhiSelfAttention {
    type Input = PhiAttnInput;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let PhiAttnInput { input, mask, cache } = input;
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

        let offset = cache.as_ref().map(|kv| kv.entries).unwrap_or(0);
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

        let (key_states, value_states) = if let Some(kv) = cache {
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
