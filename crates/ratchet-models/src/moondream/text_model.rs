use std::io::{BufRead, Seek};

use ratchet::{rvec, shape, Device, Tensor};
use ratchet_loader::gguf::gguf::Header;
use ratchet_nn::{
    Embedding, KVCache, KVEntry, LayerNorm, Linear, Module, RotaryEmbedding, RotaryInput,
};

use super::mlp::MLP;

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
    fn load_inner<F>(header: &Header, mut lt: F, device: &Device) -> anyhow::Result<Self>
    where
        F: FnMut(&str) -> anyhow::Result<Tensor>,
    {
        let qkv = Linear::new(lt("attn_qkv.weight")?, None);
        let o = Linear::new(lt("attn_output.weight")?, None);

        let metadata = &header.metadata;
        let n_heads = metadata.get("phi2.attention.head_count")?.to_u32()?;
        let n_kv_heads = metadata.get("phi2.attention.head_count_kv")?.to_u32()?;
        let d_model = metadata.get("phi2.embedding_length")?.to_u32()?;
        let rope_base = 10000.0f32;
        let rope_dim = metadata.get("phi2.rope.dimension_count")?.to_u32()?;

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
        let query_states = self.rope.schedule(RotaryInput {
            input: query_states,
            offset,
        })?;
        let key_states = self.rope.schedule(RotaryInput {
            input: key_states,
            offset,
        })?;

        let (key_states, value_states) = if let Some(kv) = cache {
            let k_cache = kv.k_cache.cache(key_states, 2, offset)?;
            let v_cache = kv.v_cache.cache(value_states, 2, offset)?;
            (k_cache, v_cache)
        } else {
            (key_states, value_states)
        };

        let mut attn_weights = query_states
            .matmul(key_states, false, true)?
            .mul(self.softmax_scale.clone())?;

        if let Some(m) = mask {
            attn_weights = attn_weights.add(m)?;
        }

        let w = attn_weights.softmax(3)?;
        let wv = w
            .matmul(value_states, false, false)?
            .permute(&[0, 2, 1, 3])?;
        let wv = wv.view(shape![batch_size as _, q_len, n_state])?;
        self.o.schedule(wv)
    }
}

#[derive(Debug)]
pub struct DecoderLayer {
    ln: LayerNorm,
    self_attn: PhiSelfAttention,
    mlp: MLP,
}

impl DecoderLayer {
    pub fn load<R: BufRead + Seek>(
        disk_model: &Header,
        reader: &mut R,
        layer_index: usize,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let self_attn = PhiSelfAttention::load(disk_model, reader, layer_index, device)?;
        let mut lt = |name: &str| {
            let key = format!("blk.{}.{}", layer_index, name);
            disk_model.tensor(reader, &key, device)
        };

        let ln = LayerNorm::new(lt("attn_norm.weight")?, Some(lt("attn_norm.bias")?), 1e-5);

        let mlp = MLP::new(
            Linear::new(lt("ffn_up.weight")?, Some(lt("ffn_up.bias")?)),
            Linear::new(lt("ffn_down.weight")?, Some(lt("ffn_down.bias")?)),
        );
        Ok(Self { ln, self_attn, mlp })
    }
}

pub struct DecoderLayerInput {
    pub x: Tensor,
    pub mask: Option<Tensor>,
    pub cache: Option<KVEntry>,
}

impl Module for DecoderLayer {
    type Input = DecoderLayerInput;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let DecoderLayerInput { x, mask, cache } = input;
        let residual = x.clone();
        let xs = self.ln.schedule(x)?;
        let attn_output = self.self_attn.schedule(PhiAttnInput {
            input: xs.clone(),
            mask,
            cache,
        })?;
        let ff_hs = self.mlp.schedule(xs)?;
        attn_output.add(ff_hs)?.add(residual)
    }
}

#[derive(Debug)]
pub struct Phi2 {
    pub embedding: Embedding,
    pub layers: Vec<DecoderLayer>,
    pub ln_post: LayerNorm,
    pub lm_head: Linear,
    pub kv_cache: KVCache,
    pub device: Device,
}

impl Module for Phi2 {
    type Input = Tensor;

    fn schedule(&self, embedding: Self::Input) -> anyhow::Result<Tensor> {
        let mut x = embedding.clone();
        let [_, seq_len, n_state]: [usize; 3] = x.shape().try_into()?;
        let mask = if seq_len <= 1 {
            None
        } else {
            Some(Self::generate_mask(seq_len, x.device())?)
        };

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let input = DecoderLayerInput {
                x,
                mask: mask.clone(),
                cache: Some(self.kv_cache[layer_idx].clone()),
            };
            x = layer.schedule(input)?;
        }
        x = self.ln_post.schedule(x)?;
        x = x.slice(&[0..1, seq_len - 1..seq_len, 0..n_state])?;
        let logits = self.lm_head.schedule(x)?;
        Ok(logits)
    }
}

impl Phi2 {
    const MAX_CACHE: usize = 1024;

    pub fn load<R: BufRead + Seek>(
        header: Header,
        reader: &mut R,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let embedding = Embedding::new(header.tensor(reader, "token_embd.weight", device)?);

        let n_layers = header.metadata.get("phi2.block_count").unwrap().to_u32()? as i32;

        let layers = (0..n_layers)
            .fold(Vec::with_capacity(n_layers as _), |mut blocks, i| {
                blocks.push(DecoderLayer::load(&header, reader, i as _, device));
                blocks
            })
            .into_iter()
            .collect::<anyhow::Result<Vec<_>>>()?;

        let mut lt = |name: &str| {
            let key = format!("output{}", name);
            header.tensor(reader, &key, device)
        };

        let ln_post = LayerNorm::new(lt("_norm.weight")?, Some(lt("_norm.bias")?), 1e-5);
        let lm_head = Linear::new(lt(".weight")?, Some(lt(".bias")?));

        Ok(Self {
            embedding,
            layers,
            ln_post,
            lm_head,
            kv_cache: KVCache::new(n_layers, shape![1, 32, Self::MAX_CACHE, 64], device),
            device: device.clone(),
        })
    }

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

    pub fn reset(&mut self) {
        self.kv_cache.reset();
    }

    pub fn cache_mut(&mut self) -> &mut KVCache {
        &mut self.kv_cache
    }
}
