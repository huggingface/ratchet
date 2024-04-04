use std::io::{BufRead, Seek};

use ratchet::{prelude::shape, Device, Tensor};
use ratchet_loader::gguf::gguf::Content;
use ratchet_nn::{Linear, Module, RotaryEmbedding};

#[derive(Debug)]
pub struct SelfAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    rope: RotaryEmbedding,
    n_heads: u32,
    softmax_scale: Tensor,
    n_kv_heads: u32,
}

impl SelfAttention {
    pub fn load<R: BufRead + Seek>(
        disk_model: &Content,
        reader: &mut R,
        layer_index: usize,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let mut lt = |name: &str| {
            let key = format!("blk.{}.{}", layer_index, name);
            disk_model.tensor(reader, &key, device)
        };
        let q = Linear::new(lt("attn_q.weight")?, Some(lt("attn_q.bias")?), true);
        let k = Linear::new(lt("attn_k.weight")?, Some(lt("attn_k.bias")?), true);
        let v = Linear::new(lt("attn_v.weight")?, Some(lt("attn_v.bias")?), true);
        let o = Linear::new(
            lt("attn_output.weight")?,
            Some(lt("attn_output.bias")?),
            true,
        );

        let n_heads = disk_model
            .metadata
            .get("phi2.attention.head_count")
            .unwrap()
            .to_u32()?;
        let softmax_scale =
            Tensor::from_data([1.0 / (n_heads as f32).sqrt()], shape![1], device.clone());
        //TODO: hardcoded for Phi2, should read from meta
        let rope_theta = 10000.0;
        let dim = (0.4 * (2560f64 / 32f64)) as usize;
        let max_position_embeddings = 2048;
        let rope = RotaryEmbedding::new(dim, max_position_embeddings, rope_theta, device.clone())?;
        Ok(Self {
            q,
            k,
            v,
            o,
            rope,
            n_heads,
            softmax_scale,
            n_kv_heads: n_heads,
        })
    }
}

impl Module for SelfAttention {
    type Input = Tensor;

    fn forward(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let [batch_size, seq_len, n_state]: [usize; 3] = input.shape().try_into()?;
        let q = self.q.forward(input.clone())?;
        let k = self.k.forward(input.clone())?;
        let v = self.v.forward(input.clone())?;

        let h_dim = n_state / self.n_heads as usize;
        //we don't support qk ln
        let q_shape = shape![batch_size as _, seq_len, self.n_heads as _, h_dim];
        let kv_shape = shape![batch_size as _, seq_len, self.n_kv_heads as _, h_dim];
        let qs = q.view(q_shape)?.permute(&[0, 2, 1, 3])?;
        let ks = k.view(kv_shape.clone())?.permute(&[0, 2, 1, 3])?;
        let vs = v.view(kv_shape)?.permute(&[0, 2, 1, 3])?;

        let query_states = self.rope.apply_rotary_embedding(qs, 0)?;
        let key_states = self.rope.apply_rotary_embedding(ks, 0)?;

        /*
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        # Partial rotary embedding
        query_rot, query_pass = (
            query_states[..., : self.rotary_emb.dim],
            query_states[..., self.rotary_emb.dim :],
        )
        key_rot, key_pass = (
            key_states[..., : self.rotary_emb.dim],
            key_states[..., self.rotary_emb.dim :],
        )
        # [batch_size, seq_length, num_heads, head_dim // config.partial_rotary_factor]
        query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)

        # [batch_size, seq_length, num_heads, head_dim]
        query_states = torch.cat((query_rot, query_pass), dim=-1)
        key_states = torch.cat((key_rot, key_pass), dim=-1)
        */
        Ok(query_states)
    }
}
