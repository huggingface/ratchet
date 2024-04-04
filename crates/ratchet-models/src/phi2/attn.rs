use std::io::{BufRead, Seek};

use ratchet::{prelude::shape, Device, Tensor};
use ratchet_loader::gguf::gguf::Content;
use ratchet_nn::{Linear, Module, RotaryEmbedding};

#[derive(Debug)]
pub struct PhiSelfAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    rope: RotaryEmbedding,
    n_heads: u32,
    softmax_scale: Tensor,
    n_kv_heads: u32,
}

impl PhiSelfAttention {
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
        let n_kv_heads = disk_model
            .metadata
            .get("phi2.attention.head_count_kv")
            .unwrap()
            .to_u32()?;
        //1 / head_dim
        let softmax_scale =
            Tensor::from_data([1.0 / (80 as f32).sqrt()], shape![1], device.clone());
        //TODO: hardcoded for Phi2, should read from meta
        let base = 10000.0;
        let dim = (0.4 * (2560f64 / 32f64)) as usize;
        let rope = RotaryEmbedding::new(dim, false, base, 1.0);
        Ok(Self {
            q,
            k,
            v,
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
}

impl Module for PhiSelfAttention {
    type Input = PhiAttnInput;

    fn forward(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let PhiAttnInput { input, mask } = input;
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

        let query_states = self.rope.forward(qs)?;
        let key_states = self.rope.forward(ks)?;

        //TODO: can we just use the built in transposed matmul?
        let mut attn_weights = query_states
            .matmul(key_states.permute(&[0, 1, 3, 2])?, false, false)?
            .mul(self.softmax_scale.clone())?;

        assert_eq!(
            attn_weights.shape(),
            &shape![batch_size as _, self.n_heads as _, seq_len, seq_len]
        );

        if let Some(m) = mask {
            attn_weights = attn_weights.add(m)?;
        }

        let w = attn_weights.softmax(3)?;
        let wv = w.matmul(vs, false, false)?.permute(&[0, 2, 1, 3])?;
        let wv = wv.view(shape![batch_size as _, seq_len, n_state])?;
        self.o.forward(wv)
    }
}
