use std::io::{BufRead, Seek};

use ratchet::{prelude::shape, Device, Tensor};
use ratchet_loader::gguf::gguf::Content;
use ratchet_nn::Linear;

#[derive(Debug)]
pub struct SelfAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
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
        Ok(Self {
            q,
            k,
            v,
            o,
            n_heads,
            softmax_scale,
            n_kv_heads: n_heads,
        })
    }
}
