use ratchet::{shape, Device, StorageView, Tensor};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RotaryEmbeddingConfig {
    pub theta: f32,
}

impl Default for RotaryEmbeddingConfig {
    fn default() -> Self {
        Self { theta: 1e-5 }
    }
}

#[derive(Clone, Debug)]
pub struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}
impl RotaryEmbedding {
    // https://github.com/facebookresearch/llama/blob/1076b9c51c77ad06e9d7ba8a4c6df775741732bd/llama/model.py#L47
    // https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/quantized_llama.rs#L278
    /// Precompute thetas
    fn precompute_freqs_cis(
        dim: u32,
        end: u32,
        theta: f32,
        device: Device,
    ) -> anyhow::Result<(Tensor, Tensor)> {
        let inv_freq = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / theta.powf(i as f32 / dim as f32))
            .collect::<Vec<f32>>();

        let inv_freq_len = inv_freq.len();
        let inv_freq =
            Tensor::from_data(inv_freq.as_slice(), shape![1, inv_freq_len], device.clone());
        let t = (0..end).map(|i| i as f32).collect::<Vec<f32>>();
        let t_tensor = Tensor::from_data(t.as_slice(), shape![end as usize], device.clone());
        let freqs = t_tensor.matmul(inv_freq, false, false)?;

        let cos = freqs.clone().cos()?;
        let sin = freqs.sin()?;
        println!("COS SHAPE: {:?}", cos.shape());
        println!("SIN SHAPE: {:?}", sin.shape());
        Ok((cos, sin))
    }

    pub fn new(dim: u32, end: u32, theta: f32, device: Device) -> anyhow::Result<RotaryEmbedding> {
        let (cos, sin) = RotaryEmbedding::precompute_freqs_cis(dim, end, theta, device)?;
        Ok(Self { cos, sin })
    }

    pub fn apply_rotary_embedding(&self, x: &Tensor, index_pos: usize) -> anyhow::Result<Tensor> {
        let [batch_size, n_heads, seq_len, n_embeddings] = x.shape().try_into()?;

        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope() -> anyhow::Result<()> {
        let rope = RotaryEmbedding::new(128, 512, 1e-5, Device::CPU)?;
        Ok(())
    }
}
