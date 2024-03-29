use ratchet::{rvec, shape, Device, StorageView, Tensor};

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
    dim: usize,
    cos: Tensor,
    sin: Tensor,
}
impl RotaryEmbedding {
    pub fn new(
        dim: usize,
        max_position_embeddings: usize,
        theta: f32,
        device: Device,
    ) -> anyhow::Result<Self> {
        let inv_freq = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / theta.powf(i as f32 / dim as f32))
            .collect::<Vec<f32>>();

        let inv_freq_len = inv_freq.len();
        let inv_freq =
            Tensor::from_data(inv_freq.as_slice(), shape![1, inv_freq_len], device.clone());

        let t = (0..max_position_embeddings)
            .map(|i| i as f32)
            .collect::<Vec<_>>();
        let t_tensor = Tensor::from_data(t, shape![max_position_embeddings, 1], device.clone());
        let freqs = t_tensor.matmul(inv_freq, false, false)?;
        let emb = Tensor::cat(rvec![freqs.clone(), freqs], 1)?;
        Ok(Self {
            dim: dim as usize,
            cos: emb.clone().cos()?,
            sin: emb.sin()?,
        })
    }

    /*
     * fn apply_rotary_emb(&self, xs: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (_b_size, _num_heads, seq_len, _headdim) = xs.dims4()?;
        let xs_rot = xs.i((.., .., .., ..self.dim))?;
        let xs_pass = xs.i((.., .., .., self.dim..))?;
        let xs12 = xs_rot.chunk(2, D::Minus1)?;
        let (xs1, xs2) = (&xs12[0], &xs12[1]);
        let c = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let s = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let rotate_half = Tensor::cat(&[&xs2.neg()?, &xs1], D::Minus1)?;
        let xs_rot = (xs_rot.broadcast_mul(&c)? + rotate_half.broadcast_mul(&s)?)?;
        Tensor::cat(&[&xs_rot, &xs_pass], D::Minus1)
    }
     */
    pub fn apply_rotary_embedding(&self, xs: Tensor, offset: usize) -> anyhow::Result<Tensor> {
        let [bsz, n_heads, seq_len, hdim] = xs.shape().try_into()?;

        let sin = self
            .sin
            .clone()
            .slice(&[offset..offset + seq_len, 0..self.dim])?;
        let cos = self
            .cos
            .clone()
            .slice(&[offset..offset + seq_len, 0..self.dim])?;

        let mut x_rot = xs
            .clone()
            .slice(&[0..bsz, 0..n_heads, 0..seq_len, 0..self.dim])?;
        let x_pass = xs.slice(&[0..bsz, 0..n_heads, 0..seq_len, self.dim..hdim])?;

        let xs1 = x_rot
            .clone()
            .slice(&[0..bsz, 0..n_heads, 0..seq_len, 0..self.dim / 2])?;
        let xs2 = x_rot
            .clone()
            .slice(&[0..bsz, 0..n_heads, 0..seq_len, self.dim / 2..self.dim])?;

        let rotate_half = Tensor::cat(rvec![xs2.neg()?, xs1], 3)?;
        let xs_cos = x_rot.mul(cos)?;
        let rh_sin = rotate_half.mul(sin)?;
        x_rot = xs_cos.add(rh_sin)?;
        Tensor::cat(rvec![x_rot, x_pass], 3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratchet::DeviceRequest;

    thread_local! {
        static GPU_DEVICE: Device = Device::request_device(DeviceRequest::GPU).unwrap();
    }

    #[test]
    fn test_rope() -> anyhow::Result<()> {
        let rope_theta = 10000.0;
        let dim = (0.4 * (2560f64 / 32f64)) as usize;
        let max_position_embeddings = 2048;
        let d = GPU_DEVICE.with(|d| d.clone());
        let rope = RotaryEmbedding::new(dim, max_position_embeddings, rope_theta, d)?;
        let rand = Tensor::randn::<f32>(shape![1, 32, 7, 80], rope.cos.device().clone());

        let roped = rope.apply_rotary_embedding(rand, 0)?.resolve()?;
        let cpu_roped = roped.to(&Device::CPU)?;

        println!("{:?}", cpu_roped);

        Ok(())
    }
}
