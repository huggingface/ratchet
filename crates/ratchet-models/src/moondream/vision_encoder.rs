use ratchet::{prelude::shape, Device, DeviceRequest, Tensor};
use ratchet_nn::{Linear, Module};

struct VitAttn {
    n_heads: usize,
    dim: usize,
    qkv: Linear,
    proj: Linear,
}

impl Module for VitAttn {
    type Input = Tensor;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let h_dim = self.dim / self.n_heads;
        let [b, n, c]: [usize; 3] = input.shape().try_into()?;
        let qkv = self
            .qkv
            .schedule(input.clone())?
            .view(shape![b, n, 3, self.n_heads, h_dim])?;
        let q = qkv
            .clone()
            .slice(&[0..b, 0..n, 0..1, 0..self.n_heads, 0..h_dim])?
            .view(shape![b, n, self.n_heads, h_dim])?
            .permute(&[0, 2, 1, 3])?;
        let k = qkv
            .clone()
            .slice(&[0..b, 0..n, 1..2, 0..self.n_heads, 0..h_dim])?
            .view(shape![b, n, self.n_heads, h_dim])?
            .permute(&[0, 2, 1, 3])?;
        let v = qkv
            .clone()
            .slice(&[0..b, 0..n, 2..3, 0..self.n_heads, 0..h_dim])?
            .view(shape![b, n, self.n_heads, h_dim])?
            .permute(&[0, 2, 1, 3])?;

        // scaled dot-product attention
        let scale_factor = Tensor::from_data(
            [1.0 / (h_dim as f32).sqrt()],
            shape![1],
            input.clone().device().clone(),
        );
        let mut attn_weights = q
            .matmul(k.permute(&[0, 1, 3, 2])?, false, false)?
            .mul(scale_factor)?;
        attn_weights = attn_weights.softmax(3)?;
        let mut x = attn_weights.matmul(v, false, false)?;

        x = x.permute(&[0, 2, 1])?.view(shape![b, n, c])?;
        x = self.proj.schedule(x)?;

        Ok(x)
    }
}