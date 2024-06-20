use ratchet::{prelude::shape, rvec, Tensor};
use ratchet_nn::{LayerNorm, Linear, Module};

use super::mlp::MLP;

#[derive(Debug, derive_new::new)]
pub struct Attention {
    n_heads: usize,
    dim: usize,
    qkv: Linear,
    proj: Linear,
    scale_factor: Tensor,
}

impl Module for Attention {
    type Input = Tensor;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let h_dim = self.dim / self.n_heads;
        let [b, n, c]: [usize; 3] = input.shape().try_into()?;
        // step 1 - 0, 1, 2, 3, 4
        // step 2 - 0, 2, 1, 3, 4
        // step 3 - 2, 0, 1, 3, 4
        // step 4 - 2, 0, 3, 1, 4

        // b, n, 3, nh, hd
        let mut qkv = self.qkv.schedule(input.clone())?;
        // b, 3, n, nh, hd
        qkv = qkv
            .view(shape![b, n, 3, self.n_heads * h_dim])?
            .permute(&[0, 2, 1, 3])?;
        // 3, b, n, nh, hd
        qkv = qkv
            .view(shape![b, 3, n * self.n_heads * h_dim])?
            .permute(&[1, 0, 2])?;
        // 3, b, nh, n, hd
        qkv = qkv
            .view(shape![3 * b, n, self.n_heads, h_dim])?
            .permute(&[0, 2, 1, 3])?
            .view(shape![3, b * self.n_heads * n * h_dim])?;

        let q = qkv
            .clone()
            .slice(&[0..1, 0..(b * self.n_heads * n * h_dim)])?
            .view(shape![b, self.n_heads, n, h_dim])?;
        let k = qkv
            .clone()
            .slice(&[1..2, 0..(b * self.n_heads * n * h_dim)])?
            .view(shape![b, self.n_heads, n, h_dim])?;
        let v = qkv
            .clone()
            .slice(&[2..3, 0..(b * self.n_heads * n * h_dim)])?
            .view(shape![b, self.n_heads, n, h_dim])?;

        // scaled dot-product attention
        let mut attn_weights = q
            .full()?
            .matmul(k.permute(&[0, 1, 3, 2])?.full()?, false, false)?
            .mul(self.scale_factor.clone())?;
        attn_weights = attn_weights.softmax(3)?.cast(v.dt())?;
        let mut x = attn_weights.matmul(v, false, false)?;
        x = x.permute(&[0, 2, 1, 3])?.view(shape![b, n, c])?;
        self.proj.schedule(x)
    }
}

#[derive(Debug, derive_new::new)]
pub struct VitBlock {
    embed_dim: usize,
    attn: Attention,
    mlp: MLP,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl Module for VitBlock {
    type Input = Tensor;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let x = input
            .clone()
            .add(self.attn.schedule(self.norm1.schedule(input)?)?)?;
        x.clone().add(self.mlp.schedule(self.norm2.schedule(x)?)?)
    }
}

#[derive(Debug, derive_new::new)]
pub struct LinearPatchEmbedding {
    linear: Linear,
}

impl Module for LinearPatchEmbedding {
    type Input = Tensor;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let [b, c, hp1, wp2]: [usize; 4] = input.shape().try_into()?;
        let (p1, p2) = (14, 14);
        let (h, w) = (hp1 / p1, wp2 / p2);
        // step 1 - 0, 1, 2, 3, 4, 5
        // step 2 - 0, 2, 1, 3, 4, 5
        // step 3 - 0, 2, 1, 4, 3, 5
        // step 4 - 0, 2, 4, 1, 3, 5

        // b, c, h, p1, w, p2
        let mut x = input
            .view(shape![b, c, h, p1 * w * p2])?
            .permute(&[0, 2, 1, 3])?;
        // b, h, c, p1, w, p2
        x = x
            .view(shape![b * h * c, p1, w, p2])?
            .permute(&[0, 2, 1, 3])?;
        // b, h, c, w, p1, p2
        x = x
            .view(shape![b * h, c, w, p1 * p2])?
            .permute(&[0, 2, 1, 3])?;
        // b, h, w, c, p1, p2
        x = x.view(shape![b, h * w, c * p1 * p2])?;
        self.linear.schedule(x)
    }
}

#[derive(Debug, derive_new::new)]
pub struct VisionTransformer {
    patch_embed: LinearPatchEmbedding,
    pos_embed: Tensor,
    blocks: Vec<VitBlock>,
    norm: LayerNorm,
}

impl Module for VisionTransformer {
    type Input = Tensor;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let mut x = self.patch_embed.schedule(input)?;
        x = x.clone().add(self.pos_embed.clone())?;
        x = self
            .blocks
            .iter()
            .fold(x.clone(), |acc, blk| blk.schedule(acc).unwrap());
        self.norm.schedule(x)
    }
}

#[derive(Debug, derive_new::new)]
pub struct VisionProjection {
    mlp: MLP,
}

impl Module for VisionProjection {
    type Input = Tensor;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        self.mlp.schedule(input)
    }
}

#[derive(Debug, derive_new::new)]
pub struct VisionEncoder {
    projection: VisionProjection,
    transformer: VisionTransformer,
}

impl Module for VisionEncoder {
    type Input = Tensor;

    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let transformed = self.transformer.schedule(input)?;
        self.projection.schedule(Tensor::cat(
            rvec![transformed.clone(), transformed.clone()],
            2,
        )?)
    }
}
