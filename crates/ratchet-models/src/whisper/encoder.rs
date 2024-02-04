use ratchet::Tensor;
use ratchet_nn::Module;

#[derive(derive_new::new)]
struct ConvBlock {
    w: Tensor,
    b: Tensor,
    stride: usize,
    padding: usize,
}

impl Module for ConvBlock {
    fn forward(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        x.conv1d(&self.w, Some(&self.b), self.stride, self.padding)?
            .gelu()
    }
}

struct EncoderStem {
    conv1: ConvBlock,
    conv2: ConvBlock,
    pos_embed: Tensor,
}

impl Module for EncoderStem {
    fn forward(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let x = self.conv1.forward(x)?;
        let x = self.conv2.forward(&x)?;
        x.permute(&[0, 2, 1])?.add(&self.pos_embed)
    }
}
