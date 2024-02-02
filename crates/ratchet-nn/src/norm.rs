use ratchet::Tensor;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerNormConfig {
    pub eps: f32,
    pub remove_mean: bool,
}

impl Default for LayerNormConfig {
    fn default() -> Self {
        Self {
            eps: 1e-5,
            remove_mean: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LayerNorm {
    weight: Tensor,
    bias: Option<Tensor>,
    remove_mean: bool,
    eps: f32,
}

impl LayerNorm {
    pub fn new(weight: Tensor, bias: Option<Tensor>, eps: f32) -> Self {
        Self {
            weight,
            bias,
            remove_mean: true,
            eps,
        }
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

impl crate::Module for LayerNorm {
    fn forward(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        x.layer_norm(&self.weight, self.bias.as_ref(), self.eps)
    }
}
