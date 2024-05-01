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

#[derive(Clone, Debug, derive_new::new)]
pub struct LayerNorm {
    weight: Tensor,
    bias: Option<Tensor>,
    eps: f32,
}

impl LayerNorm {
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

impl crate::Module for LayerNorm {
    type Input = Tensor;
    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        input.layer_norm(self.weight.clone(), self.bias.clone(), self.eps)
    }
}

/// RMSNorm
///
/// https://github.com/NVIDIA/apex/pull/1274/files
#[derive(Clone, Debug, derive_new::new)]
pub struct RMSNorm {
    weight: Tensor,
    eps: f32,
}

impl RMSNorm {
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }
}

impl crate::Module for RMSNorm {
    type Input = Tensor;
    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        input.rms_norm(self.weight.clone(), self.eps)
    }
}
