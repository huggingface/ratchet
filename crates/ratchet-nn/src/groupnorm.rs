use ratchet::Tensor;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroupNormConfig {
    pub eps: f32,
    pub num_groups: usize,
}

impl Default for GroupNormConfig {
    fn default() -> Self {
        Self {
            eps: 1e-5,
            num_groups: 1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GroupNorm {
    weight: Tensor,
    bias: Option<Tensor>,
    num_groups: usize,
    eps: f32,
}

impl GroupNorm {
    pub fn new(weight: Tensor, bias: Option<Tensor>, num_groups: usize, eps: f32) -> Self {
        Self {
            weight,
            bias,
            num_groups,
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

impl crate::Module for GroupNorm {
    type Input = Tensor;
    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        input.group_norm(
            self.num_groups,
            self.weight.clone(),
            self.bias.clone(),
            self.eps,
        )
    }
}
