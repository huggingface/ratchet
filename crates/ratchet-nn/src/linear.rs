use ratchet::Tensor;

use crate::Module;

/// # Linear
///
/// PyTorch case: y = xW^T + b
/// If your weights are already in the correct layout, you can set `transpose` to `false` to avoid the transpose operation.
#[derive(derive_new::new, Debug)]
pub struct Linear {
    pub w: Tensor,
    b: Option<Tensor>,
    transpose: bool,
}

impl Module for Linear {
    type Input = Tensor;
    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        input.gemm(self.w.clone(), self.b.clone(), false, self.transpose, false)
    }
}

#[derive(derive_new::new, Debug)]
pub struct RLinear {
    pub w: Tensor,
    b: Option<Tensor>,
}

impl Module for RLinear {
    type Input = Tensor;
    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        self.w
            .clone()
            .gemm(input, self.b.clone(), false, true, false)
    }
}
