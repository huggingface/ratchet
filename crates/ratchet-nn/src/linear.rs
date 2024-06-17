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
}

impl Module for Linear {
    type Input = Tensor;
    fn schedule(&self, input: Self::Input) -> anyhow::Result<Tensor> {
        let input_dt = input.dt();
        let w = self.w.clone().cast(input_dt)?;
        let bias = self.b.clone().map(|b| b.cast(input_dt)).transpose()?;
        w.gemm(input, bias, false, true, true)
    }
}
