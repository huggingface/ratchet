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
        //cast bias if necessary
        //
        let b = if let Some(b) = &self.b {
            Some(b.clone().cast(input.dt())?)
        } else {
            None
        };
        self.w.clone().gemm(input, b, false, true, true)
    }
}
