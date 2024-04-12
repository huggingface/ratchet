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
        //y = xW^T + b
        //input.gemm(self.w.clone(), self.b.clone(), false, true, false)
        //if we wanted to reverse this:
        //yT = Wx^T + bT
        self.w
            .clone()
            .gemm(input, self.b.clone(), false, true, true)
    }
}
