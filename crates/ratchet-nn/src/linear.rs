use ratchet::Tensor;

use crate::Module;

#[derive(derive_new::new, Debug)]
pub struct Linear {
    pub w: Tensor,
    b: Option<Tensor>,
}

impl Module for Linear {
    type Input = Tensor;
    fn forward(&self, input: &Self::Input) -> anyhow::Result<Tensor> {
        let y = input.matmul(&self.w, true)?;
        if let Some(b) = &self.b {
            y.add(b)
        } else {
            Ok(y)
        }
    }
}
