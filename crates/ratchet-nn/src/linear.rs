use ratchet::Tensor;

#[derive(derive_new::new, Debug)]
pub struct Linear {
    w: Tensor,
    b: Option<Tensor>,
}

impl Linear {
    pub fn forward(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let y = x.matmul(&self.w.permute(&[1, 0])?)?;
        if let Some(b) = &self.b {
            y.add(b)
        } else {
            Ok(y)
        }
    }
}
