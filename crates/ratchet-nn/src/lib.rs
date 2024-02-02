mod norm;
pub use norm::*;

use ratchet::Tensor;

pub trait Module {
    fn forward(&self, input: &Tensor) -> anyhow::Result<Tensor>;
}
