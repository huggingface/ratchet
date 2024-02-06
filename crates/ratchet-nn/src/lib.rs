mod linear;
mod norm;

pub use linear::*;
pub use norm::*;

use ratchet::Tensor;

pub trait Module {
    type Input;
    fn forward(&self, input: &Self::Input) -> anyhow::Result<Tensor>;
}
