mod linear;
mod norm;
mod rope;

pub use linear::*;
pub use norm::*;
pub use rope::*;

use ratchet::Tensor;

pub trait Module {
    type Input;
    fn forward(&self, input: &Self::Input) -> anyhow::Result<Tensor>;
}
