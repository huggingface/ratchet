mod timestamp_rules;

pub use timestamp_rules::*;

use ratchet::Tensor;

pub trait LogitMutator {
    fn apply(&self, logits: Tensor, tokens: Tensor) -> anyhow::Result<Tensor>;
}
