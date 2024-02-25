mod select_language;
mod timestamp_rules;

pub use select_language::*;
pub use timestamp_rules::*;

use ratchet::Tensor;

pub trait LogitMutator {
    fn apply(&self, logits: Tensor, tokens: Option<&Tensor>) -> anyhow::Result<Tensor>;
}
