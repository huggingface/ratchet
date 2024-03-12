mod timestamp_rules;

pub use timestamp_rules::*;

use ratchet::Tensor;

use crate::WhisperTokenizer;

pub trait LogitMutator {
    fn apply(
        &self,
        logits: Tensor,
        tokenizer: &WhisperTokenizer,
        tokens: Option<&Tensor>,
    ) -> anyhow::Result<Tensor>;
}
