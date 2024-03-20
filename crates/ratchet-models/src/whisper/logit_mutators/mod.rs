mod timestamp_rules;
pub use timestamp_rules::*;

use crate::whisper::tokenizer::WhisperTokenizer;
use ratchet::Tensor;

pub trait LogitMutator {
    fn apply(
        &self,
        logits: Tensor,
        tokenizer: &WhisperTokenizer,
        tokens: Option<&Tensor>,
    ) -> anyhow::Result<Tensor>;
}
