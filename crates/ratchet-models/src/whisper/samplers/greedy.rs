use crate::whisper::task::DecodeError;
use crate::whisper::tokenizer::WhisperTokenizer;

use ndarray::Axis;
use ndarray_stats::QuantileExt;
use ratchet::Tensor;

pub struct GreedySampler;

impl GreedySampler {
    pub fn sample(
        mut tokens: Vec<i32>,
        logits: Tensor,
    ) -> Result<(Tensor, Vec<i32>, bool), DecodeError> {
        let nd_logits = logits.to_ndarray_view::<f32>();
        let next_tokens = nd_logits
            .map_axis(Axis(1), |row| row.argmax_skipnan())
            .iter()
            .map(|r| {
                r.as_ref()
                    .map_err(|_| DecodeError::InvalidLogits)
                    .map(|v| *v as i32)
            })
            .collect::<Result<Vec<i32>, DecodeError>>()?;

        tokens.extend_from_slice(&next_tokens);
        let completed = tokens[tokens.len() - 1] == WhisperTokenizer::EOT;
        Ok((logits, tokens, completed))
    }
}
