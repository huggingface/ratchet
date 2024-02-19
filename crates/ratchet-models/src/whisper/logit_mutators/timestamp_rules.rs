use ndarray::s;
use ndarray_stats::QuantileExt;
use ratchet::{NDArrayExt, Tensor};

use crate::{LogitMutator, WhisperTokenizer};

#[derive(Debug, derive_new::new)]
pub struct ApplyTimestampRules {
    pub sample_begin: usize,
    pub max_initial_timestamp_index: Option<usize>,
}

impl LogitMutator for ApplyTimestampRules {
    fn apply(&self, logits: Tensor, tokens: &Tensor) -> anyhow::Result<Tensor> {
        let nd_tokens = tokens.clone().into_ndarray::<i32>();
        let mut nd_logits = logits.into_ndarray::<f32>();

        nd_logits
            .slice_mut(s![.., WhisperTokenizer::NO_TIMESTAMPS as usize])
            .map_inplace(move |el| *el = f32::NEG_INFINITY);

        for k in 0..nd_tokens.shape()[0] {
            let sampled_tokens = nd_tokens.slice(s![k, self.sample_begin..]);
            let sample_len = sampled_tokens.len();

            let last_was_timestamp = !sampled_tokens.is_empty()
                && sampled_tokens[sample_len - 1] >= WhisperTokenizer::TS_BEGIN;
            let penultimate_was_timestamp = sampled_tokens.len() < 2
                || sampled_tokens[sample_len - 2] >= WhisperTokenizer::TS_BEGIN;

            if last_was_timestamp {
                if penultimate_was_timestamp {
                    nd_logits
                        .slice_mut(s![k, WhisperTokenizer::TS_BEGIN..])
                        .map_inplace(move |el| *el = f32::NEG_INFINITY);
                } else {
                    nd_logits
                        .slice_mut(s![k, ..WhisperTokenizer::EOT])
                        .map_inplace(move |el| *el = f32::NEG_INFINITY);
                }
            }

            let timestamps = sampled_tokens
                .iter()
                .filter(|x| **x >= WhisperTokenizer::TS_BEGIN)
                .collect::<Vec<_>>();

            if !timestamps.is_empty() {
                // timestamps shouldn't decrease; forbid timestamp tokens smaller than the last
                // also force each segment to have a nonzero length, to prevent infinite looping
                let timestamp_last = if last_was_timestamp && !penultimate_was_timestamp {
                    *timestamps[timestamps.len() - 1]
                } else {
                    timestamps[timestamps.len() - 1] + 1
                };
                nd_logits
                    .slice_mut(s![k, WhisperTokenizer::TS_BEGIN..timestamp_last])
                    .map_inplace(move |el| *el = f32::NEG_INFINITY);
            }
        }
        if nd_tokens.shape()[1] == self.sample_begin {
            // suppress generating non-timestamp tokens at the beginning
            nd_logits
                .slice_mut(s![.., ..WhisperTokenizer::TS_BEGIN])
                .map_inplace(move |el| *el = f32::NEG_INFINITY);

            if self.max_initial_timestamp_index.is_some() {
                let last_allowed = (WhisperTokenizer::TS_BEGIN as usize)
                    + self.max_initial_timestamp_index.unwrap();
                nd_logits
                    .slice_mut(s![.., last_allowed + 1..])
                    .map_inplace(move |el| *el = f32::NEG_INFINITY);
            }
        }

        let logprobs = nd_logits.log_softmax(1);
        for _k in 0..nd_tokens.shape()[0] {
            let timestamp_logprob = logprobs
                .slice(s![.., WhisperTokenizer::TS_BEGIN..])
                .logsumexp(1);
            let text_logprobs = logprobs.slice(s![.., ..WhisperTokenizer::TS_BEGIN]);
            let max_text_token_logprob = text_logprobs.max()?;
            if timestamp_logprob > *max_text_token_logprob {
                nd_logits
                    .slice_mut(s![.., ..WhisperTokenizer::TS_BEGIN])
                    .map_inplace(move |el| *el = f32::NEG_INFINITY);
            }
        }
        Ok(Tensor::from(nd_logits))
    }
}
