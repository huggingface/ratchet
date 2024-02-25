use ndarray::s;
use ndarray::Axis;
use ratchet::prelude::shape;
use ratchet::Device;
use ratchet::Tensor;
use ratchet_nn::Module;

use crate::ApplyTimestampRules;
use crate::DecodingOptions;
use crate::GreedySampler;
use crate::LogitMutator;
use crate::Prompt;
use crate::Segment;
use crate::WhisperDecoder;
use crate::WhisperTokenizer;
use crate::CHUNK_LENGTH;
use crate::HOP_LENGTH;
use crate::N_AUDIO_CTX;
use crate::SAMPLE_RATE;

#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    #[error("No valid logits found")]
    NoValidLogitsFound,
    #[error("Tokenizer error: {0}")]
    TokenizerError(#[from] tokenizers::Error),
    #[error("Unknown error: {0}")]
    UnknownError(#[from] anyhow::Error),
    #[error("Failed to resolve tensor: {0}")]
    TensorResolveError(#[from] ratchet::TensorError),
}

pub struct DecodingTask {
    options: DecodingOptions,
    sample_len: u32,
    logit_mutators: Vec<Box<dyn LogitMutator>>,
    initial_tokens: Option<Vec<i32>>,
    initial_tokens_len: Option<usize>,
}

impl DecodingTask {
    fn get_initial_tokens(&self, tokenizer: &WhisperTokenizer) -> Vec<i32> {
        let mut init_tokens = tokenizer.sot_sequence();
        if let Some(prompt) = &self.options.prompt {
            let prompt_tokens = match prompt {
                Prompt::Tokens(tokens) => tokens.clone(),
                Prompt::Text(text) => tokenizer
                    .encode(format!(" {}", text).as_str(), false)
                    .unwrap(),
            };
            let max_prompt_length = 448 / 2 - 1; // equivalent to self.n_ctx // 2 - 1 in python
            let prompt_length = prompt_tokens.len().min(max_prompt_length);
            let mut tokens = vec![WhisperTokenizer::START_OF_PREV];
            tokens.extend_from_slice(&prompt_tokens[prompt_tokens.len() - prompt_length..]);
            tokens.extend(init_tokens);
            init_tokens = tokens;
        }
        init_tokens
    }

    pub fn new(options: DecodingOptions, tokenizer: &WhisperTokenizer) -> Self {
        let sample_len = options.sample_len.unwrap_or(256);
        let _selected_lang = options.language.as_ref().unwrap();
        let max_initial_timestamp = options.max_initial_timestamp;
        let mut task = DecodingTask {
            options,
            logit_mutators: vec![],
            sample_len,
            initial_tokens: None,
            initial_tokens_len: None,
        };
        task.initial_tokens = Some(task.get_initial_tokens(tokenizer));
        task.initial_tokens_len = Some(task.initial_tokens.as_ref().unwrap().len());

        let mut max_initial_timestamp_index = None;
        if let Some(max_initial_timestamp) = max_initial_timestamp {
            let precision = CHUNK_LENGTH as f32 / N_AUDIO_CTX as f32;
            max_initial_timestamp_index =
                Some((max_initial_timestamp / precision).round() as usize);
        }
        task.logit_mutators.push(Box::new(ApplyTimestampRules {
            sample_begin: task.initial_tokens_len.unwrap(),
            max_initial_timestamp_index,
        }));

        task
    }

    fn main_loop(
        &self,
        decoder: &mut WhisperDecoder,
        audio_ctx: Tensor,
        mut tokens: Vec<i32>,
    ) -> Result<Vec<i32>, DecodeError> {
        let device = audio_ctx.device().clone();

        for idx in 0..self.sample_len {
            device.try_gpu().unwrap().begin_pass(idx as _);
            let input = if tokens.len() > self.initial_tokens_len.unwrap() {
                &tokens[tokens.len() - 1..]
            } else {
                &tokens
            };
            let input_t = Tensor::from_data(input, shape![1, input.len()], device.clone());

            let logits = decoder.forward(&[audio_ctx.clone(), input_t])?.resolve()?;
            decoder.cache_mut().update(input.len());

            let mut logits = Self::slice_logits(logits.to(&Device::CPU)?);
            let token_t = Tensor::from_data(tokens.clone(), shape![1, tokens.len()], Device::CPU);
            for m in &self.logit_mutators {
                logits = m.apply(logits, Some(&token_t))?;
            }

            let (_, new_tokens, completed) = GreedySampler::sample(tokens, logits)?;

            tokens = new_tokens;
            if completed {
                break;
            }
        }
        Ok(tokens)
    }

    /// Slice logits from [1xnum_tokensx51872] -> [1x1x51865]
    pub(crate) fn slice_logits(logits: Tensor) -> Tensor {
        let nd_logits = logits.into_ndarray::<f32>();
        let sliced = nd_logits
            .slice(s![.., -1.., ..WhisperTokenizer::SIZE])
            .remove_axis(Axis(1));
        Tensor::from(sliced.to_owned().into_dyn())
    }

    pub fn build_segments(
        tokens: Vec<i32>,
        offset: f64,
        segment_size: usize,
        segment_duration: usize,
        input_stride: usize,
    ) -> (Vec<Segment>, usize) {
        let content_tokens = tokens;
        let content_length = content_tokens.len();
        if content_length < 2 {
            log::error!("Failed to build segments.");
            return (Vec::new(), 0);
        }
        let (penultimate, last) = (
            content_tokens[content_length - 2],
            content_tokens[content_length - 1],
        );

        let single_timestamp_ending =
            !WhisperTokenizer::is_timestamp(penultimate) && WhisperTokenizer::is_timestamp(last);

        let mut consecutive = content_tokens
            .windows(2)
            .enumerate()
            .filter_map(|(i, x)| {
                if WhisperTokenizer::is_timestamp(x[0]) && WhisperTokenizer::is_timestamp(x[1]) {
                    Some(i + 1)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        let advance;
        let segments;
        if !consecutive.is_empty() {
            // if the output contains two consecutive timestamp tokens
            if single_timestamp_ending {
                consecutive.push(content_length);
            }

            let mut last_slice = 0;
            (segments, last_slice) =
                consecutive
                    .iter()
                    .fold((Vec::new(), 0), |(mut acc, last_slice), &slice| {
                        let segment_tokens = &content_tokens[last_slice..slice];
                        acc.push(Segment::from_tokens(segment_tokens, offset, false));
                        (acc, slice)
                    });

            advance = if single_timestamp_ending {
                segment_size
            } else {
                let last_timestamp_pos =
                    content_tokens[last_slice - 1] - WhisperTokenizer::TS_BEGIN;
                last_timestamp_pos as usize * input_stride
            }
        } else {
            let duration = content_tokens
                .iter()
                .filter(|&x| WhisperTokenizer::is_timestamp(*x))
                .last()
                .map_or(segment_duration as f64, |&last_ts| {
                    let last_timestamp_pos = last_ts - WhisperTokenizer::TS_BEGIN;
                    last_timestamp_pos as f64 * input_stride as f64 * (HOP_LENGTH as f64)
                        / (SAMPLE_RATE as f64)
                });

            let segment_tokens = content_tokens.iter().map(|x| *x as u32).collect::<Vec<_>>();
            segments = vec![Segment::new(
                offset,
                offset + duration,
                segment_tokens,
                false,
            )];
            advance = segment_size;
        }

        (segments, advance)
    }

    pub fn run(
        &self,
        decoder: &mut WhisperDecoder,
        audio_ctx: Tensor,
        tokenizer: &WhisperTokenizer,
    ) -> Result<Vec<i32>, DecodeError> {
        let mut tokens = self.main_loop(decoder, audio_ctx, self.get_initial_tokens(tokenizer))?;

        tokens = tokens.drain(self.initial_tokens_len.unwrap()..).collect();
        let eot_index = tokens.iter().position(|x| *x == WhisperTokenizer::EOT);
        if let Some(eot_index) = eot_index {
            tokens.truncate(eot_index);
        }
        Ok(tokens)
    }
}
