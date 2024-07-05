use super::{
    decoder::WhisperDecoder, logit_mutators::*, samplers::*, spectrogram::*,
    tokenizer::WhisperTokenizer, transcript::*,
};
use crate::whisper::options::{DecodingOptions, Prompt};
use ndarray::{s, Axis};
use ratchet::{shape, Device, Tensor};
use ratchet_nn::Module;

#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    #[error("No valid logits found")]
    InvalidLogits,
    #[error("Tokenizer error: {0}")]
    TokenizerError(#[from] tokenizers::Error),
    #[error("Unknown error: {0}")]
    UnknownError(#[from] anyhow::Error),
    #[error("Failed to resolve tensor: {0}")]
    TensorResolveError(#[from] ratchet::TensorError),
}

pub struct DecodingTask {
    tokenizer: WhisperTokenizer,
    options: DecodingOptions,
    sample_len: u32,
    logit_mutators: Vec<Box<dyn LogitMutator>>,
    initial_tokens: Option<Vec<i32>>,
    initial_tokens_len: Option<usize>,
}

impl DecodingTask {
    fn get_initial_tokens(&self) -> Vec<i32> {
        let mut init_tokens = self.tokenizer.sot_sequence();
        if let Some(prompt) = &self.options.prompt {
            let prompt_tokens = match prompt {
                Prompt::Tokens(tokens) => tokens.clone(),
                Prompt::Text(text) => self
                    .tokenizer
                    .encode(format!(" {}", text).as_str(), false)
                    .unwrap(),
            };
            let max_prompt_length = 448 / 2 - 1; // equivalent to self.n_ctx // 2 - 1 in python
            let prompt_length = prompt_tokens.len().min(max_prompt_length);
            let mut tokens = vec![self.tokenizer.sot_prev()];
            tokens.extend_from_slice(&prompt_tokens[prompt_tokens.len() - prompt_length..]);
            tokens.extend(init_tokens);
            init_tokens = tokens;
        }
        init_tokens
    }

    pub fn new(options: DecodingOptions, tokenizer: WhisperTokenizer) -> Self {
        let sample_len = options.sample_len.unwrap_or(256);
        let _selected_lang = options.language.as_ref().unwrap();
        let max_initial_timestamp = options.max_initial_timestamp;
        let mut task = DecodingTask {
            tokenizer,
            options,
            logit_mutators: vec![],
            sample_len,
            initial_tokens: None,
            initial_tokens_len: None,
        };
        task.initial_tokens = Some(task.get_initial_tokens());
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

    #[cfg(not(target_arch = "wasm32"))]
    fn main_loop(
        &self,
        decoder: &mut WhisperDecoder,
        audio_ctx: Tensor,
        callback: &Option<impl Fn(StreamedSegment)>,
    ) -> Result<Vec<i32>, DecodeError> {
        use ratchet::DType;

        let mut tokens = self.get_initial_tokens();
        let sliced_vocab_size = self.tokenizer.vocab_size();
        let device = audio_ctx.device().clone();
        let mut timestamps_seen = 0;

        for _ in 0..self.sample_len {
            let input = if tokens.len() > self.initial_tokens_len.unwrap() {
                &tokens[tokens.len() - 1..]
            } else {
                &tokens
            };
            let input_t = Tensor::from_data(input, shape![1, input.len()], device.clone());

            let logits = decoder
                .schedule([audio_ctx.clone(), input_t])?
                .cast(DType::F32)?
                .resolve_debug()?;
            decoder.cache_mut().update(input.len());

            let mut logits = Self::slice_logits(logits.to(&Device::CPU)?, sliced_vocab_size);
            let token_t = Tensor::from_data(tokens.clone(), shape![1, tokens.len()], Device::CPU);
            for m in &self.logit_mutators {
                logits = m.apply(logits, &self.tokenizer, Some(&token_t))?;
            }

            let (_, new_tokens, completed) = GreedySampler::sample(tokens, logits)?;

            if let Some(ref cb) = callback {
                self.handle_callback(&self.tokenizer, &new_tokens, &mut timestamps_seen, cb);
            }

            tokens = new_tokens;
            if completed {
                break;
            }
        }
        Ok(tokens)
    }

    #[cfg(target_arch = "wasm32")]
    async fn main_loop(
        &self,
        decoder: &mut WhisperDecoder,
        audio_ctx: Tensor,
        callback: &Option<impl Fn(StreamedSegment)>,
    ) -> Result<Vec<i32>, DecodeError> {
        let mut tokens = self.get_initial_tokens();
        let device = audio_ctx.device().clone();
        let sliced_vocab_size = self.tokenizer.vocab_size();
        let mut timestamps_seen = 0;

        for _ in 0..self.sample_len {
            let input = if tokens.len() > self.initial_tokens_len.unwrap() {
                &tokens[tokens.len() - 1..]
            } else {
                &tokens
            };
            let input_t = Tensor::from_data(input, shape![1, input.len()], device.clone());

            let logits = decoder.schedule([audio_ctx.clone(), input_t])?.resolve()?;
            decoder.cache_mut().update(input.len());

            let mut logits = Self::slice_logits(logits.to(&Device::CPU).await?, sliced_vocab_size);
            let token_t = Tensor::from_data(tokens.clone(), shape![1, tokens.len()], Device::CPU);
            for m in &self.logit_mutators {
                logits = m.apply(logits, &self.tokenizer, Some(&token_t))?;
            }

            let (_, new_tokens, completed) = GreedySampler::sample(tokens, logits)?;

            if let Some(ref cb) = callback {
                self.handle_callback(&self.tokenizer, &new_tokens, &mut timestamps_seen, cb);
            }

            tokens = new_tokens;
            if completed {
                break;
            }
        }
        Ok(tokens)
    }

    fn handle_callback(
        &self,
        tokenizer: &WhisperTokenizer,
        new_tokens: &[i32],
        timestamps_seen: &mut i32,
        callback: impl Fn(StreamedSegment),
    ) {
        if tokenizer.is_timestamp(new_tokens[new_tokens.len() - 1]) {
            *timestamps_seen += 1;
            if *timestamps_seen % 2 == 0 {
                let previous_timestamp = new_tokens[..new_tokens.len() - 2]
                    .iter()
                    .rposition(|x| tokenizer.is_timestamp(*x));
                if let Some(previous_timestamp) = previous_timestamp {
                    callback(StreamedSegment::from_tokens(
                        tokenizer,
                        &new_tokens[previous_timestamp..],
                        self.options.time_offset.unwrap_or(0.0),
                        false,
                    ));
                }
            }
        }
    }

    /// Slice logits from [1xnum_tokensx51872] -> [1x1x51865]
    pub(crate) fn slice_logits(logits: Tensor, vocab_size: usize) -> Tensor {
        let nd_logits = logits.into_ndarray::<f32>();
        let sliced = nd_logits
            .slice(s![.., -1.., ..vocab_size])
            .remove_axis(Axis(1));
        Tensor::from(sliced.to_owned().into_dyn())
    }

    pub fn build_segments(
        tokenizer: &WhisperTokenizer,
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
            !tokenizer.is_timestamp(penultimate) && tokenizer.is_timestamp(last);

        let mut consecutive = content_tokens
            .windows(2)
            .enumerate()
            .filter_map(|(i, x)| {
                if tokenizer.is_timestamp(x[0]) && tokenizer.is_timestamp(x[1]) {
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

            #[allow(unused_assignments)]
            let mut last_slice = 0;
            (segments, last_slice) =
                consecutive
                    .iter()
                    .fold((Vec::new(), 0), |(mut acc, last_slice), &slice| {
                        let segment_tokens = &content_tokens[last_slice..slice];
                        acc.push(Segment::from_tokens(
                            tokenizer,
                            segment_tokens,
                            offset,
                            false,
                        ));
                        (acc, slice)
                    });

            advance = if single_timestamp_ending {
                segment_size
            } else {
                let last_timestamp_pos =
                    content_tokens[last_slice - 1] - tokenizer.timestamp_begin();
                last_timestamp_pos as usize * input_stride
            }
        } else {
            let duration = content_tokens
                .iter()
                .filter(|&x| tokenizer.is_timestamp(*x))
                .last()
                .map_or(segment_duration as f64, |&last_ts| {
                    let last_timestamp_pos = last_ts - tokenizer.timestamp_begin();
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

    #[cfg(target_arch = "wasm32")]
    pub async fn run(
        &self,
        decoder: &mut WhisperDecoder,
        audio_ctx: Tensor,
        callback: &Option<impl Fn(StreamedSegment)>,
    ) -> Result<Vec<i32>, DecodeError> {
        let mut tokens = self.main_loop(decoder, audio_ctx, &callback).await?;

        tokens = tokens.drain(self.initial_tokens_len.unwrap()..).collect();
        let eot_index = tokens.iter().position(|x| *x == WhisperTokenizer::EOT);
        if let Some(eot_index) = eot_index {
            tokens.truncate(eot_index);
        }
        Ok(tokens)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn run(
        &self,
        decoder: &mut WhisperDecoder,
        audio_ctx: Tensor,
        callback: &Option<impl Fn(StreamedSegment)>,
    ) -> Result<Vec<i32>, DecodeError> {
        let mut tokens = self.main_loop(decoder, audio_ctx, callback)?;

        tokens = tokens.drain(self.initial_tokens_len.unwrap()..).collect();
        let eot_index = tokens.iter().position(|x| *x == WhisperTokenizer::EOT);
        if let Some(eot_index) = eot_index {
            tokens.truncate(eot_index);
        }
        Ok(tokens)
    }
}
