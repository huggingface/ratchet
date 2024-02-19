use ratchet::prelude::shape;
use ratchet::Device;
use ratchet::Tensor;
use ratchet_nn::Module;

use crate::DecodingOptions;
use crate::GreedySampler;
use crate::LogitMutator;
use crate::Prompt;
use crate::WhisperDecoder;
use crate::WhisperTokenizer;
use crate::CHUNK_LENGTH;
use crate::N_AUDIO_CTX;

#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    #[error("No valid logits found")]
    NoValidLogitsFound,
    #[error("Tokenizer error: {0}")]
    TokenizerError(#[from] tokenizers::Error),
    #[error("Unknown error: {0}")]
    UnknownError(#[from] anyhow::Error),
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
        let selected_lang = options.language.as_ref().unwrap();
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
        task
    }

    fn main_loop(
        &self,
        decoder: &WhisperDecoder,
        audio_ctx: Tensor,
        mut tokens: Vec<i32>,
    ) -> Result<(), DecodeError> {
        let mut timestamps_seen = 0;
        let device = audio_ctx.device().clone();

        for _ in 0..self.sample_len {
            let input_tokens = if tokens.len() > self.initial_tokens_len.unwrap() {
                &tokens[tokens.len() - 1..]
            } else {
                &tokens
            };
            let input_t =
                Tensor::from_data(input_tokens, shape![1, input_tokens.len()], device.clone());

            let mut logits = decoder.forward(&[audio_ctx.clone(), input_t])?;

            let token_t = Tensor::from_data(tokens.clone(), shape![1, tokens.len()], Device::CPU);
            for m in &self.logit_mutators {
                logits = m.apply(logits, &token_t)?;
            }

            let (_, new_tokens, completed) = GreedySampler::sample(tokens, logits)?;

            tokens = new_tokens;
            if completed {
                break;
            }
        }
        Ok(())
    }

    pub fn run(
        &self,
        decoder: &WhisperDecoder,
        audio_ctx: &Tensor,
        tokenizer: &WhisperTokenizer,
    ) -> Result<Vec<i32>, DecodeError> {
        let mut tokens = self.get_initial_tokens(tokenizer);
        let result = self.main_loop(decoder, audio_ctx.clone(), tokens.clone());

        tokens = tokens.drain(self.initial_tokens_len.unwrap()..).collect();
        let eot_index = tokens.iter().position(|x| *x == WhisperTokenizer::EOT);
        if let Some(eot_index) = eot_index {
            tokens.truncate(eot_index);
        }
        return Ok(tokens);
    }
}
