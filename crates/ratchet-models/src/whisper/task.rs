use crate::DecodingOptions;
use crate::LogitMutator;
use crate::Prompt;
use crate::WhisperTokenizer;
use crate::CHUNK_LENGTH;
use crate::N_AUDIO_CTX;

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

        task.logit_mutators.push(Box::new(ApplyTimestampRules {
            sample_begin: task.initial_tokens_len.unwrap(),
            max_initial_timestamp_index,
        }));
        if let Some(suppress_tokens) = &task.options.suppress_tokens {
            let mut suppress_tokens = suppress_tokens.clone();
            if suppress_tokens.contains(&-1) {
                let neg_one = suppress_tokens.iter().position(|x| *x == -1).unwrap();
                suppress_tokens.remove(neg_one);
                suppress_tokens.extend(WhisperTokenizer::NON_SPEECH);
            }
            task.logit_mutators
                .push(Box::new(SuppressNonSpeech { suppress_tokens }));
        }
        task
    }
}
